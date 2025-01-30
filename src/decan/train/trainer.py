""" Trainer module for training DeCAN models """

from datetime import timedelta
import os
from os import devnull
import time
from typing import Literal
from contextlib import contextmanager,redirect_stderr,redirect_stdout

import tqdm
import wandb
import binpacking
import numpy as np

import torch
import torch.distributed as dist
from torch.amp.grad_scaler import GradScaler
from torch.optim import Optimizer, AdamW

import datasets
from transformers import AutoTokenizer
from transformers.utils import logging
from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.modeling_outputs import CausalLMOutputWithPast

from lm_eval.models.huggingface import HFLM, eval_logger
from lm_eval.evaluator import simple_evaluate

from model import DeCANConfig, DeCANForCausalLM
from model.utils import load_tokenizer, set_pretrained_embeddings
from model.modeling_decan import DeCANTrainingCache
from data import CommonCorpusDataset, SlimPajamaDataset, PileDataset, SmolLMCorpusDataset

from .utils import DDPModelWrapper, MeanMetric
from .configuration_trainer import TrainerConfig

from .evaluator import OWT10kEvaluator

@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

# Default DDP address and port
DEFAULT_DDP_ADDR = 'localhost'
DEFAULT_DDP_PORT = '12355'

# Default dataset TCPStore address and port
DEFAULT_TCP_ADDR = 'localhost'
DEFAULT_TCP_PORT = 15815

logger = logging.get_logger( __name__ )

class Trainer:
    """ DeCAN trainer class """
    
    def __init__( self, trainer_config: TrainerConfig, world_size: int, world_rank: int ):
        # Set some performance flags
        torch.backends.cuda.matmul.allow_tf32 = True # type: ignore # pylint: disable=W0212
        torch.backends.cudnn.allow_tf32 = True # type: ignore # pylint: disable=W0212
        torch._dynamo.config.cache_size_limit = 1024 * 1024 * 1024 # type: ignore # pylint: disable=W0212
        torch._dynamo.config.optimize_ddp = True # type: ignore # pylint: disable=W0212
        torch.backends.cuda.enable_cudnn_sdp( False ) # type: ignore # pylint: disable=W0212
        torch.backends.cuda.enable_math_sdp( False )
        
        datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True # type: ignore
        datasets.disable_progress_bar()
        eval_logger.setLevel( 'ERROR' )
    
        self.trainer_config = trainer_config
        self.world_size = world_size
        self.world_rank = world_rank
                
        # If wandb is disabled we're in debug mode, enable extra logging
        if self.world_rank == 0 and self.trainer_config.wandb_mode == 'disabled':
            torch._logging.set_logs(
                graph_breaks=True,
                recompiles=True,
            )

        if trainer_config.num_devices != world_size:
            raise ValueError( '`trainer_config.num_devices` is not equal to `world_size`!' )

        # Perform DDP setup
        if trainer_config.use_ddp:
            # Set cuda device rank
            torch.cuda.set_device( world_rank )

            # Set DDP communication envars
            os.environ[ 'MASTER_ADDR' ] = DEFAULT_DDP_ADDR
            os.environ[ 'MASTER_PORT' ] = DEFAULT_DDP_PORT

            # Init process group
            dist.init_process_group( 'nccl', rank=world_rank, world_size=world_size, timeout=timedelta( minutes=30 ) )
        
        # Load the model and tokenizer from disk
        self.model: DeCANForCausalLM = DeCANForCausalLM.from_pretrained( trainer_config.curr_checkpoint_dir ).cuda() # type: ignore
        self.tokenizer = AutoTokenizer.from_pretrained( trainer_config.curr_checkpoint_dir, use_fast=True )
        
        # Freeze frozen params
        for n, p in self.model.named_parameters():
            for exclude in self.trainer_config.frozen_params:
                if exclude in n:
                    p.requires_grad_( False )

        # Check model and tokenizer agree on special token IDs
        for id_name in [ 'bos', 'eos', 'pad', 'sep', 'cls' ]:
            full_id_name = f'{id_name}_token_id'
            model_id = getattr( self.model.config, full_id_name )
            tokenizer_id = getattr( self.tokenizer, full_id_name )

            if model_id != tokenizer_id:
                raise ValueError( f'Special token ID missmatch! Got `model.config.{full_id_name}={model_id}` and `tokenizer.{full_id_name}={tokenizer_id}`' )

        # Wrap model for DDP
        if trainer_config.use_ddp:
            self.model = DDPModelWrapper( self.model, device_ids=[ world_rank ] ) # type: ignore

        # Set initial training step and starting shard
        self.training_step = 0
        self.starting_shard = trainer_config.starting_shard

        # Create optimizer
        self.optimizer = self.create_optimizer()
        self.optimizer_scaler = GradScaler( enabled=not self.model.config.use_bfloat16 )

        # Check if trainer and optimizer states exist
        trainer_state_exists = os.path.exists( os.path.join( trainer_config.curr_checkpoint_dir, 'trainer_state.pt' ) )
        optimizer_state_exists = os.path.exists( os.path.join( trainer_config.curr_checkpoint_dir, 'optimizer_state.pt' ) )
        scaler_state_exists = os.path.exists( os.path.join( trainer_config.curr_checkpoint_dir, 'scaler_state.pt' ) )

        # If we are NOT resuming we do NOT want trainer and optimizer states to exist
        if not trainer_config.do_resume:
            if trainer_state_exists:
                raise ValueError( 'Trying to start from an initial checkpoint but `trainer_state.pt` already exists!' )
            if optimizer_state_exists:
                raise ValueError( 'Trying to start from an initial checkpoint but `optimizer_state.pt` already exists!' )
            if scaler_state_exists:
                raise ValueError( 'Trying to start from an initial checkpoint but `scaler_state.pt` already exists!' )

            self.trainer_config.do_resume = True
        
        # If we ARE resuming we DO want trainer and optimizer states to exist so we can load them
        else:
            if not trainer_state_exists:
                raise ValueError( 'Trying to resume from a checkpoint but `trainer_state.pt` does not exist!' )
            if not optimizer_state_exists:
                raise ValueError( 'Trying to resume from a checkpoint but `optimizer_state.pt` does not exist!' )
            if not scaler_state_exists:
                raise ValueError( 'Trying to resume from a checkpoint but `scaler_state.pt` does not exist!' )

            # Load trainer and optimizer states
            self.load_trainer_state()
            self.load_optimizer_state()
            self.load_scaler_state()

        # Load dataset with potentially updated starting shard
        self.dataset = self.create_dataset()

        # Create training metrics
        self.metrics = {
            'loss': MeanMetric( trainer_config.use_ddp ),
            'acc': MeanMetric( trainer_config.use_ddp ),
        }

        self.evaluator = OWT10kEvaluator(
            self.model,
            self.tokenizer,
            self.trainer_config.eval_batch_size,
            self.trainer_config.cache_length,
            self.world_size,
            self.world_rank,
        )

        self.eval_model = HFLM(
            pretrained=self.model,
            tokenizer=self.tokenizer,
            add_bos_token=True,
            dtype=torch.bfloat16 if self.model.config.use_bfloat16 else torch.float16,
            batch_size=self.trainer_config.eval_batch_size,
            backend='causal'
        )

        eval_task_distribution = {
            'hellaswag': 10042,
            'openbookqa': 500,
            'winogrande': 1267,
            'arc_easy': 2376,
            'arc_challenge': 1172,
            'boolq': 3270,
            'piqa': 1838,
        }

        self.eval_tasks = list( binpacking.to_constant_bin_number( eval_task_distribution, self.world_size )[ self.world_rank ].keys() ) # type: ignore

    def create_optimizer( self ) -> Optimizer:
        """ Returns the optimizer specified by the trainer config """
        
        # Get the decay parameters from model
        decay_parameters = get_parameter_names( self.model, [ *ALL_LAYERNORM_LAYERS, torch.nn.Embedding ] )
        decay_parameters = [ name for name in decay_parameters if 'bias' not in name ]
        for exclude in self.trainer_config.weight_decay_exclude:
            decay_parameters = [ name for name in decay_parameters if exclude not in name ]

        # Create param groups for weight decay and non weight decay
        param_groups = [
            {
                'params': [ p for n, p in self.model.named_parameters() if ( n in decay_parameters and p.requires_grad ) ],
                'weight_decay': self.trainer_config.weight_decay,
            },
            {
                'params': [ p for n, p in self.model.named_parameters() if ( n not in decay_parameters and p.requires_grad ) ],
                'weight_decay': 0,
            },
        ]

        # Get our optimizer class
        optimizer_cls: type[Optimizer] = {
            'adamw': AdamW
        }[ self.trainer_config.optimizer ]

        # Return wrapped ZeRO optimizer if enabled
        if self.trainer_config.use_zero_optimizer:
            from torch.distributed.optim import ZeroRedundancyOptimizer # pylint: disable=C0415
            return ZeroRedundancyOptimizer(
                param_groups,
                optimizer_cls,
                lr=0,
                **self.trainer_config.optimizer_kwargs,
            )
        
        # Return base optimizer
        return optimizer_cls(
            param_groups,
            lr=0,
            **self.trainer_config.optimizer_kwargs
        )

    def create_dataset( self ):
        """ Returns the dataset specified by the trainer config """
        
        match self.trainer_config.training_dataset:
            case 'pile':
                return PileDataset(
                    tokenizer=self.tokenizer,
                    seq_length=self.trainer_config.sequence_length,
                    global_batch_size=self.trainer_config.global_batch_size,
                    starting_shard=self.starting_shard,
                    server_ip=DEFAULT_TCP_ADDR,
                    server_port=DEFAULT_TCP_PORT,
                    num_procs=self.trainer_config.num_workers_per_device,
                    world_size=self.world_size,
                    world_rank=self.world_rank,
                )
            case 'slim_pajama':
                return SlimPajamaDataset(
                    tokenizer=self.tokenizer,
                    seq_length=self.trainer_config.sequence_length,
                    global_batch_size=self.trainer_config.global_batch_size,
                    starting_shard=self.starting_shard,
                    server_ip=DEFAULT_TCP_ADDR,
                    server_port=DEFAULT_TCP_PORT,
                    num_procs=self.trainer_config.num_workers_per_device,
                    world_size=self.world_size,
                    world_rank=self.world_rank,
                )
            case 'common_corpus':
                return CommonCorpusDataset(
                    tokenizer=self.tokenizer,
                    seq_length=self.trainer_config.sequence_length,
                    global_batch_size=self.trainer_config.global_batch_size,
                    starting_shard=self.starting_shard,
                    server_ip=DEFAULT_TCP_ADDR,
                    server_port=DEFAULT_TCP_PORT,
                    num_procs=self.trainer_config.num_workers_per_device,
                    world_size=self.world_size,
                    world_rank=self.world_rank,
                )
            case 'smollm_corpus':
                return SmolLMCorpusDataset(
                    tokenizer=self.tokenizer,
                    seq_length=self.trainer_config.sequence_length,
                    global_batch_size=self.trainer_config.global_batch_size,
                    starting_shard=self.starting_shard,
                    server_ip=DEFAULT_TCP_ADDR,
                    server_port=DEFAULT_TCP_PORT,
                    num_procs=self.trainer_config.num_workers_per_device,
                    world_size=self.world_size,
                    world_rank=self.world_rank,
                )
            case _:
                raise ValueError( f'Dataset {self.trainer_config.training_dataset} is not a valid choice' )

    def load_trainer_state( self ) -> None:
        """ Loads the strainer state from trainer_state.pt"""
        
        # Load state dict
        trainer_state_path = os.path.join( self.trainer_config.curr_checkpoint_dir, 'trainer_state.pt' )
        trainer_state = torch.load( trainer_state_path, weights_only=True )

        # Set training step and starting shard
        self.training_step = trainer_state[ 'training_step' ]
        self.starting_shard = trainer_state[ 'current_shard' ]

    def save_trainer_state( self ) -> None:
        """ Saves the trainer state to trainer_state.pt and trainer.json """

        # Save training_step and current_shard into state dict
        state_dict = {
            'training_step': self.training_step,
            'current_shard': self.dataset.get_current_shard()
        }

        # Save state dict and config to disk only on rank zero
        if self.world_rank == 0:
            trainer_state_path = os.path.join( self.trainer_config.curr_checkpoint_dir, 'trainer_state.pt' )
            torch.save( state_dict, trainer_state_path )
            self.trainer_config.save_config( self.trainer_config.curr_checkpoint_dir )

    def load_optimizer_state( self ) -> None:
        """ Loads optimizer state from optimizer_state.pt """

        # Load the state dict from disk and set optimizer state
        optimizer_state_path = os.path.join( self.trainer_config.curr_checkpoint_dir, 'optimizer_state.pt' )
        state_dict = torch.load( optimizer_state_path, weights_only=True )
        self.optimizer.load_state_dict( state_dict )

    def save_optimizer_state( self ) -> None:
        """ Saves optimizer state to optimizer_state.pt """

        # If we're using zero we need to consolidate to rank zero
        if self.trainer_config.use_zero_optimizer:
            self.optimizer.consolidate_state_dict( 0 ) # type: ignore

        # We can only save to disk if we're on rank zero
        if self.world_rank == 0:
            optimizer_state_path = os.path.join( self.trainer_config.curr_checkpoint_dir, 'optimizer_state.pt' )
            state_dict = self.optimizer.state_dict()
            torch.save( state_dict, optimizer_state_path )

    def load_scaler_state( self ) -> None:
        """ Loads the mixed precision scaler from scaler_state.pt """

        # Load the state dict from disk and set scaler state
        scaler_state_path = os.path.join( self.trainer_config.curr_checkpoint_dir, 'scaler_state.pt' )
        state_dict = torch.load( scaler_state_path, weights_only=True )
        self.optimizer_scaler.load_state_dict( state_dict )

    def save_scaler_state( self ) -> None:
        """ Saves the mixed precision scaler from scaler_state.pt """

        # We can only save to disk if we're on rank zero, but should we? There is no consolidation mechanic...
        if self.world_rank == 0:
            scaler_state_path = os.path.join( self.trainer_config.curr_checkpoint_dir, 'scaler_state.pt' )
            state_dict = self.optimizer_scaler.state_dict()
            torch.save( state_dict, scaler_state_path )

    def save_temp_checkpoint( self ) -> None:
        """ Saves, and maybe overwrites, a temporay checkpoint.
        This saves the:
        - trainer state (trainer_state.pt, trainer.json)
        - optimizer state (optimizer_state.pt)
        - scaler state (scaler_state.pt)
        - tokenizer files and config
        - model weights (FP32) and config
        """
        
        self.save_trainer_state()
        self.save_optimizer_state()
        self.save_scaler_state()
        
        save_dir = self.trainer_config.curr_checkpoint_dir
        self.model.save_pretrained( save_dir, is_main_process=self.world_rank==0 )
        self.tokenizer.save_pretrained( save_dir, is_main_process=self.world_rank==0 )

    def save_perm_checkpoint( self ) -> None:
        """ Saves a permanent checkpoint, indexed by the training step.
        This saves the:
        - tokenizer files and config
        - model weights (FP32) and config
        """
        
        save_dir = self.trainer_config.perm_checkpoint_dir( self.training_step )
        self.model.save_pretrained( save_dir, is_main_process=self.world_rank==0 )
        self.tokenizer.save_pretrained( save_dir, is_main_process=self.world_rank==0 )

    def save_final_checkpoint( self ) -> None:
        """ Saves the final model checkpoint once training is done.
        This saves the:
        - tokenizer files and config
        - model weights (BF16/FP16) and config
        """
        
        save_dtype = torch.bfloat16 if self.model.config.use_bfloat16 else torch.float16
        save_dir = self.trainer_config.final_checkpoint_dir
        self.model.to( dtype=save_dtype ).save_pretrained( save_dir, is_main_process=self.world_rank==0 ) # type: ignore
        self.tokenizer.save_pretrained( save_dir, is_main_process=self.world_rank==0 )

    def get_learning_rate( self ) -> float:
        """ Computes the learning rate at the current step and returns it as a float"""
        
        warmup_ratio = min( self.training_step / self.trainer_config.warmup_steps, 1.0 )
        warmup_lr = warmup_ratio * self.trainer_config.lr_max
        
        cooldown_ratio = min( max( self.training_step - self.trainer_config.warmup_steps, 0.0 ) / ( self.trainer_config.max_steps - self.trainer_config.warmup_steps ), 1.0 )
        cooldown_alpha = np.cos( cooldown_ratio * np.pi ) * 0.5 + 0.5
        cooldown_lr = self.trainer_config.lr_max * cooldown_alpha + self.trainer_config.lr_min * ( 1.0 - cooldown_alpha )

        return min( warmup_lr, cooldown_lr )

    def reset_metrics( self ) -> dict[str, float]:
        """ Resets all metrics in the state dict and returns their current values. """
        
        stats = {}
        for name, metric in self.metrics.items():
            stats[name] = float( metric.compute() )
            metric.reset()
        return stats

    def cleanup( self ) -> None:
        """ Runs cleanup at the end of training. """
        
        # Cleanup DDP processes if we're using DDP
        if self.trainer_config.use_ddp:
            dist.destroy_process_group()

        # Cleanup dataset cache
        self.dataset.cleanup_cache()

    def progress_bar( self, elapsed: float ) -> str:
        r""" Returns a TQDM progress bar string with training stats.

        Args:
            elapsed (float): Time elapsed since epoch start.

        Returns:
            str: Formatted TQDM string. Should be prepended by `\r` when printing to clear the previous progress bar.
        """
        return tqdm.tqdm.format_meter(
            n=( ( self.training_step - 1 ) % self.trainer_config.steps_per_epoch ) + 1,
            total=self.trainer_config.steps_per_epoch,
            elapsed=elapsed,
            ncols=100,
            unit='it',
            bar_format='{desc}: {percentage:.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}, {rate_fmt}{postfix}]',
            postfix=f"loss={self.metrics['loss'].compute():.3f}, acc={self.metrics['acc'].compute():.3f}, lr={self.get_learning_rate():.3e}",
            prefix=f'Step {self.training_step}'
        )

    def lm_eval( self ) -> dict[str, float]:
        self.model.eval()

        with suppress_stdout_stderr():
            eval_results = simple_evaluate(
                self.eval_model,
                tasks=self.eval_tasks,
                device='cuda',
                log_samples=False,
                batch_size=self.trainer_config.eval_batch_size,
                verbosity='ERROR',
                cache_requests='LM_HARNESS_CACHE_PATH' in os.environ
            )[ 'results' ] # type: ignore

        if self.world_size > 1:
            if self.world_rank > 0:
                dist.send_object_list( [ eval_results ], dst=0 )
            else:
                for rank in range( 1, self.world_size ):
                    rcv_list = [ {} ]
                    dist.recv_object_list( rcv_list, rank )
                    eval_results.update( rcv_list[0] )
        
        flat_metrics = {}

        for task, metrics in eval_results.items():
            for metric_name, value in metrics.items():
                if metric_name != 'alias' and '_stderr' not in metric_name:
                    flat_metrics[ f'{task}/{metric_name}' ] = value

        return flat_metrics

    def train( self ):
        # Create dataloader
        dataloader = self.dataset.as_data_loader()
        iterator = iter( dataloader )

        if self.world_rank == 0:
            wandb.login( key=os.environ[ 'WANDB_API_KEY' ] )
            wandb.init(
                mode=self.trainer_config.wandb_mode,
                project=self.trainer_config.wandb_project,
                group=self.trainer_config.wandb_group,
                tags=self.trainer_config.wandb_tags,
                name=self.trainer_config.run_name,
                config={
                    'trainer': self.trainer_config.to_wandb_dict(),
                    'model': self.model.config.to_diff_dict(),
                    'num_parameters': self.model.num_parameters( only_trainable=False ),
                    'num_parameters_non_embedding': self.model.num_parameters( only_trainable=False, exclude_embeddings=True ),
                    'num_parameters_trainable': self.model.num_parameters( only_trainable=True ),
                    'num_parameters_trainable_non_embedding': self.model.num_parameters( only_trainable=True, exclude_embeddings=True ),
                }
            )

        cache_list = [
            DeCANTrainingCache( max_cache_length=self.trainer_config.cache_length )
            for _ in range( self.trainer_config.gradient_accumulation_steps )
        ]

        # Create loop until session end or max_steps
        session_max_step = self.trainer_config.max_steps if self.trainer_config.epochs_per_session == -1 else self.training_step + self.trainer_config.steps_per_epoch * self.trainer_config.epochs_per_session

        start_time = time.time()
        for _ in range( self.training_step, session_max_step ):
            metrics = {}

            self.train_step( next( iterator ), cache_list )

            progress_bar = self.progress_bar( time.time() - start_time)
            if self.world_rank == 0:
                print( '\r' + progress_bar, end='', flush=True )

            do_eval = self.training_step % ( self.trainer_config.steps_per_epoch * self.trainer_config.validation_freq ) == 0
            do_temp_checkpoint = self.training_step % ( self.trainer_config.steps_per_epoch * self.trainer_config.temp_checkpoint_freq ) == 0
            do_perm_checkpoint = self.training_step % ( self.trainer_config.steps_per_epoch * self.trainer_config.perm_checkpoint_freq ) == 0
            do_final_checkpoint = self.training_step % self.trainer_config.max_steps == 0
            do_log = self.training_step % self.trainer_config.steps_per_epoch == 0

            if do_final_checkpoint:
                self.trainer_config.is_complete = True

            if do_eval:
                eval_metrics = { f'validation/{k}': v for k, v in self.evaluator.eval().items() }
                lm_eval_metrics = { f'validation/{k}': v for k, v in self.lm_eval().items() }
                metrics.update( eval_metrics )
                metrics.update( lm_eval_metrics )

            if do_log or do_eval or do_temp_checkpoint or do_perm_checkpoint or do_final_checkpoint:
                metrics.update( { f'train/{k}': v for k, v in self.reset_metrics().items() } )

                if self.world_rank == 0:
                    metrics.update( self.get_parameter_histograms() )
                    metrics.update( {
                        'stats/training_step': self.training_step,
                        'stats/num_tokens': self.training_step * self.trainer_config.global_batch_size * self.trainer_config.sequence_length,
                        'stats/learning_rate': self.get_learning_rate(),
                    } )
                    
                    wandb.log( metrics )
                    print()

                # Reset start time
                start_time = time.time()

            if do_temp_checkpoint or do_perm_checkpoint or do_final_checkpoint:
                self.save_temp_checkpoint()

                # Re-reset start time because saving takes time
                start_time = time.time()

            if do_perm_checkpoint or do_final_checkpoint:
                self.save_perm_checkpoint()

                # Re-reset start time because saving takes time
                start_time = time.time()

            if do_final_checkpoint:
                self.save_final_checkpoint()
                break

            if do_eval or do_temp_checkpoint or do_perm_checkpoint or do_final_checkpoint:
                torch.cuda.empty_cache()

        if self.world_rank == 0:
            wandb.finish()

    def train_step( self, batch, cache_list: list[DeCANTrainingCache] ):
        self.model.train()
        
        # Unpack batch
        tokens, targets, documents = batch

        tokens_list = torch.split( tokens.to( device='cuda', non_blocking=True ), self.trainer_config.micro_batch_size )
        targets_list = torch.split( targets.to( device='cuda', non_blocking=True ), self.trainer_config.micro_batch_size )
        documents_list = torch.split( documents.to( device='cuda', non_blocking=True ), self.trainer_config.micro_batch_size )

        for step in range( self.trainer_config.gradient_accumulation_steps ):
            curr_tokens = tokens_list[step]
            curr_targets = targets_list[step]
            curr_documents = documents_list[step] if self.trainer_config.document_masking else None
            curr_cache = cache_list[step]

            if self.trainer_config.use_ddp:
                grad_sync = step == ( self.trainer_config.gradient_accumulation_steps - 1 )
            else:
                grad_sync = None

            curr_cache.cache_to( device='cuda', non_blocking=True )
            loss, acc = self.train_micro_step( curr_tokens, curr_targets, curr_documents, curr_cache, grad_sync )
            curr_cache.pre_trim( self.trainer_config.sequence_length )
            curr_cache.detach_cache_to( device='cpu' if self.trainer_config.cpu_offload_cache else 'cuda', non_blocking=True )

            self.metrics[ 'loss' ].update( loss )
            self.metrics[ 'acc' ].update( acc )

        self.optimizer_step()

    def optimizer_step( self ):
        self.training_step += 1

        for p_group in self.optimizer.param_groups:
            p_group[ 'lr' ] = self.get_learning_rate()

        self.optimizer_scaler.unscale_( self.optimizer )
        torch.nn.utils.clip_grad_norm_( self.model.parameters(), self.trainer_config.max_grad_norm )

        self.optimizer_scaler.step( self.optimizer )
        self.optimizer_scaler.update()
        self.optimizer.zero_grad()

    @torch.compile
    def train_micro_step(
        self,
        curr_tokens: torch.Tensor,
        curr_targets: torch.Tensor,
        curr_documents: torch.Tensor | None,
        curr_cache: DeCANTrainingCache,
        grad_sync: bool | None
    ):
        if grad_sync is not None:
            self.model.require_backward_grad_sync = grad_sync # type: ignore
        
        autocast_dtype = torch.bfloat16 if self.model.config.use_bfloat16 else torch.float16
        with torch.autocast( device_type='cuda', dtype=autocast_dtype ):
            model_outputs: CausalLMOutputWithPast = self.model(
                input_ids=curr_tokens,
                document_ids=curr_documents,
                past_key_values=curr_cache,
                use_cache=True,
                output_hidden_states=False,
                output_attentions=False,
                return_dict=True,
            )

            logits = model_outputs.logits

            pad_token_id = self.model.config.pad_token_id or -100

            valid_tokens = ( curr_targets != pad_token_id )
            valid_length = valid_tokens.float().sum( -1 ).clamp( min=1.0 )

            loss = torch.nn.functional.cross_entropy(
                input=logits.transpose( 2, 1 ).float(),
                target=curr_targets,
                ignore_index=pad_token_id,
                reduction='none'
            ) * valid_tokens
            
            acc = ( logits.argmax( dim=-1 ) == curr_targets ) * valid_tokens

            loss = ( loss.sum( -1 ) / valid_length ).mean()
            acc = ( acc.float().sum( -1 ) / valid_length ).mean()
        self.optimizer_scaler.scale( loss / self.trainer_config.gradient_accumulation_steps ).backward()

        return loss.detach(), acc.detach()

    def get_parameter_histograms( self ):
        histograms = {}
        with torch.inference_mode():
            for name, p in self.model.named_parameters():
                if p.requires_grad:
                    histograms[ f"parameters/{name.replace( '.', '/' ) }" ] = wandb.Histogram( p.cpu().numpy() ) # type: ignore
        return histograms

    @staticmethod
    def initialize(
        init_mode: Literal['new', 'setup', 'resume'],
        trainer_kwargs: dict,
        model_kwargs: dict
    ) -> TrainerConfig:
        """ Initialization method to create or resume a training run.

        Must be called BEFORE spawning new processes with torch.multiprocessing.spawn.

        Args:
            init_mode (Literal['new', 'setup', 'resume']): Run initialisation mode produced by `TrainerConfig.parse_arguments(...)`
            trainer_kwargs (dict): Additional trainer arguments produced by `TrainerConfig.parse_arguments(...)`
            model_kwargs (dict): Additional model arguments produced by `TrainerConfig.parse_arguments(...)`

        Returns:
            TrainerConfig: The config object to be passed to Trainer.__init__() in spawned processes.
        """
        
        match init_mode:
            case 'setup':
                # Initialize fresh configs
                trainer_config = TrainerConfig( **trainer_kwargs )
                model_config = DeCANConfig( **model_kwargs )

                if not trainer_config.do_init:
                    raise ValueError( 'Got `do_init=False` when trying to setup a new run!' )

                # Load our modified tokenizer
                separate_bos_eos = model_config.bos_token_id != model_config.eos_token_id
                tokenizer = load_tokenizer( separate_bos_eos=separate_bos_eos )

                # Instantiate model and overwrite embeddings
                model = DeCANForCausalLM( model_config )
                set_pretrained_embeddings( model )

                # Now that everything is loaded, set do_init to False
                trainer_config.do_init = False

                # Get output save directory and create it
                save_dir = trainer_config.curr_checkpoint_dir
                os.makedirs( save_dir, exist_ok=False )

                # Save all required files
                model.save_pretrained( save_dir )
                tokenizer.save_pretrained( save_dir )
                trainer_config.save_config( save_dir )

                exit() # TODO: handle this more gracefully, maybe add logging?
                
            case 'new':
                # Initialize fresh configs
                trainer_config = TrainerConfig( **trainer_kwargs )
                model_config = DeCANConfig( **model_kwargs )

                if not trainer_config.do_init:
                    raise ValueError( 'Got `do_init=False` when trying to start a new run!' )

                # Load our modified tokenizer
                separate_bos_eos = model_config.bos_token_id != model_config.eos_token_id
                tokenizer = load_tokenizer( separate_bos_eos=separate_bos_eos )

                # Instantiate model and overwrite embeddings
                model = DeCANForCausalLM( model_config )
                set_pretrained_embeddings( model )

                # Now that everything is loaded, set do_init to False
                trainer_config.do_init = False

                # Get output save directory and create it
                save_dir = trainer_config.curr_checkpoint_dir
                os.makedirs( save_dir, exist_ok=False )

                # Save all required files
                model.save_pretrained( save_dir )
                tokenizer.save_pretrained( save_dir )
                trainer_config.save_config( save_dir )

                return trainer_config

            case 'resume':
                # Infer directory from trainer_kwargs
                load_dir = os.path.join(
                    os.path.expanduser( trainer_kwargs[ 'output_dir' ] ),
                    trainer_kwargs[ 'run_name' ],
                    'checkpoint_curr'
                )
                
                # Load config and update
                trainer_config = TrainerConfig.load_config( load_dir, trainer_kwargs )

                if not trainer_config.is_complete:
                    raise ValueError( 'Got `is_complete=True` when trying to resume a run!' )

                return trainer_config

            case _:
                raise ValueError( f"`mode` must be either 'new' or 'resume' but got {init_mode}" )
