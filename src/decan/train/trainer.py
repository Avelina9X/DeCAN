""" Trainer module for training DeCAN models """

import os
from typing import Literal

import torch
import torch.distributed as dist

from transformers import AutoTokenizer

from model import DeCANConfig, DeCANForCausalLM
from model.utils import load_tokenizer, set_pretrained_embeddings

from data import CommonCorpusDataset

from .configuration_trainer import TrainerConfig

class DDPModelWrapper( torch.nn.parallel.DistributedDataParallel ):
    """ Custom DDP wrapper. Defers method and attribute accesses to underlying module. """
    def __getattr__( self, name ):
        try:
            return super().__getattr__( name )
        except AttributeError:
            return getattr( self.module, name )

class Trainer:
    def __init__( self, trainer_config: TrainerConfig, world_size: int, world_rank: int ):
        self.trainer_config = trainer_config
        self.world_size = world_size
        self.world_rank = world_rank

        if trainer_config.num_devices != world_size:
            raise ValueError( '`trainer_config.num_devices` is not equal to `world_size`!' )

        # Perform DDP setup
        if trainer_config.use_ddp:
            # Set cuda device rank
            torch.cuda.set_device( world_rank )

            # Set DDP communication envars
            os.environ[ 'MASTER_ADDR' ] = 'localhost'
            os.environ[ 'MASTER_PORT' ] = '12355'

            # Init process group
            dist.init_process_group( 'nccl', rank=world_rank, world_size=world_size )
        
        # First, load the model and tokenizer
        self.model: DeCANForCausalLM = DeCANForCausalLM.from_pretrained( trainer_config.curr_checkpoint_dir ).cuda() # type: ignore
        self.tokenizer = AutoTokenizer.from_pretrained( trainer_config.curr_checkpoint_dir )

        # Wrap model for DDP
        if trainer_config.use_ddp:
            self.model = DDPModelWrapper( self.model, device_ids=[ world_rank ] ) # type: ignore

        trainer_state_exists = os.path.exists( os.path.join( trainer_config.curr_checkpoint_dir, 'trainer_state.pt' ) )
        optimizer_state_exists = os.path.exists( os.path.join( trainer_config.curr_checkpoint_dir, 'optimizer_state.pt' ) )

        self.training_step = 0
        self.starting_shard = 0

        self.optimizer = self.create_optimizer()
        
        if not trainer_config.do_resume:
            if trainer_state_exists:
                raise ValueError( 'Trying to start from an initial checkpoint but `trainer_state.pt` already exists!' )
            if optimizer_state_exists:
                raise ValueError( 'Trying to start from an initial checkpoint but `optimizer_state.pt` already exists!' )

        else:
            if not trainer_state_exists:
                raise ValueError( 'Trying to resume from a checkpoint but `trainer_state.pt` does not exist!' )
            if not optimizer_state_exists:
                raise ValueError( 'Trying to resume from a checkpoint but `optimizer_state.pt` does not exist!' )

            self.load_trainer_state()
            self.load_optimizer_state()

        self.dataset = CommonCorpusDataset(
            tokenizer=self.tokenizer,
            seq_length=trainer_config.sequence_length,
            global_batch_size=trainer_config.global_batch_size,
            starting_shard=self.starting_shard,
            server_ip='localhost',
            server_port=15815,
            num_procs=trainer_config.num_workers_per_device,
            world_size=world_size,
            world_rank=world_rank,
        )

    def create_optimizer( self ):
        ...
            
    def load_trainer_state( self ):
        trainer_state_path = os.path.join( self.trainer_config.curr_checkpoint_dir, 'trainer_state.pt' )
        trainer_state = torch.load( trainer_state_path, weights_only=True )

        self.training_step = trainer_state[ 'training_step' ]
        self.starting_shard = trainer_state[ 'current_shard' ] + 1

    def load_optimizer_state( self ):
        optimizer_state_path = os.path.join( self.trainer_config.curr_checkpoint_dir, 'optimizer_state.pt' )

        ...

    def save_optimizer_state( self ):
        optimizer_state_path = os.path.join( self.trainer_config.curr_checkpoint_dir, 'optimizer_state.pt' )

        if self.trainer_config.use_zero_optimizer:
            ...

        # get state dict

        if self.world_rank == 0:
            ...

    def save_temp_checkpoint( self ):
        ...

    def save_perm_checkpoint( self ):
        ...

    def save_final_checkpoint( self ):
        ...

    def cleanup( self ):
        dist.destroy_process_group()
        self.dataset.cleanup_cache()
        
    @staticmethod
    def initialize( mode: Literal['new', 'start', 'resume'], trainer_kwargs: dict, model_kwargs: dict ):
        match mode:
            case 'new':
                # Initialize fresh configs
                trainer_config = TrainerConfig( **trainer_kwargs )
                model_config = DeCANConfig( **model_kwargs )

                if not trainer_config.do_init:
                    raise ValueError( 'Got `do_init=False` when trying to start a new run!' )

                # Load our modified tokenizer
                tokenizer = load_tokenizer()

                # Instantiate model
                model = DeCANForCausalLM( model_config )

                # Overwrite model embeddings
                set_pretrained_embeddings( model )

                # Now that everything is loaded, set do_init to False
                trainer_config.do_init = False

                # Get output save directory
                save_dir = trainer_config.curr_checkpoint_dir

                # Save all required files
                model.save_pretrained( save_dir )
                tokenizer.save_pretrained( save_dir, legacy_format=False )
                trainer_config.save_config( save_dir )

                return trainer_config

            case 'start':
                # Infer directory from trainer_kwargs
                load_dir = os.path.join(
                    os.path.expanduser( trainer_kwargs[ 'output_dir' ] ),
                    trainer_kwargs[ 'run_name' ],
                    'checkpoint_curr'
                )
                
                # Load config and update
                trainer_config = TrainerConfig.load_config( load_dir, trainer_kwargs )

                if trainer_config.do_resume:
                    raise ValueError( 'Got `do_resume=True` when trying to start a run!' )

                if trainer_config.do_init:
                    raise ValueError( 'Got `do_init=True` when trying to start a run!' )

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

                if not trainer_config.do_resume:
                    raise ValueError( 'Got `do_resume=False` when trying to resume a run!' )

                return trainer_config

            case _:
                raise ValueError( f"`mode` must be either 'new' or 'resume' but got {mode}" )
