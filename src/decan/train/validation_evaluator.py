""" Validation Evaluator class. Handles train-time validation set eval. """

import os
from abc import ABC, abstractmethod

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.modeling_outputs import CausalLMOutputWithPast
import datasets

from .utils import MeanMetric


class ValidationEvaluator( ABC ):
    """ Base class for all ValidationEvaluator classes. """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        eval_batch_size: int,
        eval_max_len: int,
        world_size: int,
        world_rank: int,
    ):
        """ Instantiates an evaluator.

        Args:
            model (PreTrainedModel): Model to use for evaluation.
            tokenizer (PreTrainedTokenizerBase): Tokenizer for pre-tokenizing the dataset.
            eval_batch_size (int): Batch size to use for evaluation.
            eval_max_len (int): Length to pad or truncate all evaluation samples to.
            world_size (int): World size of the training run. Used to split dataset into shards.
            world_rank (int): World rank of the current process. Used to select the dataset shard.

        Raises:
            ValueError: Raised when the tokenizer and model config disagree on special token IDs.
        """

        self.model = model
        self.tokenizer = tokenizer
        self.eval_batch_size = eval_batch_size
        self.eval_max_len = eval_max_len
        self.world_size = world_size
        self.world_rank = world_rank

        # Check model and tokenizer agree on special token IDs
        for id_name in [ 'bos', 'eos', 'pad', 'sep', 'cls' ]:
            full_id_name = f'{id_name}_token_id'
            model_id = getattr( self.model.config, full_id_name )
            tokenizer_id = getattr( self.tokenizer, full_id_name )

            if model_id != tokenizer_id:
                raise ValueError( f'Special token ID missmatch! Got `model.config.{full_id_name}={model_id}` and `tokenizer.{full_id_name}={tokenizer_id}`' )

        # Load shard for this rank
        self.dataset_shard = self.load_dataset_shard()

        # Create metrics with DDP awareness if needed
        self.metrics = {
            'loss': MeanMetric( world_size > 1 ),
            'acc': MeanMetric( world_size > 1 ),
            'ppl': MeanMetric( world_size > 1 ),
        }

    @abstractmethod
    def load_dataset_shard( self ) -> datasets.Dataset:
        """ Loads the dataset, performs tokenization, and returns the shard for the current rank.

        Returns:
            datasets.Dataset: Pre-tokenized, pre-batched and pre-sharded dataset.
        """
        raise NotImplementedError( 'Must implement dataset shard loader in all subclasses!' )

    @abstractmethod
    def dataset_name( self ) -> str:
        """ Returns the name of the dataset.
        
        Returns:
            str: Dataset name.
        """
        raise NotImplementedError( 'Must implement dataset name in all subclasses!' )

    def __iter__( self ):
        for sample in self.dataset_shard:
            assert isinstance( sample, dict )
            yield torch.LongTensor( sample[ 'tokens' ] ), torch.LongTensor( sample[ 'targets' ] )

    @torch.compile
    def forward_pass( self, tokens: torch.Tensor, targets: torch.Tensor ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Performs a compiled forward pass on a single batch, returning loss, accuracy and PPL.

        Args:
            tokens (torch.Tensor): Batched tensor of input IDs.
            targets (torch.Tensor): Batched tensor of target IDs.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple of [loss, acc, ppl] scalar tensors.
        """

        self.model.eval()
        with torch.no_grad():
            autocast_dtype = torch.bfloat16 if self.model.config.use_bfloat16 else torch.float16
            with torch.autocast( device_type='cuda', dtype=autocast_dtype ):
                model_outputs: CausalLMOutputWithPast = self.model(
                    input_ids=tokens,
                    use_cache=True,
                    output_hidden_states=False,
                    output_attentions=False,
                    return_dict=True,
                )

            logits = model_outputs.logits

            pad_token_id = self.model.config.pad_token_id or -100

            valid_tokens = ( targets != pad_token_id )
            valid_length = valid_tokens.float().sum( -1 ).clamp( min=1.0 )

            loss = torch.nn.functional.cross_entropy(
                input=logits.transpose( 2, 1 ).float(),
                target=targets,
                ignore_index=pad_token_id,
                reduction='none'
            ) * valid_tokens

            acc = ( logits.argmax( dim=-1 ) == targets ) * valid_tokens

            seq_loss = ( loss.sum( -1 ) / valid_length )
            seq_acc = ( acc.float().sum( -1 ) / valid_length )
            seq_ppl = torch.exp( seq_loss )

            out_loss = seq_loss.mean()
            out_acc = seq_acc.mean()
            out_ppl = seq_ppl.mean()
        return out_loss, out_acc, out_ppl

    def eval( self ) -> dict[str, float]:
        """ Performs the evaluation loop and returns validation metrics.

        Returns:
            dict[str, float]: loss, acc and ppl metrics.
        """

        for tokens, targets in self.__iter__():
            tokens = tokens.cuda()
            targets = targets.cuda()

            loss, acc, ppl = self.forward_pass( tokens, targets )
            self.metrics[ 'loss' ].update( loss )
            self.metrics[ 'acc' ].update( acc )
            self.metrics[ 'ppl' ].update( ppl )

        stats: dict[str, float] = {}
        for name, metric in self.metrics.items():
            stats[f'{self.dataset_name()}/{name}'] = float( metric.compute() )
            metric.reset()
        return stats


class OWT10kEvaluator( ValidationEvaluator ):
    """ ValidationEvaluator for the OpenWebText 10k subset. """

    def load_dataset_shard( self ) -> datasets.Dataset:
        def token_func( line ):
            tokens_raw = self.tokenizer.encode( line[ 'text' ], add_special_tokens=False )
            tokens_raw = [ self.tokenizer.bos_token_id ] + tokens_raw + [ self.tokenizer.eos_token_id ]

            padding_needed = max( 0, self.eval_max_len + 1 - len( tokens_raw ) )

            tokens_padded = ( tokens_raw + [ self.tokenizer.pad_token_id ] * padding_needed )[ : self.eval_max_len + 1 ]

            tokens_x = tokens_padded[ : -1 ]
            tokens_y = tokens_padded[ 1 : ]

            return { 'tokens': tokens_x, 'targets': tokens_y }

        datasets.disable_progress_bar()

        dataset = datasets.load_dataset( 'stas/openwebtext-10k', split='train', cache_dir=os.environ[ 'HF_CACHE_DIR' ] )
        dataset_shard = dataset.shard( self.world_size, self.world_rank ).map( token_func, remove_columns='text', keep_in_memory=True ) # type: ignore
        return dataset_shard.batch( self.eval_batch_size, drop_last_batch=True ) # type: ignore


    def dataset_name( self ) -> str:
        return 'OWT10k'
