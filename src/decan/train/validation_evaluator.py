
import os
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.modeling_outputs import CausalLMOutputWithPast
import datasets

from .utils import MeanMetric

class ValidationEvaluator:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        eval_batch_size: int,
        eval_max_len: int,
        world_size: int,
        world_rank: int,
    ):
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

        self.dataset_shard = self.load_dataset_shard()

        self.metrics = {
            'loss': MeanMetric( world_size > 1 ),
            'acc': MeanMetric( world_size > 1 ),
            'ppl': MeanMetric( world_size > 1 ),
        }

    def load_dataset_shard( self ) -> datasets.Dataset:
        raise NotImplementedError( 'Must implement dataset shard loader in all subclasses!' )

    def dataset_name( self ) -> str:
        raise NotImplementedError( 'Must implement dataset shard loader in all subclasses!' )

    def __iter__( self ):
        for sample in self.dataset_shard:
            assert isinstance( sample, dict )
            yield torch.LongTensor( sample[ 'tokens' ] ), torch.LongTensor( sample[ 'targets' ] )

    @torch.compile
    def forward_pass( self, tokens, targets ):
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

    def eval( self ):
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
