""" Utils module for data """

import torch
from transformers import PreTrainedTokenizerBase

def base_batch_iterator(
    iterator: enumerate,
    tokenizer: PreTrainedTokenizerBase,
    seq_length: int,
    worker_batch_size: int,
):
    tokens_x_container = [ [] for _ in range( worker_batch_size ) ]
    tokens_y_container = [ [] for _ in range( worker_batch_size ) ]
    tokens_d_container = [ [] for _ in range( worker_batch_size ) ]

    while True:
        try:
            for i in range( worker_batch_size ):
                while len( tokens_x_container[i] ) < seq_length:
                    document_id, document = next( iterator )
                    text = document

                    tokens_raw = tokenizer.encode( text, add_special_tokens=False )
                    tokens_x = [ tokenizer.bos_token_id ] + tokens_raw
                    tokens_y = tokens_raw + [ tokenizer.bos_token_id ]
                    documet_ids = [ document_id ] * len( tokens_x )

                    tokens_x_container[i] += tokens_x
                    tokens_y_container[i] += tokens_y
                    tokens_d_container[i] += documet_ids
            
            output_x_container = [ [] for _ in range( worker_batch_size ) ]
            output_y_container = [ [] for _ in range( worker_batch_size ) ]
            output_d_container = [ [] for _ in range( worker_batch_size ) ]

            for i in range( worker_batch_size ):
                output_x_container[i] = tokens_x_container[i][ : seq_length ]
                tokens_x_container[i] = tokens_x_container[i][ seq_length : ]

                output_y_container[i] = tokens_y_container[i][ : seq_length ]
                tokens_y_container[i] = tokens_y_container[i][ seq_length : ]

                output_d_container[i] = tokens_d_container[i][ : seq_length ]
                tokens_d_container[i] = tokens_d_container[i][ seq_length : ]
            
            yield torch.LongTensor( output_x_container ), torch.LongTensor( output_y_container ), torch.LongTensor( output_d_container )
        except StopIteration:
            return
