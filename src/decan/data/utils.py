""" Utils module for data """

import torch
from transformers import PreTrainedTokenizerBase

def base_batch_iterator(
    iterator: enumerate,
    tokenizer: PreTrainedTokenizerBase,
    seq_length: int,
    worker_batch_size: int,
):
    # Create rolling queues for inputs, targets and document ids
    tokens_x_container = [ [] for _ in range( worker_batch_size ) ]
    tokens_y_container = [ [] for _ in range( worker_batch_size ) ]
    tokens_d_container = [ [] for _ in range( worker_batch_size ) ]

    # Compute if BOS/EOS are shared (False) or separate (True)
    separate_bos_eos = tokenizer.bos_token_id != tokenizer.eos_token_id

    # Loop forever
    while True:
        # Enter try-catch for StopIteration
        try:
            # Loop over all sub-batch entries
            for i in range( worker_batch_size ):
                
                # While sequence i is shorter than the sequence length keep appending documents
                while len( tokens_x_container[i] ) < seq_length:

                    # Get the document id and document text
                    document_id, text = next( iterator )

                    # Tokenize just the document with special tokens
                    tokens_raw = tokenizer.encode( text, add_special_tokens=False )

                    # Add separate BOS/EOS special tokens
                    if separate_bos_eos:
                        # inputs: [ BOS, ..., EOS ]
                        # target: [ ..., EOS, PAD ]
                        tokens_x = [ tokenizer.bos_token_id ] + tokens_raw + [ tokenizer.eos_token_id ]
                        tokens_y = tokens_raw + [ tokenizer.eos_token_id, tokenizer.pad_token_id ]

                    # Otherwise add shared BOS/EOS special tokens
                    else:
                        # inputs: [ EOS, ... ]
                        # target: [ ..., EOS ]
                        tokens_x = [ tokenizer.bos_token_id ] + tokens_raw
                        tokens_y = tokens_raw + [ tokenizer.eos_token_id ]

                    # Tile document id over length of sequence
                    documet_ids = [ document_id ] * len( tokens_x )

                    # Add inputs, targets and document ids to queues
                    tokens_x_container[i] += tokens_x
                    tokens_y_container[i] += tokens_y
                    tokens_d_container[i] += documet_ids

            # Create output sub-containers for the whole sub-batch
            output_x_container = [ [] for _ in range( worker_batch_size ) ]
            output_y_container = [ [] for _ in range( worker_batch_size ) ]
            output_d_container = [ [] for _ in range( worker_batch_size ) ]

            # Add the first seq_length tokens from the queue to the output, then dequeue
            for i in range( worker_batch_size ):
                output_x_container[i] = tokens_x_container[i][ : seq_length ]
                tokens_x_container[i] = tokens_x_container[i][ seq_length : ]

                output_y_container[i] = tokens_y_container[i][ : seq_length ]
                tokens_y_container[i] = tokens_y_container[i][ seq_length : ]

                output_d_container[i] = tokens_d_container[i][ : seq_length ]
                tokens_d_container[i] = tokens_d_container[i][ seq_length : ]

            # Yield the 3 token lists as Long Tensors
            yield torch.LongTensor( output_x_container ), torch.LongTensor( output_y_container ), torch.LongTensor( output_d_container )
        except StopIteration:
            return
