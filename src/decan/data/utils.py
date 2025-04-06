""" Utils module for data """
import time
import urllib
import urllib.request
from urllib.error import URLError, HTTPError, ContentTooShortError
from http.client import HTTPException
from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch
from transformers import PreTrainedTokenizerBase

import zstandard

def base_batch_iterator(
    iterator: enumerate,
    tokenizer: PreTrainedTokenizerBase,
    seq_length: int,
    worker_batch_size: int,
):
    """ Returns an iterator of tokenized and document-masked batches for TransformerXL training.

    Args:
        iterator (enumerate): Iterator of document strings.
        tokenizer (PreTrainedTokenizerBase): Tokenizer object to tokenize the documents in batch.
        seq_length (int): Maximum sequence length for each batch.
        worker_batch_size (int): The batch size for this worker; should be an integer division of global batch size.

    Yields:
        tuple[LongTensor, LongTensor, LongTensor]: Three-tuple of long tensors representing the current batch;
            yields the (input_tokens, output_tokens, document_ids)
    """

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

def sft_batch_iterator(
    iterator: enumerate,
    tokenizer: PreTrainedTokenizerBase,
    seq_length: int,
    worker_batch_size: int,
):
    """ Returns an iterator of tokenized and document-masked batches for TransformerXL SFT.

    Args:
        iterator (enumerate): Iterator of document strings.
        tokenizer (PreTrainedTokenizerBase): Tokenizer object to tokenize the documents in batch.
        seq_length (int): Maximum sequence length for each batch.
        worker_batch_size (int): The batch size for this worker; should be an integer division of global batch size.

    Yields:
        tuple[LongTensor, LongTensor, LongTensor]: Three-tuple of long tensors representing the current batch;
            yields the (input_tokens, output_tokens, document_ids)
    """

    # Create rolling queues for inputs, targets and document ids
    tokens_x_container = [ [] for _ in range( worker_batch_size ) ]
    tokens_y_container = [ [] for _ in range( worker_batch_size ) ]
    tokens_d_container = [ [] for _ in range( worker_batch_size ) ]

    # Loop forever
    while True:
        # Enter try-catch for StopIteration
        try:
            # Loop over all sub-batch entries
            for i in range( worker_batch_size ):

                # While sequence i is shorter than the sequence length keep appending documents
                while len( tokens_x_container[i] ) < seq_length:

                    # Get the document id and document text
                    document_id, messages = next( iterator )

                    # Tokenize the messages
                    tensors = tokenizer.apply_chat_template(
                        conversation=messages,
                        return_assistant_tokens_mask=True,
                        return_dict=True,
                    )

                    # Shut up the linter
                    if TYPE_CHECKING:
                        assert isinstance( tensors, Mapping )

                    # Get the tokens and assistant mask
                    tokens_x = tensors[ 'input_ids' ]
                    mask = tensors[ 'assistant_masks' ]

                    # Shut up the linter
                    if TYPE_CHECKING:
                        assert isinstance( tokens_x, list )
                        assert isinstance( mask, list )

                    # Get left shift of inputs and apply masking
                    tokens_y = [ t if mask == 1 else tokenizer.pad_token_id for t, mask in zip( tokens_x[ 1 : ], mask[ 1 : ] ) ]

                    # Remove last additional token from inputs
                    tokens_x = tokens_x[ : -1 ]
                    
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

def request_retry( url: str, file_path: str, max_retries: int = 30 ):
    """ Implements `urllib.request.urlretrieve` with retries.
    Uses linear backoff of 10 seconds * retry count.

    Args:
        url (str): The URL of the file to download.
        file_path (str): Filepath to store downloaded file.
        max_retries (int, optional): _description_. Defaults to 30.
    """
    download_retries = 0
    while True:
        try:
            urllib.request.urlretrieve( url, file_path )
        except ( URLError, HTTPError, ContentTooShortError, HTTPException ) as err:
            download_retries += 1

            if download_retries > max_retries:
                raise err
            else:
                print( f'Download error. Retrying in {download_retries * 10} seconds.' )
                time.sleep( download_retries * 10 )
                continue
        else:
            break

def read_lines_zst( file_name: str ):
    """ Performs in memory compression and reads lines from a ZST file.

    Args:
        file_name (str): Path to ZST file

    Yields:
        str: Each individual line in the file
    """

    with open( file_name, 'rb' ) as file_handle:
        buffer = ''
        reader = zstandard.ZstdDecompressor( max_window_size=2**31 ).stream_reader( file_handle )
        while True:
            chunk = reader.read( 2**27 ).decode()
            if not chunk:
                break
            lines = ( buffer + chunk ).split( "\n" )

            for line in lines[ : -1 ]:
                yield line

            buffer = lines[-1]
        reader.close()