
from datetime import timedelta
import os
import shutil
import json
from json import JSONDecodeError


from langdetect import DetectorFactory, detect, LangDetectException

import torch
from torch.distributed import TCPStore # type: ignore
from torch.utils.data import IterableDataset, DataLoader

from transformers import PreTrainedTokenizerBase
from datasets import DownloadConfig, load_dataset, disable_progress_bar


class PileClientDataset( IterableDataset ):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        seq_length: int,
        worker_batch_size: int,
        starting_shard: int,
        server_ip: str,
        server_port: int,
        num_procs: int,
        local_worker_id: int,
        world_size: int,
        world_rank: int,
    ):
        """ SlimPajama iterable dataset for a single process on a single device.

        Args:
            tokenizer (PreTrainedTokenizerBase): Text tokenizer.
            seq_length (int): Number of tokens per sample in sequence.
            worker_batch_size (int): Sub batch size for this process on this device.
            starting_shard (int): Initial common corpus shard to start from.
            server_ip (str): TCPStore ip address.
            server_port (int): TCPStore port number.
            num_procs (int): Total number of processes running on this device.
            local_worker_id (int): Local worker id of this process (NOT global worker id).
            world_size (int): Total number of devices in the group.
            world_rank (int): Global device rank.
        """
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.worker_batch_size = worker_batch_size
        self.starting_shard = starting_shard + local_worker_id + num_procs * world_rank

        self.server_ip = server_ip
        self.server_port = server_port
        self.num_procs = num_procs
        
        # Compute global worker id from the local worker id and the world rank
        self.global_worker_id = local_worker_id + num_procs * world_rank
        
        self.world_size = world_size
        self.world_rank = world_rank

    @classmethod
    def set_current_shard( cls, global_worker_id: int, client_store: TCPStore, value: int ):
        """ Sets the current shard number. Only effective on the first worker of the first device.
        
        Args:
            value (int): Zero indexed shard number.
        """
        if global_worker_id == 0:
            client_store.set( 'current_shard', str( value ) )

    @classmethod
    def get_url_from_shard( cls, index: int ) -> str:
        """ Computes the HF url from an integer shard index.

        Args:
            index (int): Zero indexed shard number.

        Returns:
            str: Huggingface URL string.
        """
        if index >= 30:
            raise ValueError( f'Shard index must be less than 30 but received {index}' )

        return '/data/lhk3/the_pile/{:02d}.jsonl'.format( index )

    @classmethod
    def line_iterator(
        cls,
        starting_shard,
        server_ip,
        server_port,
        num_procs,
        global_worker_id,
        world_size
    ):        
        client_store = TCPStore( server_ip, server_port, None, False, timeout=timedelta( seconds=30 ) )
        
        # Iterate from current shard until end of the dataset
        for current_shard in range( starting_shard, 30, num_procs * world_size ):
            # Update current shard number to resume later (only worker zero updates this value)
            cls.set_current_shard( global_worker_id, client_store, current_shard + num_procs * world_size )

            # Get the URL from the shard index
            url = cls.get_url_from_shard( current_shard )
            
            # Get the file path from the shard index
            path = cls.get_url_from_shard( current_shard )

            with open( path, 'rt', encoding="utf-8", buffering=1 ) as file:

                # Iterate over the entire parquet shard
                for i, line in enumerate( file ):
                    try:
                        obj = json.loads(line)
                        text = obj[ 'text' ]
                        yield text
                    except ( KeyError, JSONDecodeError ):
                        # print()
                        # print( f'shard={current_shard}, line={i}' )
                        pass
            
    @classmethod
    def batch_iterator(
        cls,
        tokenizer,
        seq_length,
        worker_batch_size,
        starting_shard,
        server_ip,
        server_port,
        num_procs,
        global_worker_id,
        world_size
    ):
        print( f'{seq_length=} {worker_batch_size=} {starting_shard=} {num_procs=} {global_worker_id=} {world_size=}')
        # Create iterator over all lines in all shards
        iterator = enumerate( cls.line_iterator( starting_shard, server_ip, server_port, num_procs, global_worker_id, world_size ) )

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

    def __iter__( self ):
        disable_progress_bar()
        
        return iter( self.batch_iterator(
            self.tokenizer,
            self.seq_length,
            self.worker_batch_size,
            self.starting_shard,
            self.server_ip,
            self.server_port,
            self.num_procs,
            self.global_worker_id,
            self.world_size
        ) )
    
    def __getitem__( self, index ):
        return NotImplementedError( 'This dataset does not support random access using __getitem__' )
    
    def as_data_loader( self ):
        return DataLoader(
            self,
            batch_size=None,
            num_workers=1,
            prefetch_factor=16,
        )

class PileDataset( IterableDataset ):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        seq_length: int,
        global_batch_size: int,
        starting_shard: int,
        server_ip: str,
        server_port: int,
        num_procs: int,
        world_size: int = 1,
        world_rank: int = 0,
    ):
        """ SlimPajama iterable dataset for all processes on a single device.

        Args:
            tokenizer (PreTrainedTokenizerBase): Text tokenizer.
            seq_length (int): Number of tokens per sample in sequence.
            global_batch_size (int): Full batch size across all devices, will automatically divide by world size.
            starting_shard (int): Initial common corpus shard to start from.
            server_ip (str): TCPStore ip address.
            server_port (int): TCPStore port number.
            num_procs (int): Total number of processes running on this device.
            world_size (int): Total number of devices in the group. Defaults to 1.
            world_rank (int): Global device rank. Defaults to 0.
        """
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.global_batch_size = global_batch_size
        self.starting_shard = starting_shard

        self.server_ip = server_ip
        self.server_port = server_port
        self.num_procs = num_procs
        
        self.world_size = world_size
        self.world_rank = world_rank
        
        if global_batch_size % ( num_procs * world_size ) != 0:
            raise ValueError( 'batch_size must be divisible by num_procs * world_size!' )

        self.master_store = TCPStore( self.server_ip, self.server_port, None, world_rank == 0, timeout=timedelta( seconds=30 ) )
        
        self.cleanup_cache()
    
    def cleanup_cache( self ):
        pass

    def set_current_shard( self, value: int ):
        """ Sets the current shard number. Only effective on the first device.
        
        Args:
            value (int): Zero indexed shard number.
        """
        if self.world_rank == 0:
            self.master_store.set( 'current_shard', str( value ) )

    def get_current_shard( self ) -> int:
        """ Gets the current shard from the first worker on the first device.
        Shard number should be fairly constant across all workers, but may differ slightly.
        
        Returns:
            int: Zero indexed shard number.
        """
        return int( self.master_store.get( 'current_shard' ) )
    
    def create_worker( self, idx: int ) -> DataLoader:
        """ Creates a worker DataLoader for this device.

        Args:
            idx (int): Local worker id of this process (NOT global worker id).

        Returns:
            DataLoader: DataLoader for the local worker index.
        """
        return PileClientDataset(
            tokenizer=self.tokenizer,
            seq_length=self.seq_length,
            worker_batch_size=self.global_batch_size // ( self.num_procs * self.world_size ),
            starting_shard=self.starting_shard,
            server_ip=self.server_ip,
            server_port=self.server_port,
            num_procs=self.num_procs,
            local_worker_id=idx,
            world_size=self.world_size,
            world_rank=self.world_rank,
        ).as_data_loader()
    
    def __iter__( self ):
        # Create a worker iterator for each local process
        workers = [ iter( self.create_worker( i ) ) for i in range( self.num_procs ) ]

        try:
            while True:
                test_next = [ next( i ) for i in workers ]
                test_next_x = torch.cat( [ i[0] for i in test_next ] )
                test_next_y = torch.cat( [ i[1] for i in test_next ] )
                test_next_d = torch.cat( [ i[2] for i in test_next ] )

                yield test_next_x, test_next_y, test_next_d
        except StopIteration:
            return
    
    def __getitem__( self, index ):
        return NotImplementedError( 'This dataset does not support random access using __getitem__' )

    def as_data_loader( self, pin_memory=True, pin_memory_device='cuda' ) -> DataLoader:
        """ Instantiates a multi-worker DataLoader for this device.

        Args:
            pin_memory (bool, optional): If `True`, the data loader will copy Tensors into device pinned memory before returning them. Defaults to True.
            pin_memory_device (str, optional): The device to pin memory to if `pin_memory` is enabled. Defaults to 'cuda'.

        Returns:
            DataLoader: The multi-worker DataLoader for this device.
        """
        return DataLoader(
            self,
            batch_size=None,
            pin_memory=pin_memory,
            pin_memory_device=pin_memory_device
        )
