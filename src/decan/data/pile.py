
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

class LegacyPileShardDataset( IterableDataset ):
    """ Iterable Dataset for a single Pile shard.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        seq_length: int,
        shards_per_file: int,
        file_idx: int,
        dir_pattern: str,
    ):
        """
        Creates an iterable dataset for a single shard of the pile.

        Args:
            tokenizer (PreTrainedTokenizerBase): tokenizer to encode text.
            seq_length (int): desired sequence length.
            shards_per_file (int): number of shards to split iteration over.
            file_idx (int): id of the pile shard.
            dir_pattern (str): python format string for pile directory
        """
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.shards_per_file = shards_per_file
        self.file_idx = file_idx
        self.dir_pattern = dir_pattern

    @classmethod
    def tokenize_line( cls, line: str, tokenizer: PreTrainedTokenizerBase ):
        tokens = tokenizer.encode( line, add_special_tokens=False )
        tokens_x = [ tokenizer.bos_token_id ] + tokens
        tokens_y = tokens + [ tokenizer.eos_token_id ]

        for x, y in zip( tokens_x, tokens_y ):
            yield ( x, y )

    @classmethod
    def line_parser( cls, path: str, shard_num: int, shard_id: int ):
        with open( path, 'rt', encoding="utf-8", buffering=1 ) as file:
            for line_num, line in enumerate( file ):
                if ( line_num % shard_num ) == shard_id:
                    try:
                        text = json.loads( line )[ 'text' ]
                        yield text
                    except JSONDecodeError:
                        pass

    @classmethod
    def line_token_generator( cls, path: str, tokenizer: PreTrainedTokenizerBase, shard_num: int, shard_id: int ):
        for d, line in enumerate( cls.line_parser( path, shard_num, shard_id ) ):
            for x, y in cls.tokenize_line( line, tokenizer ):
                yield x, y, d

    @classmethod
    def sequence_generator( cls, path: str, tokenizer: PreTrainedTokenizerBase, shard_num: int, seq_length: int ):
        def reset():
            count = 0
            xs = []
            ys = []
            ds = []
            for _ in range( shard_num ):
                xs.append( [] )
                ys.append( [] )
                ds.append( [] )

            return count, xs, ys, ds

        count, xs, ys, ds = reset()

        generators = [ iter( cls.line_token_generator( path, tokenizer, shard_num, i ) ) for i in range( shard_num ) ]

        try:
            while True:
                for g_idx, generator in enumerate( generators ):
                    x, y, d = next( generator )
                    xs[ g_idx ].append( x )
                    ys[ g_idx ].append( y )
                    ds[ g_idx ].append( d )
                count += 1

                if count == seq_length:
                    yield ( torch.LongTensor( xs ), torch.LongTensor( ys ), torch.LongTensor( ds ) )

                    count, xs, ys, ds = reset()
        except StopIteration:
            return

    def __iter__( self ):
        return iter( self.sequence_generator(
            path=self.dir_pattern.format( self.file_idx ),
            tokenizer=self.tokenizer,
            shard_num=self.shards_per_file,
            seq_length=self.seq_length,
        ) )

    def as_data_loader( self ):
        return DataLoader(
            self,
            num_workers=1,
            batch_size=None,
            prefetch_factor=2,
        )

    def __getitem__( self, index ):
        raise NotImplementedError( "This dataset does not support random access using __getitem__" )

class LegacyPileDeviceDataset( IterableDataset ):
    """ Iterable Dataset for a multiple Pile shards.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        seq_length: int,
        batch_size: int,
        dir_pattern: str,
        pile_shards: list[int] | None=None
    ):
        """
        Creates an iterable dataset for multiple shards over the pile.

        Args:
            tokenizer (PreTrainedTokenizerBase): tokenizer to encode text.
            seq_length (int): desired sequence length.
            batch_size (int): desired local batch size.
            pile_shards (Optional[List[int]], optional): List of shard IDs to use, when None uses all 30.
            dir_pattern (str): python format string for pile directory.
        """
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.dir_pattern = dir_pattern
        self.pile_shards = pile_shards or list( range( 30 ) )

        if batch_size % len( self.pile_shards ) != 0:
            raise ValueError( 'batch size must be divisible by pile shard count' )

        self.shards_per_file = batch_size // len( self.pile_shards )

    def __iter__( self ):
        gen = [
            iter(
                LegacyPileShardDataset(
                    self.tokenizer,
                    self.seq_length,
                    self.shards_per_file,
                    i,
                    self.dir_pattern
                ).as_data_loader()
            ) for i in self.pile_shards
        ]

        try:
            while True:
                test_next = [ next( i ) for i in gen ]
                test_next_x = torch.cat( [ i[0] for i in test_next ] )
                test_next_y = torch.cat( [ i[1] for i in test_next ] )
                test_next_d = torch.cat( [ i[2] for i in test_next ] )

                yield test_next_x, test_next_y, test_next_d
        except StopIteration:
            return

    def as_data_loader( self ):
        return DataLoader(
            self,
            num_workers=0,
            batch_size=None,
            pin_memory=True,
            pin_memory_device='cuda',
        )


class LegacyPileDataset( IterableDataset ):
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
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.global_batch_size = global_batch_size
        self.starting_shard = starting_shard
        self.server_ip = server_ip
        self.server_port = server_port
        self.num_procs = num_procs
        self.world_size = world_size
        self.world_rank = world_rank

        self.local_batch_size = self.global_batch_size // self.world_size

        self.pile_shards = list( range( world_rank, 24, world_size ) )

    def as_data_loader( self, pin_memory=True, pin_memory_device='cuda' ) -> DataLoader:
        return DataLoader(
            self,
            num_workers=0,
            batch_size=None,
            pin_memory=pin_memory,
            pin_memory_device=pin_memory_device,
        )

    def get_current_shard( self ):
        return -1

    def __iter__( self ):
        return iter(
            LegacyPileDeviceDataset(
                tokenizer=self.tokenizer,
                seq_length=self.seq_length,
                batch_size=self.local_batch_size,
                dir_pattern='/data/lhk3/the_pile/{:02d}.jsonl',
                pile_shards=self.pile_shards,
            )
        )

    def __getitem__( self, index ):
        return NotImplementedError( 'This dataset does not support random access using __getitem__' )

    def cleanup_cache( self ):
        pass





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
        self.starting_shard = starting_shard

        self.server_ip = server_ip
        self.server_port = server_port
        self.num_procs = num_procs
        
        # Compute global worker id from the local worker id and the world rank
        self.global_worker_id = local_worker_id + num_procs * world_rank
        
        self.world_size = world_size
        self.world_rank = world_rank
        
        self.worker_cache_dir = os.path.join( os.environ[ 'HF_TEMP_DIR' ], 'slim_pajama', f'worker_{self.global_worker_id}' )

    
    def set_current_shard( self, value: int ):
        """ Sets the current shard number. Only effective on the first worker of the first device.
        
        Args:
            value (int): Zero indexed shard number.
        """
        if self.global_worker_id == 0:
            self.client_store.set( 'current_shard', str( value ) )
    
    def get_url_from_shard( self, index: int ) -> str:
        """ Computes the HF url from an integer shard index.

        Args:
            index (int): Zero indexed shard number.

        Returns:
            str: Huggingface URL string.
        """
        if index >= 30:
            raise ValueError( f'Shard index must be less than 30 but received {index}' )

        return '/data/lhk3/the_pile/{:02d}.jsonl'.format( index )
    
    def tokenize_line( self, iterator ):
        document = 0
        while True:
            # Get the next line from the shared iterator
            text = next( iterator )
            
            # Tokenize without additional special tokens
            tokens = self.tokenizer.encode( text, add_special_tokens=False )
            
            # Add special tokens needed for training
            tokens_x = [ self.tokenizer.eos_token_id ] + tokens
            tokens_y = tokens + [ self.tokenizer.eos_token_id ]

            # Yield ( input, target ) tokens one-by-one
            for x, y in zip( tokens_x, tokens_y ):
                yield ( x, y, document )
                document += 1
            
            # Cleanup because I don't trust generators
            del text
            del tokens_x
            del tokens_y

    def line_iterator( self ):        
        # Iterate from current shard until end of the dataset
        for current_shard in range( self.starting_shard, 30 ):
            # Update current shard number to resume later (only worker zero updates this value)
            self.set_current_shard( current_shard )

            # Get the file path from the shard index
            path = self.get_url_from_shard( current_shard )

            with open( path, 'rt', encoding="utf-8", buffering=1 ) as file:

                # Iterate over the entire parquet shard
                for i, line in enumerate( file ):
                    # Yield only if (iterator rank) % (number of workers in world) equals worker id
                    if i % ( self.num_procs * self.world_size ) == self.global_worker_id:
                        try:
                            obj = json.loads(line)
                            text = obj[ 'text' ]
                            yield text
                        except ( KeyError, JSONDecodeError ):
                            # print()
                            # print( f'shard={current_shard}, line={i}' )
                            pass

    def line_iterator_new( self, batch_id: int ):        
        # Iterate from current shard until end of the dataset
        for current_shard in range( self.starting_shard, 30 ):
            # Update current shard number to resume later (only worker zero updates this value)
            if batch_id == 0:
                self.set_current_shard( current_shard )

            # Get the file path from the shard index
            path = self.get_url_from_shard( current_shard )

            with open( path, 'rt', encoding="utf-8", buffering=1 ) as file:

                # Iterate over the entire parquet shard
                for i, line in enumerate( file ):
                    # Yield only if (iterator rank) % (number of workers in world) equals worker id
                    if i % ( self.num_procs * self.world_size * self.worker_batch_size ) == ( self.global_worker_id * self.worker_batch_size + batch_id ):
                        try:
                            obj = json.loads(line)
                            text = obj[ 'text' ]
                            yield text
                        except ( KeyError, JSONDecodeError ):
                            # print()
                            # print( f'shard={current_shard}, line={i}' )
                            pass
            
    def batch_iterator( self ):
        # Create iterator over all lines in all shards
        # iterator = iter( self.line_iterator() )

        # Reset function for token yielding
        def reset():
            count = 0
            xs = []
            ys = []
            ds = []
            for _ in range( self.worker_batch_size ):
                xs.append( [] )
                ys.append( [] )
                ds.append( [] )

            return count, xs, ys, ds

        # Initialise token count, inputs and targets
        count, xs, ys, ds = reset()

        # Create a tokenizer iterator for each line in the micro batch
        # generators = [ iter( self.tokenize_line( iterator ) ) for _ in range( self.worker_batch_size ) ]
        generators = [ iter( self.tokenize_line( iter( self.line_iterator_new( i ) ) ) ) for i in range( self.worker_batch_size ) ]

        try:
            while True:
                # Continuously add new tokens to the batch
                for g_idx, generator in enumerate( generators ):
                    x, y, d = next( generator )
                    xs[ g_idx ].append( x )
                    ys[ g_idx ].append( y )
                    ds[ g_idx ].append( d )
                count += 1

                # When maximum sequence length is reached, yield the micro batch and reset
                if count == self.seq_length:
                    yield ( torch.LongTensor( xs ), torch.LongTensor( ys ), torch.LongTensor( ds ) )

                    count, xs, ys, ds = reset()
        except StopIteration:
            return

    def __iter__( self ):
        disable_progress_bar()
        self.client_store = TCPStore( self.server_ip, self.server_port, None, False, timeout=timedelta( seconds=30 ) )
        
        for batch in iter( self.batch_iterator() ):
            yield batch
    
    def __getitem__( self, index ):
        return NotImplementedError( 'This dataset does not support random access using __getitem__' )
    
    def as_data_loader( self ):
        return DataLoader(
            self,
            batch_size=1,
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
                xs = []
                ys = []
                ds = []
                
                # Continuously add new micro batches to the sub batch
                for worker in workers:
                    x, y, d = next( worker )

                    xs.append( x[0] )
                    ys.append( y[0] )
                    ds.append( d[0] )
                
                # Yield sub batch when full
                x_out, y_out, d_out = torch.cat( xs ), torch.cat( ys ), torch.cat( ds )
                assert x_out.shape[0] == self.global_batch_size // self.world_size
                yield x_out, y_out, d_out
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
