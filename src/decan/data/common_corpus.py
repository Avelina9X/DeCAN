
from datetime import timedelta
import os
import shutil

import torch
from torch.distributed import TCPStore # type: ignore
from torch.utils.data import IterableDataset, DataLoader

from transformers import PreTrainedTokenizerBase
from datasets import DownloadConfig, load_dataset

class CommonCorpusClientDataset( IterableDataset ):
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
        """ CommonCorpus iterable dataset for a single process on a single device.

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

        self.client_store = TCPStore( self.server_ip, self.server_port, None, False, timeout=timedelta( seconds=30 ) )
        
        self.worker_cache_dir = os.path.join( os.environ[ 'HF_TEMP_DIR' ], 'common_corpus', f'worker_{self.global_worker_id}' )
    
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
        if index >= 10_000:
            raise ValueError( f'Shard index must be less than 10,000 but received {index}' )
        
        i = index % 10
        j = ( index // 10 ) % 100
        k = ( index // 1000 ) % 10

        return f'https://huggingface.co/datasets/PleIAs/common_corpus/resolve/main/common_corpus_{k+1}/subset_{j+1}_{i+1}.parquet'
    
    def tokenize_line( self, iterator ):
        document = 0
        while True:
            # Get the next line from the shared iterator
            text = next( iterator )
            
            # Tokenize without additional special tokens
            tokens = self.tokenizer.encode( text, add_special_tokens=False )
            
            # Add special tokens needed for training
            tokens_x = [ self.tokenizer.bos_token_id ] + tokens + [ self.tokenizer.eos_token_id ]
            tokens_y = tokens + [ self.tokenizer.eos_token_id, self.tokenizer.bos_token_id ]

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
        for current_shard in range( self.starting_shard, 10_000 ):
            # Update current shard number to resume later (only worker zero updates this value)
            self.set_current_shard( current_shard )

            # Get the URL from the shard index
            url = self.get_url_from_shard( current_shard )
            
            # Download the parquet shard to a worker-indexed temporary cache
            dataset = load_dataset(
                'parquet',
                data_files=url,
                cache_dir=self.worker_cache_dir,
                streaming=False,
                download_config=DownloadConfig( max_retries=256 ),
                split='train'
            )

            # Iterate over the entire parquet shard
            for i, line in enumerate( dataset ):
                assert isinstance( line, dict )
                
                # Yield only if (iterator rank) % (number of workers in world) equals worker id
                if i % ( self.num_procs * self.world_size ) == self.global_worker_id:
                    yield line[ 'text' ]
            
            # Delete in memory dataset and in cache dataset
            del dataset
            shutil.rmtree( self.worker_cache_dir )
            
    def batch_iterator( self ):
        # Create iterator over all lines in all shards
        iterator = iter( self.line_iterator() )

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
        generators = [ iter( self.tokenize_line( iterator ) ) for _ in range( self.worker_batch_size ) ]

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

class CommonCorpusDataset( IterableDataset ):
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
        """ CommonCorpus iterable dataset for all processes on a single device.

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
        
        self.master_cache_dir = os.path.join( os.environ[ 'HF_TEMP_DIR' ], 'common_corpus' )
        
        self.cleanup_cache()
    
    def cleanup_cache( self ):
        """ Cleans up the temporary dataset cache. Only effective on the first device in the world.
        TODO: make changes for multi system cache cleanup
        """
        if self.world_rank == 0:
            os.makedirs( self.master_cache_dir, exist_ok=True )
            shutil.rmtree( self.master_cache_dir )

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
        return CommonCorpusClientDataset(
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
                yield torch.cat( xs ), torch.cat( ys ), torch.cat( ds )
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
