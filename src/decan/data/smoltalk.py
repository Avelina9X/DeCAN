""" SmolLMCorpusCleaned dataset class """

import os

import torch
from torch.utils.data import IterableDataset, DataLoader

from transformers import PreTrainedTokenizerBase
from datasets import DownloadConfig, load_dataset
from datasets import Dataset as HFDataset

from .utils import sft_batch_iterator

class SmolTalkClientDataset( IterableDataset ):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        seq_length: int,
        worker_batch_size: int,
        smol: bool,
    ):
        """ SmolTalk iterable dataset for a single process on a single device.

        Smol-SmolTalk has approx 91,607,859 tokens with chat template.
        SmolTalk has approx 978,150,229 tokens with chat template.

        Args:
            tokenizer (PreTrainedTokenizerBase): Text tokenizer.
            seq_length (int): Number of tokens per sample in sequence.
            worker_batch_size (int): Sub batch size for this process on this device.
            smol (bool): When True uses SmolSmolTalk.
        """

        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.worker_batch_size = worker_batch_size
        self.smol = smol
    
    @classmethod
    def line_iterator( cls, smol: bool ):

        # Grab correct version of dataset
        dataset = load_dataset(
            path='HuggingFaceTB/smol-smoltalk' if smol else 'HuggingFaceTB/smoltalk',
            name=None if smol else 'all',
            cache_dir=os.environ[ 'HF_CACHE_DIR' ],
            download_config=DownloadConfig(
                max_retries=256,
                cache_dir=os.environ[ 'HF_CACHE_DIR' ],
                disable_tqdm=True,
            ),
            split='train'
        )

        assert isinstance( dataset, HFDataset )

        while True:
            for line in dataset.shuffle( keep_in_memory=True ):
                assert isinstance( line, dict )
                messages = line[ 'messages' ]
                assert isinstance( messages, list )
                yield messages

    @classmethod
    def batch_iterator(
        cls,
        tokenizer,
        seq_length,
        worker_batch_size,
        smol,
    ):
        # Create iterator over all lines in all shards
        iterator = enumerate( cls.line_iterator( smol ) )

        for batch in sft_batch_iterator( iterator, tokenizer, seq_length, worker_batch_size ):
            yield batch

    def __iter__( self ):
        return iter( self.batch_iterator(
            self.tokenizer,
            self.seq_length,
            self.worker_batch_size,
            self.smol
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

class SmolTalkDataset( IterableDataset ):
    """
    SmolTalkDataset IterableDataset which spawns multiple workers on a single device.
    """

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
        smol: bool = False,
    ):
        """ SmolTalkDataset iterable dataset for all processes on a single device.

        Args:
            tokenizer (PreTrainedTokenizerBase): Text tokenizer.
            seq_length (int): Number of tokens per sample in sequence.
            global_batch_size (int): Full batch size across all devices, will automatically divide by world size.
            starting_shard (int): Initial common corpus shard to start from (not used).
            server_ip (str): TCPStore ip address (not used).
            server_port (int): TCPStore port number (not used).
            num_procs (int): Total number of processes running on this device.
            world_size (int): Total number of devices in the group (not used). Defaults to 1.
            world_rank (int): Global device rank (not used). Defaults to 0.
            smol (bool): When True uses SmolSmolTalk. Defaults to False.
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

        self.smol = smol

    def cleanup_cache( self ):
        """ NoOP """

    def set_current_shard( self, value: int ):
        """ NoOP """

    def get_current_shard( self ) -> int:
        """ NoOP """
        return 0

    def create_worker( self, idx: int ) -> DataLoader:
        """ Creates a worker DataLoader for this device.

        Args:
            idx (int): Local worker id of this process (NOT global worker id).

        Returns:
            DataLoader: DataLoader for the local worker index.
        """
        return SmolTalkClientDataset(
            tokenizer=self.tokenizer,
            seq_length=self.seq_length,
            worker_batch_size=self.global_batch_size // ( self.num_procs * self.world_size ),
            smol=self.smol,
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

class SmolSmolTalkDataset( SmolTalkDataset ):
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
        super().__init__(
            tokenizer,
            seq_length,
            global_batch_size,
            starting_shard,
            server_ip, server_port,
            num_procs,
            world_size,
            world_rank,
            smol=True
        )