
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

class PileShardDataset( IterableDataset ):
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

class PileDeviceDataset( IterableDataset ):
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
                PileShardDataset(
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
            PileDeviceDataset(
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
