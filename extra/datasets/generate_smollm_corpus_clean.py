import multiprocessing as mp
import argparse
import tqdm

import datasets
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--hf_cache_dir',
        type=str,
        required=True,
    )

    parser.add_argument(
        '--destination_repo',
        type=str,
        required=True,
    )

    parser.add_argument(
        '--local_dir',
        type=str,
        required=True,
    )

    return parser.parse_args()

def main():
    args = parse_args()

    HF_CACHE_DIR = args.hf_cache_dir
    DESTINATION_REPO = args.destination_repo
    OUTPUT_FMT = args.local_dir

    OUTPUT_NUM_SHARDS = 23698

    ds_python = load_dataset( 'Avelina/python-edu-cleaned', cache_dir=HF_CACHE_DIR )
    print( f'{ds_python=}' )

    ds_cosmo = load_dataset( 'HuggingFaceTB/smollm-corpus', 'cosmopedia-v2', cache_dir=HF_CACHE_DIR )
    print( f'{ds_cosmo=}' )

    ds_edu = load_dataset( 'HuggingFaceTB/smollm-corpus', 'fineweb-edu-dedup', cache_dir=HF_CACHE_DIR )
    print( f'{ds_edu=}' )

    assert isinstance( ds_python, datasets.DatasetDict )
    assert isinstance( ds_cosmo, datasets.DatasetDict )
    assert isinstance( ds_edu, datasets.DatasetDict )

    ds_python = ds_python.select_columns( 'text' )[ 'train' ]
    ds_cosmo = ds_cosmo.select_columns( 'text' )[ 'train' ]
    ds_edu = ds_edu.select_columns( 'text' )[ 'train' ]

    print( f'{ds_python=}' )
    print( f'{ds_cosmo=}' )
    print( f'{ds_edu=}' )

    if input( f'Type CONFIRM to generate dataset: ' ) == 'CONFIRM':
        for index in tqdm.tqdm( range( OUTPUT_NUM_SHARDS ) ):
            group = index // 1000
            
            curr_python = ds_python.shard( num_shards=OUTPUT_NUM_SHARDS, index=index, contiguous=False, keep_in_memory=True )
            curr_cosmo = ds_cosmo.shard( num_shards=OUTPUT_NUM_SHARDS, index=index, contiguous=False, keep_in_memory=True )
            curr_edu = ds_edu.shard( num_shards=OUTPUT_NUM_SHARDS, index=index, contiguous=False, keep_in_memory=True )

            curr_shard = datasets.concatenate_datasets( [ curr_python, curr_cosmo, curr_edu ] )
            curr_shard = curr_shard.shuffle( seed=index, keep_in_memory=True )

            curr_shard.to_json( OUTPUT_FMT.format( index=index, num_shards=OUTPUT_NUM_SHARDS, group=group ), batch_size=20000 )
    else:
        print( 'Aborting...' )

if __name__ == '__main__':
    main()