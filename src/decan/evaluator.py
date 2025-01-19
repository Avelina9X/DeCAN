""" Module which handles all evaluations """

import os

import rich

import torch

import datasets
from transformers import AutoTokenizer
from transformers.utils import logging

from lm_eval.models.huggingface import HFLM, eval_logger
from lm_eval.evaluator import simple_evaluate

from model import DeCANConfig, DeCANForCausalLM
from train import TrainerConfig

logger = logging.get_logger( __name__ )

TASKS = {
    'leaderboard': [
        ( 'arc_challenge', 25, 'acc_norm,none' ),
        ( 'hellaswag', 10, 'acc_norm,none' ),
        ( 'truthfulqa_mc2', 0, 'acc,none' ),
        ( 'mmlu', 5, 'acc,none' ),
        ( 'winogrande', 5, 'acc,none' ),
        ( 'gsm8k', 5, 'exact_match,strict-match' ),
    ],
    'commonsense': [
        ( 'hellaswag', 0, 'acc_norm,none' ),
        ( 'openbookqa', 0, 'acc_norm,none' ),
        ( 'winogrande', 0, 'acc,none' ),
        ( 'arc_easy', 0, 'acc_norm,none' ),
        ( 'arc_challenge', 0, 'acc_norm,none' ),
        ( 'boolq', 0, 'acc,none' ),
        ( 'piqa', 0, 'acc_norm,none' ),
    ]
}

def make_csv( metrics: list, metadata: dict | None ):
    csv = ''
    if metadata is not None:
        csv += f"name, {metadata['name']}\n"
        csv += f"dataset, {metadata['dataset']}\n"
        csv += f"starting_shard, {metadata['starting_shard']}\n"
    
    for name, value in metrics:
        csv += f"{name}, {value}\n"

    return csv

def eval_suite( lm: HFLM, suite: str ):
    tasks = TASKS[ suite ]

    score_list = []

    for task_name, num_fewshot, metric_name in tasks:
        task_score = eval_task( lm, task_name, num_fewshot, metric_name )

        rich.print( f'{task_name} ({num_fewshot}-shot) - {task_score * 100:.2f}%' )

        score_list.append( ( f'{task_name} ({num_fewshot}-shot)', task_score ) )

    average_score = sum( i[1] for i in score_list ) / len( score_list )

    rich.print( f'{suite} average - {average_score * 100:.2f}%' )
    
    score_list.append( ( f'{suite} average', average_score ) )

    return score_list

def eval_task( lm: HFLM, task_name: str, num_fewshot: int, metric_name: str ) -> float:
    with torch.inference_mode():
        eval_results = simple_evaluate(
            lm,
            tasks=[ task_name ],
            num_fewshot=num_fewshot,
            device='cuda',
            log_samples=False,
            batch_size=1,
            verbosity='ERROR',
            cache_requests='LM_HARNESS_CACHE_PATH' in os.environ
        )[ 'results' ] # type: ignore

    rich.print( eval_results )

    return eval_results[task_name][ f'{metric_name}' ]

def metadata_setup( checkpoint_dir: str ):
    trainer_path = os.path.join( checkpoint_dir, 'trainer.json' )
    trainer_exists = os.path.isfile( trainer_path )

    if trainer_exists:
        trainer_config = TrainerConfig.load_config( checkpoint_dir )

        return {
            'name': trainer_config.run_name,
            'dataset': trainer_config.training_dataset,
            'starting_shard': trainer_config.starting_shard,
        }
    
    return None

def model_setup( checkpoint_dir: str ) -> HFLM:
    # Set some performance flags
    torch.backends.cuda.matmul.allow_tf32 = True # type: ignore # pylint: disable=W0212
    torch.backends.cudnn.allow_tf32 = True # type: ignore # pylint: disable=W0212
    torch._dynamo.config.cache_size_limit = 1024 * 1024 * 1024 # type: ignore # pylint: disable=W0212
    torch._dynamo.config.optimize_ddp = True # type: ignore # pylint: disable=W0212
    torch.backends.cuda.enable_cudnn_sdp( False )
    
    datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True # type: ignore
    datasets.disable_progress_bar()
    eval_logger.setLevel( 'ERROR' )

    # Load the model and tokenizer from disk
    model: DeCANForCausalLM = DeCANForCausalLM.from_pretrained( checkpoint_dir ) # type: ignore
    tokenizer = AutoTokenizer.from_pretrained( checkpoint_dir, use_fast=True )
    model_config: DeCANConfig = model.config # type: ignore

    # Get the compute dtype
    compute_dtype = torch.bfloat16 if model_config.use_bfloat16 else torch.float16

    # Cast model to compute dtype and move to GPU
    model = model.to( dtype=compute_dtype, device='cuda' ).eval() # type: ignore

    # Wrap model for the eval harness
    eval_model = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        add_bos_token=True,
        dtype=compute_dtype,
        batch_size=1,
        backend='causal',
        max_length=model_config.max_position_embeddings
    )

    return eval_model

def run( checkpoint: str, suite_list: list[str] ):
    eval_model = model_setup( checkpoint )
    metadata = metadata_setup( checkpoint )

    metrics = []

    for suite in suite_list:
        metrics += eval_suite( eval_model, suite )

    csv = make_csv( metrics, metadata )

    rich.print( csv )
    
    return csv

def setup():
    checkpoints = [
        # './checkpoints/pretrain/small/Vanilla-Small_Wxqo/checkpoint_curr',
        # './checkpoints/pretrain/small/Vanilla-Small_4hQ9/checkpoint_curr',
        # './checkpoints/pretrain/small/DeCAN-Small_FuRZ/checkpoint_curr',
        # './checkpoints/pretrain/small/DeCAN-Small_VMBH/checkpoint_curr',
        # './checkpoints/pretrain/small/Vanilla-Small_Ywuw/checkpoint_curr',
        # './checkpoints/pretrain/small/DeCAN-Small_LbYJ/checkpoint_curr',
        # './checkpoints/pretrain/small/Vanilla-Small_KGuv/checkpoint_curr',
        # './checkpoints/pretrain/small/DeCAN-Small_KrAb/checkpoint_curr',
        # './checkpoints/pretrain/small/Vanilla-Small_UPMB/checkpoint_curr',
        # './checkpoints/pretrain/small/DeCAN-Small_FgWq/checkpoint_curr',
        # './checkpoints/pretrain/small/Vanilla-Small_Hdvj/checkpoint_curr',
        # './checkpoints/pretrain/small/DeCAN-Small_aeTZ/checkpoint_curr',
        # './checkpoints/pretrain/small/Vanilla-Small_V968/checkpoint_curr',
        # './checkpoints/pretrain/small/DeCAN-Small_Fydk/checkpoint_curr',
        './checkpoints/pretrain/small/Vanilla-Small_LpXm/checkpoint_curr',
    ]
    
    suite_list = [ 'commonsense', 'leaderboard' ]

    csvs = []

    for checkpoint in checkpoints:
        csvs.append( run( checkpoint, suite_list ) )

    for csv in csvs:
        print()
        rich.print( csv )
