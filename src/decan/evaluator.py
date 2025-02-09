""" Module which handles all evaluations """

import os
from argparse import ArgumentParser
import yaml

import rich

import torch

import datasets
from transformers import AutoTokenizer
from transformers.utils import logging

from lm_eval.models.huggingface import HFLM, eval_logger
from lm_eval.evaluator import simple_evaluate

from model import DeCANConfig, DeCANForCausalLM

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

def eval_suite( lm: HFLM, suite: str, verbosity: int ) -> list[tuple[str, float]]:
    """ Evaluates an LM on all tasks in a suite, yielding a list of corresponding metrics and a macro average.

    Args:
        lm (HFLM): Language model instance.
        suite (str): Name of the evaluation suite
        verbosity (int): Verbosity level.

    Returns:
        list[tuple[str, float]]: List of metric name-value pairs, including the macro average.
    """
    
    tasks = TASKS[ suite ]

    score_list: list[tuple[str, float]] = []

    for task_name, num_fewshot, metric_name in tasks:
        task_score = eval_task( lm, task_name, num_fewshot, metric_name, verbosity )
        score_list.append( ( f'{task_name} ({num_fewshot}-shot)', task_score ) )
        rich.print( f'{task_name} ({num_fewshot}-shot) - {task_score * 100:.2f}%' )

    average_score = sum( i[1] for i in score_list ) / len( score_list )
    score_list.append( ( f'{suite} average', average_score ) )
    rich.print( f'{suite} average - {average_score * 100:.2f}%' )

    return score_list

def eval_task( lm: HFLM, task_name: str, num_fewshot: int, metric_name: str, verbosity: int ) -> float:
    """ Evals an LM on a single task, yielding a specific metric.

    Args:
        lm (HFLM): Language model instance.
        task_name (str): Task name.
        num_fewshot (int): Number of shots for eval.
        metric_name (str): Metric name to yield.
        verbosity (int): Verbosity level.

    Returns:
        float: Selected metric. 
    """
    
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

    if verbosity > 1:
        rich.print( eval_results )
    elif verbosity == 1:
        rich.print( { f'{task_name} ({num_fewshot}-shot)/{metric_name}': eval_results[task_name][metric_name] } )

    return eval_results[task_name][metric_name]


def local_model_setup( checkpoint_dir: str ) -> HFLM:
    """ Instantiates a local DeCAN model.

    Args:
        checkpoint_dir (str): Path to the checkpoint directory.

    Returns:
        HFLM: The instantiated and wrapped LM.
    """
    
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

def online_model_setup( model_name: str ) -> HFLM:
    """ Instantiates an LM from the HF hub.

    Args:
        model_name (str): Model name identifier.

    Returns:
        HFLM: The instantiated and wrapped LM.
    """
    return HFLM( pretrained=model_name, batch_size=1, backend='causal' )


def print_csv( results: dict[str, list[tuple[str, float]]] ):
    """ Converts the results dict into CSV lines and prints them out.

    Args:
        results (dict[str, list[tuple[str, float]]]): Results dict produced by the eval loop.
    """
    
    columns = [ 'model' ] + [ i[0] for i in list( results.values() )[0] ]
    data = []

    for model_name, metric_list in results.items():
        curr_data = [ model_name ] + [ i[1] for i in metric_list ]
        data.append( curr_data )

    print( ','.join( columns ) )
    for line in data:
        print( ','.join( [ str( i ) for i in line ] ) )


def run( arguments: dict ):
    """ Runs the eval with the parsed arguments.

    Args:
        arguments (dict): Dict of namespace.
    """

    # Set some performance flags
    torch.backends.cuda.matmul.allow_tf32 = True # type: ignore # pylint: disable=W0212
    torch.backends.cudnn.allow_tf32 = True # type: ignore # pylint: disable=W0212
    torch._dynamo.config.cache_size_limit = 1024 * 1024 * 1024 # type: ignore # pylint: disable=W0212
    torch._dynamo.config.optimize_ddp = True # type: ignore # pylint: disable=W0212
    torch.backends.cuda.enable_cudnn_sdp( False )
    torch.backends.cuda.enable_math_sdp( False )

    # Set some HF flags
    datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True # type: ignore
    datasets.disable_progress_bar()
    eval_logger.setLevel( 'ERROR' )

    # Santiy check, print args
    rich.print( arguments )

    # Get the suite list
    suite_list = arguments[ 'suite' ]

    # Get the verbosity level
    verbosity = arguments[ 'verbosity' ]

    # Parse either local or online
    match arguments[ 'group' ]:
        case 'local':
            output_dir = arguments[ 'output_dir' ]
            manifest_file = arguments[ 'manifest_file' ]
            run_names = arguments[ 'run_name' ]
            checkpoint_subdir = arguments[ 'checkpoint_subdir' ]

            # Check mutual exclusion
            if not ( ( manifest_file is not None ) ^ ( run_names is not None ) ):
                raise ValueError( 'Exactly ONE of `manifest_file` or `run_name` must be specified!' )

            # If we have a manigest file set run names from contents
            if arguments[ 'manifest_file' ] is not None:
                manifest_file_path = os.path.join( output_dir, manifest_file )
                
                with open( manifest_file_path, 'r', encoding='utf-8' ) as f:
                    run_dict = yaml.load( f, yaml.FullLoader )
                    assert isinstance( run_dict, dict )
                run_names = run_dict[ 'runs' ]

            # Join directories
            run_dirs = [ os.path.join( output_dir, run, checkpoint_subdir ) for run in run_names ]

            # Build a tuple of model names and model instances
            model_tuples: list[tuple[str, HFLM]] = [
                ( run_name, local_model_setup( run_dir ) )
                for run_name, run_dir in zip( run_names, run_dirs )
            ]

        case 'online':
            hf_names = arguments[ 'hf_model' ]

            # Build a tuple of model names and model instances
            model_tuples: list[tuple[str, HFLM]] = [
                ( hf_name, online_model_setup( hf_name ) )
                for hf_name in hf_names
            ]

        case _:
            raise ValueError( 'Must specify either `local` or `online`!' )

    # I hate this format
    results: dict[str, list[tuple[str, float]]] = {}

    # Run evals for all LMs on all suites.
    for name, lm in model_tuples:
        metrics: list[tuple[str, float]] = []
        for suite in suite_list:
            metrics += eval_suite( lm, suite, verbosity )
        results[name] = metrics

    print_csv( results )

def setup():
    """ Setup function for evaluation. Parses command line arguments, instantiates models and performs evals. """
    
    # CLI argument parser
    parser = ArgumentParser()

    # Add mode argument
    parser.add_argument(
        'mode',
        choices=[ 'evaluate' ],
        help='Evaluation mode. Currently only supports `evaluate`.'
    )

    sub_parser = parser.add_subparsers( dest='group', required=True )

    # Create groups
    local_group = sub_parser.add_parser( 'local', help='Evaluate one or more local models. Use `local -h` for full list of args.' )
    online_group = sub_parser.add_parser( 'online', help='Evaluate one or more HF models. Use `online -h` for full list of args.' )

    # Add argument for specifying model directories
    local_group.add_argument(
        '--output_dir',
        '--output-dir',
        required=True,
        type=lambda x: os.path.expanduser( x.format( **os.environ ) ),
        help='The output directory where runs are stored. May be a formatted string to use envars.',
        dest='output_dir',
    )

    # Add argument for specifying manifest_file_path
    local_group.add_argument(
        '--manifest_file',
        '--manifest-file',
        nargs='?',
        help='Name of manifest yaml file to parse runs from. Mutually exclusive with `run_name`.',
        dest='manifest_file',
    )

    # Add argument for specifying run name(s)
    local_group.add_argument(
        '--run_name',
        '--run-name',
        nargs='*',
        help='The name(s) of run(s) in output_dir to evaluate. Mutually exclusive with `manifest_file`.',
        dest='run_name',
    )

    # Add argument for specifying run subdir
    local_group.add_argument(
        '--checkpoint_subdir',
        '--checkpoint-subdir',
        nargs='?',
        default='checkpoint_final',
        help='The subdir inside a run to load the checkpoint from. Defaults to \'checkpoint_final\'.',
        dest='checkpoint_subdir',
    )

    # Add argument for specifying HF model(s)
    online_group.add_argument(
        '--hf_model',
        '--hf-model',
        required=True,
        nargs='+',
        help='Online HF model(s) to evaluate. Mutually exclusive with all local model arguments',
        dest='hf_model',
    )

    # Add common args
    for group in [ local_group, online_group ]:
        group.add_argument(
            '--suite',
            nargs='+',
            default=[ 'commonsense', 'leaderboard' ],
            choices=[ 'commonsense', 'leaderboard' ],
            help='The task suites to evaluate models on.',
            dest='suite',
        )

        group.add_argument(
            '--verbosity',
            nargs='?',
            type=int,
            default=1,
            choices=[ 0, 1, 2 ],
            help='Verbosity level of task printing. 0 for no printing, 1 prints the extracted metrics only, 2 prints all metrics.',
            dest='verbosity',
        )

    # Parse arguments as dict
    arguments = parser.parse_args().__dict__

    # Run eval!
    run( arguments )