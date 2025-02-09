""" Benchmark Evaluator class. Handles train-time benchmark eval. """

import os
from os import devnull
from contextlib import contextmanager,redirect_stderr,redirect_stdout

import binpacking

import torch
import torch.distributed as dist

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from lm_eval.models.huggingface import HFLM
from lm_eval.evaluator import simple_evaluate

@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

class BenchmarkEvaluator:
    """ Base class for all BenchmarkEvaluator classes. """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        eval_batch_size: int,
        eval_max_len: int,
        world_size: int,
        world_rank: int,
    ):
        self.model = model
        self.eval_batch_size = eval_batch_size
        self.eval_max_len = eval_max_len
        self.world_size = world_size
        self.world_rank = world_rank
        
        self.eval_model = HFLM(
            pretrained=model,
            tokenizer=tokenizer, # type: ignore
            add_bos_token=True,
            dtype=torch.bfloat16 if model.config.use_bfloat16 else torch.float16,
            batch_size=eval_batch_size,
            backend='causal'
        )

        eval_task_distribution = {
            'hellaswag': 10042,
            'openbookqa': 500,
            'winogrande': 1267,
            'arc_easy': 2376,
            'arc_challenge': 1172,
            'boolq': 3270,
            'piqa': 1838,
        }

        self.eval_tasks = list( binpacking.to_constant_bin_number( eval_task_distribution, world_size )[ world_rank ].keys() ) # type: ignore

    def eval( self ) -> dict[str, float]:
        """ Runs evaluations using the LM Eval Harness on 1 or more GPU and gathers task metrics back to rank 0.
        Note that tasks themselves are distributed across GPUs, but not samples from each task. This may result
        in some GPUs waiting for others to finish eval, but gaurantees identical behaviour for all world sizes.

        Returns:
            dict[str, float]: Dict mapping metric name to metric value. Note stderr values are not tracked.
        """
        self.model.eval()

        with suppress_stdout_stderr():
            eval_results = simple_evaluate(
                self.eval_model,
                tasks=self.eval_tasks,
                device='cuda',
                log_samples=False,
                batch_size=self.eval_batch_size,
                verbosity='ERROR',
                cache_requests='LM_HARNESS_CACHE_PATH' in os.environ
            )[ 'results' ] # type: ignore

        if self.world_size > 1:
            if self.world_rank > 0:
                dist.send_object_list( [ eval_results ], dst=0 )
            else:
                for rank in range( 1, self.world_size ):
                    rcv_list = [ {} ]
                    dist.recv_object_list( rcv_list, rank )
                    eval_results.update( rcv_list[0] )

        flat_metrics = {}

        for task, metrics in eval_results.items():
            for metric_name, value in metrics.items():
                if metric_name != 'alias' and '_stderr' not in metric_name:
                    flat_metrics[ f'{task}/{metric_name}' ] = value

        return flat_metrics