""" Benchmark Evaluator class. Handles train-time benchmark eval. """

import time
import os
from os import devnull
from contextlib import contextmanager,redirect_stderr,redirect_stdout
from requests.exceptions import HTTPError

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
        mixed_precision: bool = True,
    ):
        self.model = model
        self.eval_batch_size = eval_batch_size
        self.eval_max_len = eval_max_len
        self.world_size = world_size
        self.world_rank = world_rank
        self.mixed_precision = mixed_precision

        self.eval_model = HFLM(
            pretrained=model,
            tokenizer=tokenizer, # type: ignore
            add_bos_token=True,
            dtype=torch.bfloat16 if model.config.use_bfloat16 else torch.float16,
            batch_size=eval_batch_size,
            backend='causal'
        )

        eval_task_distribution = {
            'hellaswag': 40168,
            'openbookqa': 2000,
            'winogrande': 2534,
            'arc_easy': 9501,
            'arc_challenge': 4687,
            'boolq': 6540,
            'piqa': 3676,
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

        max_retries = 30
        download_retries = 0

        while True:
            try:
                with suppress_stdout_stderr():
                    autocast_dtype = torch.bfloat16 if self.model.config.use_bfloat16 else torch.float16
                    with torch.autocast( device_type='cuda', dtype=autocast_dtype, enabled=self.mixed_precision ):
                        eval_results = simple_evaluate(
                            self.eval_model,
                            tasks=self.eval_tasks,
                            device='cuda',
                            log_samples=False,
                            batch_size=self.eval_batch_size,
                            verbosity='ERROR',
                            cache_requests='LM_HARNESS_CACHE_PATH' in os.environ
                        )[ 'results' ] # type: ignore
            except HTTPError as err:
                download_retries += 1

                if download_retries > max_retries:
                    raise err
                else:
                    print( f'Download error. Retrying in {download_retries * 10} seconds.' )
                    time.sleep( download_retries * 10 )
                    continue
            else:
                break

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
