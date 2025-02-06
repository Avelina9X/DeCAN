""" Module containing the trainer configuration class. """

from argparse import ArgumentParser
import os
import json
import yaml
from dataclasses import dataclass, field, fields, Field
from typing import Any, Literal, Optional
from collections.abc import Sequence

import shortuuid
import torch

from transformers.utils import logging

from .utils import (
    field_parser,
    model_kwargs_parser,
    recursive_dict_update,
    parse_yaml_file,
)

logger = logging.get_logger( __name__ )

@dataclass
class TrainerConfig:
    output_dir: str = field( metadata={
        'help': 'The output directory to store the run folder. May be a formatted string to use envars.',
        'track': False
    } )
    
    run_name: str = field( metadata={
        'help': 'The run name and child directory where model, optimizer and dataset checkpoints will be written. May be a formatted to which includes a 4 character UUID with {uuid}.',
        'track': False
    } )

    wandb_mode: str = field( metadata={ 'help': 'WandB tracking mode. Must be `online`, `offline`, or `disabled`. May be a formatted string to use envars.', 'track': False } )
    wandb_project: str = field( metadata={ 'help': 'Name of the project to track runs to in WandB. May be a formatted string to use envars.', 'track': False } )
    wandb_group: str = field( metadata={ 'help': 'Name of the group inside the WandB project to assign runs to. May be a formatted string to use envars.', 'track': False } )
    wandb_tags: list[str] = field( metadata={ 'help': 'A list of strings to populate the tags property of a WandB run.', 'track': False } )

    micro_batch_size: int = field( metadata={ 'help': 'Micro batch size of a single gradient accumulation step per device.' } )
    global_batch_size: int = field( default=2048, metadata={ 'help': 'The global batch size which may be sharded across multiple devices.' } )
    eval_batch_size: int = field( default=-1, metadata={ 'help': 'The batch size to use during validation. When set to -1 we use `micro_batch_size`.' } )

    max_steps: int = field( default=262144, metadata={ 'help': 'Maximum number of training steps.' } )
    warmup_steps: int = field( default=2048, metadata={ 'help': 'Number of warmup steps from 0 to lr_max.' } )

    steps_per_epoch: int = field( default=256, metadata={ 'help': 'Number of steps which constitute an `epoch`' } )
    validation_freq: int = field( default=4, metadata={ 'help': 'Number of `epochs` that must pass before performing validation.' } )
    temp_checkpoint_freq: int = field( default=4, metadata={ 'help': 'Number of `epochs` that must pass before saving a temporary checkpoint.' } )
    perm_checkpoint_freq: int = field( default=64, metadata={ 'help': 'Number of `epochs` that must pass before saving a permanent checkpoint.' } )

    training_dataset: str = field( default='smollm_corpus', metadata={ 'help': 'Training dataset to use.' } )
    starting_shard: int = field( default=0, metadata={ 'help': 'Initial shard number to start with. Defaults to zero.' } )

    epochs_per_session: int = field( default=-1, metadata={
        'help': 'Number of `epochs` that must pass before we end the close the session. When set to -1 we end after `max_steps` have passed.',
        'track': False,
    } )

    sequence_length: int = field( default=2048, metadata={ 'help': 'Number of tokens per sequence.' } )
    cache_length: int = field( default=8192, metadata={ 'help': 'Maximum number of tokens to store in the cache.' } )

    document_masking: bool = field( default=True, metadata={ 'help': 'Prevents attention between different documents.' } )

    lr_max: float = field( default=1e-3, metadata={ 'help': 'Maximum absolute learning rate after warmup.' } )
    lr_min: float = field( default=1e-4, metadata={ 'help': 'Minimum learning rate at end of annealing.' } )

    weight_decay: float = field( default=0.1, metadata={ 'help': 'Weight decay to apply to all parameters but bias, embedding and normalization.' } )
    weight_decay_exclude: list[str] = field( default_factory=list, metadata={ 'help': 'Partial string matches for weight names to exclude from weight decay.' } )

    optimizer: str = field( default='adamw', metadata={ 'help': 'The name of the optimizer to use. Currently supports `adamw` only.' } )
    optimizer_kwargs: dict = field( default_factory=dict, metadata={ 'help': 'Optimizer arguments. Must NOT include learning rate or weight decay.' } )
    optimizer_zero: bool = field( default=False, metadata={ 'help': 'Enables the ZeRO optimizer if using DDP. Has no effect on single device training.' } )
    
    frozen_params: list[str] = field( default_factory=list, metadata={ 'help': 'Partial string matches for weight names to freeze.' } )

    cpu_offload_cache: bool = field( default=True, metadata={ 'help': 'Enables offloading the KV cache to system memory.' } )

    max_grad_norm: float = field( default=1.0, metadata={ 'help': 'Global gradient norm clipping value.' } )

    num_workers_per_device: int = field( default=1, metadata={
        'help': 'Number of parallel dataloading workers per device.',
        'track': False,
    } )

    num_devices: int = field( default=-1, metadata={
        'help': 'Number of GPUs to use. -1 means use all available GPUs.',
        'track': False,
    } )

    do_init: bool = field( default=True, metadata={
        'help': 'When `True` we create a new directory for the run and initialise the model and tokenizer. This should be set to `False` after initialisation.',
        'track': False,
    } )

    do_resume: bool = field( default=False, metadata={
        'help': 'When `True` we find the most recent temporary checkpoint and resume training. This should be set to `True` after initialisation.',
        'track': False,
    } )

    is_complete: bool = field( default=False, metadata={
        'help': 'When `True` this model has finished training.',
        'track': False,
    } )
    
    set_init_seed: Optional[int] = field( default=None, metadata={ 'help': 'Sets the seed used for model initialisation when not `None`. This does *not* affect the seed used during training.' } )

    def __post_init__( self ):
        if self.do_init and self.do_resume:
            raise ValueError( 'Cannot set both `do_init` and `do_resume` to True' )

        if self.num_devices == -1:
            self.num_devices = torch.cuda.device_count()

        if self.max_steps % ( self.steps_per_epoch * self.num_devices ) != 0:
            raise ValueError( '`max_steps` must be divisible by `steps_per_epoch * num_devices` to prevent epoch misalignment!' )

        if self.global_batch_size % ( self.micro_batch_size * self.num_devices ) != 0:
            raise ValueError( '`global_batch_size` must be divisible by `micro_batch_size * num_devices` to prevent batch misalignment!' )

        if self.global_batch_size % ( self.num_workers_per_device * self.num_devices ) != 0:
            raise ValueError( '`global_batch_size` must be divisible by `num_workers_per_device * num_devices` to prevent dataloader misalignment!' )

        # Compute the wandb properties with potential envar string replacements
        self.wandb_mode = self.wandb_mode.format( **os.environ )
        self.wandb_project = self.wandb_project.format( **os.environ )
        self.wandb_group = self.wandb_group.format( **os.environ )

        # Compute the output directory with potential envar string replacements
        self.output_dir = os.path.expanduser( self.output_dir.format( **os.environ ) )

        # Compute the run name with potential UUID string replacements
        self.run_name = self.run_name.format( uuid=shortuuid.uuid()[ : 4 ] )

        # Set eval batch size
        if self.eval_batch_size == -1:
            self.eval_batch_size = self.micro_batch_size


    @property
    def run_dir( self ) -> str:
        return os.path.join( self.output_dir, self.run_name )

    @property
    def curr_checkpoint_dir( self ) -> str:
        return os.path.join( self.run_dir, 'checkpoint_curr' )

    @property
    def final_checkpoint_dir( self ) -> str:
        return os.path.join( self.run_dir, 'checkpoint_final' )

    def perm_checkpoint_dir( self, step: int ):
        return os.path.join( self.run_dir, f'checkpoint_step_{step}' )


    @property
    def use_ddp( self ) -> bool:
        return self.num_devices > 1

    @property
    def use_zero_optimizer( self ) -> bool:
        return self.use_ddp and self.optimizer_zero


    @property
    def local_batch_size( self ) -> int:
        return self.global_batch_size // self.num_devices

    @property
    def gradient_accumulation_steps( self ) -> int:
        return self.local_batch_size // self.micro_batch_size


    def to_dict( self, tracked_only=False ) -> dict:
        def cond( f: Field ):
            return ( f.metadata.get( 'track', f.init ) if tracked_only else f.init )

        d = { f.name: getattr( self, f.name ) for f in fields( self ) if cond( f ) }
        return d

    def to_json_string( self, tracked_only=False ) -> str:
        return json.dumps( self.to_dict( tracked_only ), indent=2 )

    def to_wandb_dict( self ) -> dict:
        return self.to_dict( tracked_only=True )

    def save_config( self, save_directory: str ):
        os.makedirs( save_directory, exist_ok=True )
        json_file_path = os.path.join( save_directory, 'trainer.json' )
        with open( json_file_path, 'w', encoding='utf-8' ) as writer:
            writer.write( self.to_json_string() )

    @classmethod
    def load_config( cls, save_directory: str, trainer_kwargs: dict | None = None ):
        json_file_path = os.path.join( save_directory, 'trainer.json' )
        with open( json_file_path, 'r', encoding='utf-8' ) as reader:
            obj = json.load( reader )

        config = recursive_dict_update( obj, trainer_kwargs or {} )

        return cls( **config )


    def __str__( self ):
        self_as_attr = ',\n'.join( [ f'    {k}={repr(v)}' for k, v in self.to_dict().items() ] )
        return f'{self.__class__.__name__}(\n{self_as_attr}\n)'

    __repr__ = __str__


    @classmethod
    def parse_arguments( cls, parser: ArgumentParser, args: Sequence[str] | None = None ):

        # Add mode argument
        parser.add_argument(
            'mode',
            choices=[ 'pretrain' ],
            help='Training mode. Currently only supports `pretrain`.'
        )

        # Add argument for trainer type
        parser.add_argument(
            'init_mode',
            choices=[ 'new', 'setup', 'dummy', 'resume' ],
            help=(
                'Launch mode for the trainer:'
                ' `new` initialises a new checkpoint directory and saves the configs, and initialized model.'
                ' `setup` like new, but will exit after creating the mode directory.'
                ' `dummy` like setup, but will skip creating actual files and instead print configs to stdout. Use to test launch arguments.'
                ' `resume` continues training from the most recent checkpoint (use if a crash occurs, when training over multiple sessions, or after using `setup`.)'
            )
        )

        # Add argument for specifying model_config_path
        parser.add_argument(
            '--model_config',
            '--model-config',
            nargs='?',
            help='Path to model config yaml file.',
            dest='model_config_path'
        )

        # Add argument for specifying trainer_config_path
        parser.add_argument(
            '--trainer_config',
            '--trainer-config',
            nargs='?',
            help='Path to trainer config yaml file.',
            dest='trainer_config_path'
        )

        # Add argument for specifying manifest_file_path
        parser.add_argument(
            '--manifest_file',
            '--manifest-file',
            nargs='?',
            help='Name of manifest yaml file to create or load.',
            dest='manifest_file_name'
        )

        # Add arguments for all fields in the trainer
        for f in fields( cls ):
            field_parser( parser, f )

        # Add argument to parse model kwargs as key=value pairs
        model_kwargs_parser( parser )

        # Parse arguments as dict
        arguments = parser.parse_args( args ).__dict__

        # Pop mode, model config and trainer config
        arguments.pop( 'mode' )
        init_mode: Literal['new', 'setup', 'dummy', 'resume'] = arguments.pop( 'init_mode' )
        model_config_path: str | None = arguments.pop( 'model_config_path' )
        trainer_config_path: str | None = arguments.pop( 'trainer_config_path' )

        # Pop model kwargs
        model_kwargs: dict[str, Any] = arguments.pop( 'model_kwargs' ) or {}

        # Maybe get head expansion kwargs and set
        model_kwargs_hexp: dict[str, Any] | None = arguments.pop( 'model_kwargs_hexp', None )
        if model_kwargs_hexp is not None:
            model_kwargs[ 'head_expansion' ] = model_kwargs_hexp

        # All remaining arguments are trainer kwargs
        trainer_kwargs: dict[str, Any] = arguments

        # Handle manifest file
        manifest_file_name: str | None = arguments.pop( 'manifest_file_name' )
        

        ###### if `resume` and `manifest_file`
        # TODO: implement manifest_file system
        # - check we have `manifist_file` in trainer_kwargs but NOT `run_name`
        # - get all potential `run_name`s from manifest file
        # - get first non-completed run
        # - set `run_name` from manifest file
        ######

        ###### if `setup` or `new`
        # TODO: implement manifest_file system
        # - if `setup` we NEED a manifest file, otherwise if `new` and no file we skip
        # - check if manifest file is on disk:
        # -- if not on disk, create file, create runs list, add run to list
        # -- if on disk, load file, add run to list
        # - write back to manifest file
        ######

        # Check that output dir and run name are specified when resuming
        if init_mode in [ 'resume' ]:
            if not 'output_dir' in trainer_kwargs:
                raise ValueError( 'When resuming runs you MUST set `output_dir` on the command line.' )

            # Maybe load run from manifest
            if manifest_file_name:
                cls.get_manifest_runs( manifest_file_name, trainer_kwargs )

            # We need a run name to resume
            if not 'run_name' in trainer_kwargs:
                raise ValueError( 'When resuming runs you MUST set `run_name` on the command line.' )

            # Model cannot be modified from config or CLI
            if model_config_path or len( model_kwargs ) != 0:
                raise ValueError(
                    'When resuming runs you CANNOT modify model params.'
                )

            # Trainer cannot be modified from config
            if trainer_config_path:
                raise ValueError(
                    'When resuming runs you CANNOT set trainer parameters from a YAML config file. '
                    'If you would like to change parameters use the appropriate command line args.'
                )
            
        # If model config is specified, load and update the model kwargs
        if model_config_path:
            config_kwargs = parse_yaml_file( model_config_path, 'model' )
            model_kwargs = recursive_dict_update( config_kwargs, model_kwargs )

        # If trainer config is specified, load and update the trainer kwargs
        if trainer_config_path:
            config_kwargs = parse_yaml_file( trainer_config_path, 'trainer' )
            trainer_kwargs = recursive_dict_update( config_kwargs, trainer_kwargs )
        
        return init_mode, trainer_kwargs, model_kwargs, manifest_file_name

    @classmethod
    def check_run_is_complete( cls, output_dir: str, run_name: str ) -> bool:
        # Get load directory
        load_dir = os.path.join(
            os.path.expanduser( output_dir ),
            run_name,
            'checkpoint_curr'
        )

        # Load config and check is_complete field
        return cls.load_config( load_dir, {} ).is_complete

    @classmethod
    def get_manifest_runs( cls, manifest_file_name: str, trainer_kwargs: dict ):
        if not 'output_dir' in trainer_kwargs:
            raise ValueError( 'When resuming runs you MUST set `output_dir` on the command line.' )

        output_dir = trainer_kwargs[ 'output_dir' ]
        file_path = os.path.join( output_dir, manifest_file_name )

        if 'run_name' in trainer_kwargs:
            raise ValueError( 'When resuming runs from manifest you MUST NOT set `run_name` on the command line.' )

        with open( file_path, 'r', encoding='utf-8' ) as f:
            run_dict = yaml.load( f, yaml.FullLoader )
        runs = run_dict[ 'runs' ]

        for run_num in runs:
            if not cls.check_run_is_complete( output_dir, run_num ):
                trainer_kwargs[ 'run_name' ] = run_num
                break

    @classmethod
    def add_manifest_run( cls, manifest_file_name: str, output_dir: str, run_name: str ):
        # Create directory if it doesn't exist
        os.makedirs( output_dir, exist_ok=True )

        # Get file path
        file_path = os.path.join( output_dir, manifest_file_name )

        if os.path.exists( file_path ):
            with open( file_path, 'r', encoding='utf-8' ) as f:
                run_dict = yaml.load( f, yaml.FullLoader )
            run_dict[ 'runs' ].append( run_name )
        else:
            run_dict = { 'runs': [ run_name ] }

        with open( file_path, 'w', encoding='utf-8' ) as f:
            yaml.dump( run_dict, f, default_flow_style=False, sort_keys=False )
        
        
        