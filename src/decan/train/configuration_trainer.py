""" Module containing the trainer configuration class. """

import os
import json
from dataclasses import dataclass, field, fields, Field

import shortuuid
import torch

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

    wandb_mode: str = field( metadata={ 'help': 'WandB tracking mode. Must be `online`, `offline`, or `disabled`', 'track': False } )
    wandb_project: str = field( metadata={ 'help': 'Name of the project to track runs to in WandB.', 'track': False } )
    wandb_group: str = field( metadata={ 'help': 'Name of the group inside the WandB project to assign runs to.', 'track': False } )
    wandb_tags: list[str] = field( metadata={ 'help': 'A list of strings to populate the tags property of a WandB run.', 'track': False } )

    micro_batch_size: int = field( metadata={ 'help': 'Micro batch size of a single gradient accumulation step per device.' } )
    global_batch_size: int = field( default=2048, metadata={ 'help': 'The global batch size which may be sharded across multiple devices.' } )
    # eval_batch_size: int = field( default=-1, metadata={ 'help': 'The batch size to use during validation. When set to -1 we use `micro_batch_size`.' } )

    max_steps: int = field( default=262144, metadata={ 'help': 'Maximum number of training steps.' } )
    warmup_steps: int = field( default=2048, metadata={ 'help': 'Number of warmup steps from 0 to lr_max.' } )

    steps_per_epoch: int = field( default=256, metadata={ 'help': 'Number of steps which constitute an `epoch`' } )
    validation_freq: int = field( default=4, metadata={ 'help': 'Number of `epochs` that must pass before performing validation.' } )
    temp_checkpoint_freq: int = field( default=4, metadata={ 'help': 'Number of `epochs` that must pass before saving a temporary checkpoint.' } )
    perm_checkpoint_freq: int = field( default=64, metadata={ 'help': 'Number of `epochs` that must pass before saving a permanent checkpoint.' } )

    epochs_per_session: int = field( default=-1, metadata={
        'help': 'Number of `epochs` that must pass before we end the close the session. When set to -1 we end after `max_steps` have passed.',
        'track': False,
    } )

    sequence_length: int = field( default=2048, metadata={ 'help': 'Number of tokens per sequence.' } )
    cache_length: int = field( default=8192, metadata={ 'help': 'Maximum number of tokens to store in the cache.' } )

    lr_max: float = field( default=1e-3, metadata={ 'help': 'Maximum absolute learning rate after warmup.' } )
    lr_min: float = field( default=1e-4, metadata={ 'help': 'Minimum learning rate at end of annealing.' } )

    weight_decay: float = field( default=0.1, metadata={ 'help': 'Weight decay to apply to all parameters but bias, embedding and normalization.' } )

    optimizer: str = field( default='adamw', metadata={ 'help': 'The name of the optimizer to use. Currently supports `adamw` only.' } )
    optimizer_kwargs: dict = field( default_factory=dict, metadata={ 'help': 'Optimizer arguments. Must NOT include learning rate or weight decay.' } )
    optimizer_zero: bool = field( default=False, metadata={ 'help': 'Enables the ZeRO optimizer if using DDP. Has no effect on single device training.' } )

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

    def __post_init__( self ):
        if not ( self.do_init ^ self.do_resume ):
            raise ValueError( 'Exactly one of `do_init` or `do_resume` must be True' )

        if self.num_devices == -1:
            self.num_devices = torch.cuda.device_count()

        if self.max_steps % ( self.steps_per_epoch * self.num_devices ) != 0:
            raise ValueError( '`max_steps` must be divisible by `steps_per_epoch * num_devices` to prevent epoch misalignment!' )

        if self.global_batch_size % ( self.micro_batch_size * self.num_devices ) != 0:
            raise ValueError( '`global_batch_size` must be divisible by `micro_batch_size * num_devices` to prevent batch misalignment!' )

        if self.global_batch_size % ( self.num_workers_per_device * self.num_devices ) != 0:
            raise ValueError( '`global_batch_size` must be divisible by `num_workers_per_device * num_devices` to prevent dataloader misalignment!' )

        # Compute the output directory with potential envar string replacements
        self.output_dir = os.path.expanduser( os.path.join( *self.output_dir.format( **os.environ ).split( '/' ) ) )

        # Compute the run name with potential UUID string replacements
        self.run_name = self.run_name.format( uuid=shortuuid.uuid()[ : 4 ] )


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


    def __str__( self ):
        self_as_attr = ',\n'.join( [ f'    {k}={repr(v)}' for k, v in self.to_dict().items() ] )
        return f'{self.__class__.__name__}(\n{self_as_attr}\n)'

    __repr__ = __str__
