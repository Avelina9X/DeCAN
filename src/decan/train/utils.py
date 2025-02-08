""" Utils module for training """

import os
import yaml
from typing import Mapping, Union
from inspect import isclass
from dataclasses import Field
from argparse import ArgumentParser, SUPPRESS, Action

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from torcheval import metrics
from torcheval.metrics.toolkit import sync_and_compute

class MeanMetric():
    """ A mean metric implementing `Mean` from `torcheval.metrics`.
    
    This class differs from torcheval in that it will automatically sync across ranks when `is_distributed=True`
    """
    def __init__( self, is_distributed: bool ):
        """ Initialize a metric object and its internal states.

        Args:
            is_distributed (bool): Set to True when using DDP to enable automatic sync.
        """
        self.metric = metrics.Mean().to( 'cuda' )
        self.is_distributed = is_distributed

    def update( self, input: torch.Tensor, *, weight: float | int | torch.Tensor = 1.0 ): # pylint: disable=W0622
        """ Updates the mean metric using the underlying `Mean` object.

        Args:
            input (torch.Tensor): Tensor of input values.
            weight (float | int | torch.Tensor, optional): Float or Int or Tensor of input weights. If weight is a Tensor, its size should match the input tensor size. Defaults to 1.0.

        Returns:
            Mean: returns the underlying `torcheval.metrics.Mean` object.
        """
        return self.metric.update( input, weight=weight)

    def compute( self ) -> torch.Tensor:
        """ Computes the weighted mean using the underlying `Mean` object
        and syncronises across ranks if `is_ditributed=True`

        Returns:
            torch.Tensor: Weighted mean.
        """
        if self.is_distributed:
            return sync_and_compute( self.metric )
        return self.metric.compute()

    def reset( self ):
        """ Resets the metric state variables to their default value
        and applies a distributed barrier if `is_ditributed=True`
        """
        if self.is_distributed:
            dist.barrier()
        self.metric.reset()


class DDPModelWrapper( DistributedDataParallel ):
    """ Custom DDP wrapper. Defers method and attribute accesses to underlying module. """
    def __getattr__( self, name ):
        try:
            return super().__getattr__( name )
        except AttributeError:
            return getattr( self.module, name )


class ParseKwargs( Action ):
    """ Special Action for argparse to parse key value pairs """
    def __call__( self, parser, namespace, values, option_string=None ):
        setattr( namespace, self.dest, dict() )
        for value in values: # type: ignore
            key, value = value.split( '=' )
            getattr( namespace, self.dest )[ key ] = yaml.load( value, yaml.FullLoader )


def field_parser( parser: ArgumentParser, f: Field ):
    """ Automatically adds fields of a dataclass as arguments to an ArgumentParaser.

    Args:
        parser (ArgumentParser): The parser to add arguments to.
        f (Field): The dataclass field to turn into an argument.
    """
    if isinstance( f.type, str ):
        raise RuntimeError( 'Unresolved type detected!' )

    origin_type = getattr( f.type, '__origin__', f.type )

    is_list = isclass( origin_type ) and issubclass( origin_type, list )
    is_dict = isclass( origin_type ) and issubclass( origin_type, dict )
    is_optional = origin_type is Union and len( f.type.__args__ ) == 2
    is_bool = f.type is bool
    true_type = f.type.__args__[0] if is_list or is_optional else f.type

    if is_list:
        parser.add_argument(
            f"--{f.name}",
            f"--{f.name.replace( '_', '-' )}",
            type=true_type,
            nargs='+',
            dest=f.name,
            default=SUPPRESS,
            help=f.metadata[ 'help' ]
        )
    elif is_dict:
        parser.add_argument(
            f"--{f.name}",
            f"--{f.name.replace( '_', '-' )}",
            nargs='*',
            dest=f.name,
            default=SUPPRESS,
            action=ParseKwargs,
            help=f.metadata[ 'help' ]
        )
    elif is_bool:
        parser.add_argument(
            f"--{f.name}",
            f"--{f.name.replace( '_', '-' )}",
            type=lambda x: str( x ).lower() in [ 'true', '1', 'yes' ],
            nargs='?',
            dest=f.name,
            default=SUPPRESS,
            help=f.metadata[ 'help' ]
        )
    else:
        parser.add_argument(
            f"--{f.name}",
            f"--{f.name.replace( '_', '-' )}",
            type=true_type,
            nargs='?',
            dest=f.name,
            default=SUPPRESS,
            help=f.metadata[ 'help' ]
        )
        
def model_kwargs_parser( parser: ArgumentParser ):
    """ Adds special model kwarg fields to an ArgumentParaser.
    
    This adds:
    - `--model-kwargs` to parse key-value pairs for the model config.
    - `--model-kwargs-hexp` to parse key-value pairs for the `head_expansion` dict in the model config.

    Args:
        parser (ArgumentParser): The parser to add arguments to.
    """
    parser.add_argument(
        '--model_kwargs',
        '--model-kwargs',
        nargs='*',
        action=ParseKwargs,
        dest='model_kwargs',
        help='Additional `key=value` pairs to be passed to the model config loader.'
    )

    parser.add_argument(
        '--model_kwargs_hexp',
        '--model-kwargs-hexp',
        nargs='*',
        default=SUPPRESS,
        action=ParseKwargs,
        dest='model_kwargs_hexp',
        help='Additional `key=value` pairs to be passed to the model config loader.'
    )

def recursive_dict_update( base_dict: dict, new_dict: Mapping ) -> dict:
    """ Overwrites a base dict with elements from a new dict recursively.
    
    When overwriting a list elements from the new dict will be APPENDED to the old dict.
    When overwriting a dict, elements from the new dict will UPDATE the old dict.
    
    Note the updates will happen in place, but we will also return the base dict.

    Args:
        base_dict (dict): The base dict to receive updates.
        new_dict (Mapping): The new dict to retrieve keys and values from.

    Returns:
        dict: A reference to the updated base dict.
    """
    for k, v in new_dict.items():
        if isinstance( v, Mapping ):
            base_dict[k] = recursive_dict_update( base_dict.get( k, {} ), v )
        elif isinstance( v, list ):
            base_dict[k] = base_dict.get( k, [] ) + v
        else:
            base_dict[k] = v
    return base_dict

def parse_yaml_file( path: str, prefix: str ) -> dict:
    """ Parses a YAML file and returns the conents of the `prefix` element.
    This is intended for parsing config YAML files which may contain both `train` and `model` keys.

    Args:
        path (str): Path to the YAML file to parse.
        prefix (str): Top level key in the YAML structure to return.

    Returns:
        dict: The value of the top level key given by `prefix`.
    """
    if not os.path.isfile( path ):
        raise ValueError( f'Config file {path} does not exist!' )

    with open( path, 'r', encoding='utf-8' ) as f:
        new_dict = yaml.load( f, yaml.FullLoader )
        assert isinstance( new_dict, dict )

    return new_dict[ prefix ]
