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
    def __init__( self, is_distributed: bool ):
        self.metric = metrics.Mean().to( 'cuda' )
        self.is_distributed = is_distributed

    def update( self, input: torch.Tensor, *, weight: float | int | torch.Tensor = 1.0 ):
        return self.metric.update( input, weight=weight)

    def compute( self ):
        if self.is_distributed:
            return sync_and_compute( self.metric )
        return self.metric.compute()

    def reset( self ):
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
    def __call__( self, parser, namespace, values, option_string=None ):
        setattr( namespace, self.dest, dict() )
        for value in values: # type: ignore
            key, value = value.split( '=' )
            getattr( namespace, self.dest )[ key ] = yaml.load( value, yaml.FullLoader )


def field_parser( parser: ArgumentParser, f: Field ):
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
    for k, v in new_dict.items():
        if isinstance( v, Mapping ):
            base_dict[k] = recursive_dict_update( base_dict.get( k, {} ), v )
        elif isinstance( v, list ):
            base_dict[k] = base_dict.get( k, [] ) + v
        else:
            base_dict[k] = v
    return base_dict

def parse_yaml_file( path: str, prefix: str ) -> dict:
    if not os.path.isfile( path ):
        raise ValueError( f'Config file {path} does not exist!' )

    with open( path, 'r', encoding='utf-8' ) as f:
        new_dict = yaml.load( f, yaml.FullLoader )

    return new_dict[ prefix ]
