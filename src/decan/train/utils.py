""" Utils module for training """

import os
import yaml
from typing import Mapping
from inspect import isclass
from dataclasses import Field
from argparse import ArgumentParser, SUPPRESS, Action


class ParseKwargs( Action ):
    def __call__( self, parser, namespace, values, option_string=None ):
        setattr( namespace, self.dest, dict() )
        for value in values: # type: ignore
            key, value = value.split( '=' )
            getattr( namespace, self.dest )[ key ] = value

def field_parser( parser: ArgumentParser, f: Field ):
    if isinstance( f.type, str ):
        raise RuntimeError( 'Unresolved type detected!' )

    origin_type = getattr( f.type, '__origin__', f.type )

    is_list = isclass( origin_type ) and issubclass( origin_type, list )
    is_dict = isclass( origin_type ) and issubclass( origin_type, dict )
    true_type = f.type.__args__[0] if is_list else f.type

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
