""" Trainer module for training DeCAN models """

import os
from typing import Literal

from model import DeCANConfig, DeCANForCausalLM
from model.utils import load_tokenizer, set_pretrained_embeddings
from .configuration_trainer import TrainerConfig


class Trainer:
    @staticmethod
    def initialize( mode: Literal['new', 'start', 'resume'], trainer_kwargs: dict, model_kwargs: dict ):
        match mode:
            case 'new':
                # Initialize fresh configs
                trainer_config = TrainerConfig( **trainer_kwargs )
                model_config = DeCANConfig( **model_kwargs )

                if not trainer_config.do_init:
                    raise ValueError( 'Got `do_init=False` when trying to start a new run!' )

                # Load our modified tokenizer
                tokenizer = load_tokenizer()

                # Instantiate model
                model = DeCANForCausalLM( model_config )

                # Overwrite model embeddings
                set_pretrained_embeddings( model )

                # Now that everything is loaded, set do_init to False
                trainer_config.do_init = False

                # Get output save directory
                save_dir = trainer_config.curr_checkpoint_dir

                # Save all required files
                model.save_pretrained( save_dir )
                tokenizer.save_pretrained( save_dir, legacy_format=False )
                trainer_config.save_config( save_dir )

                return trainer_config

            case 'start':
                # Infer directory from trainer_kwargs
                load_dir = os.path.join(
                    os.path.expanduser( trainer_kwargs[ 'output_dir' ] ),
                    trainer_kwargs[ 'run_name' ],
                    'checkpoint_curr'
                )
                
                # Load config and update
                trainer_config = TrainerConfig.load_config( load_dir, trainer_kwargs )

                if trainer_config.do_resume:
                    raise ValueError( 'Got `do_resume=True` when trying to start a run!' )

                if trainer_config.do_init:
                    raise ValueError( 'Got `do_init=True` when trying to start a run!' )

                return trainer_config

            case 'resume':
                # Infer directory from trainer_kwargs
                load_dir = os.path.join(
                    os.path.expanduser( trainer_kwargs[ 'output_dir' ] ),
                    trainer_kwargs[ 'run_name' ],
                    'checkpoint_curr'
                )
                
                # Load config and update
                trainer_config = TrainerConfig.load_config( load_dir, trainer_kwargs )

                if not trainer_config.do_resume:
                    raise ValueError( 'Got `do_resume=False` when trying to resume a run!' )

                return trainer_config

            case _:
                raise ValueError( f"`mode` must be either 'new' or 'resume' but got {mode}" )
