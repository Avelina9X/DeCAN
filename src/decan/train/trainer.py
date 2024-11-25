""" Trainer module for training DeCAN models """

from typing import Literal
import json

import rich

from model import DeCANConfig, DeCANForCausalLM
from model.utils import load_tokenizer, set_pretrained_embeddings
from .configuration_trainer import TrainerConfig



class Trainer:
    @staticmethod
    def initialize( mode: Literal['new', 'resume'], trainer_kwargs: dict, model_kwargs: dict ):
        match mode:
            case 'new':
                # Initialize fresh configs
                trainer_config = TrainerConfig( **trainer_kwargs )
                model_config = DeCANConfig( **model_kwargs )

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

            case 'resume':
                ...

            case _:
                raise ValueError( f"`mode` must be either 'new' or 'resume' but got {mode}" )
