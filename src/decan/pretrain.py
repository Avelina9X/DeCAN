""" CLI loader for pretrain """

from argparse import ArgumentParser

import torch.multiprocessing as mp

from train import TrainerConfig, Trainer

def run( rank: int, world_size: int, config: TrainerConfig ) -> None:
    """ Defines the main entry point for pretraining *after* initialisation.

    NOTE: Only supports single-node, single- or multi-device training.

    Args:
        rank (int): Device rank.
        world_size (int): Device world size.
        config (TrainerConfig): Config of already initialise run.
    """
    
    trainer = Trainer( config, world_size, rank )
    try:
        trainer.train()
    except KeyboardInterrupt:
        print( 'KeyboardInterrupt: aborting early!' )
    trainer.cleanup()

def setup() -> None:
    """ Setup function for pretraining.
    Parses command line arguments and initialises new runs or checks existing runs for resuming.
    """

    # Parse command line arguments
    parser = ArgumentParser()
    init_mode, trainer_kwargs, model_kwargs = TrainerConfig.parse_arguments( parser )

    # Perform initialisation and return the corresponding config
    config = Trainer.initialize( init_mode, trainer_kwargs, model_kwargs )

    # Check if multi-device
    if config.use_ddp:
        # We MUST use spawning for all child processes to prevent locks
        mp.set_start_method( 'spawn' )

        # Run the main entry point on all devices
        mp.spawn( # type: ignore
            fn=run,
            args=( config.num_devices, config ),
            nprocs=config.num_devices,
            join=True,
        )
    else:
        # If single device we can just call the main entry point
        run( 0, 1, config )
