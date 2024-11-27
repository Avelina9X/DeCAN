""" CLI module """

from argparse import ArgumentParser

import torch.multiprocessing as mp

from train import TrainerConfig, Trainer

def run( rank, world_size, config: TrainerConfig ):
    trainer = Trainer( config, world_size, rank )
    try:
        trainer.train()
    except KeyboardInterrupt:
        print( 'KeyboardInterrupt: aborting early!' )
    trainer.cleanup()

def setup():
    parser = ArgumentParser()
    init_mode, trainer_kwargs, model_kwargs = TrainerConfig.parse_arguments( parser )
    config = Trainer.initialize( init_mode, trainer_kwargs, model_kwargs )

    if config.use_ddp:
        mp.set_start_method( 'spawn' )
        mp.spawn( # type: ignore
            fn=run,
            args=( config.num_devices, config ),
            nprocs=config.num_devices,
            join=True,
        )
    else:
        run( 0, 1, config )

if __name__ == '__main__':
    setup()
