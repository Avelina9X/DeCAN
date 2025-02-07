""" CLI module """

import sys

import pretrain
import evaluator

if __name__ == '__main__':
    # Check the first argument to delegate to the correct module;
    # remaining args will be parsed in the corresponding setup()
    match sys.argv[1]: # type: ignore

        # Pretain module for, well, pretraining!
        case 'pretrain': pretrain.setup()

        case 'evaluate': evaluator.setup()

        # Invalid choice!
        case _: raise ValueError( f'Invalid first argument `{sys.argv[1]}`' )
