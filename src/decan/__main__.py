""" CLI module """

import sys

import pretrain

if __name__ == '__main__':
    # Check the first argument to delegate to the correct module
    match sys.argv[1]:

        # Pretain module for, well, pretraining!
        case 'pretrain': pretrain.setup()

        # Invalid choice!
        case _: raise ValueError( f'Invalid first argument `{sys.argv[1]}`' )
