""" CLI module """

import sys

if __name__ == '__main__':
    # Check the first argument to delegate to the correct module;
    # remaining args will be parsed in the corresponding setup()
    match sys.argv[1]: # type: ignore

        # Pretain module for, well, pretraining!
        case 'pretrain':
            import pretrain
            pretrain.setup()

        # Evaluation module for, well evaluation!
        case 'evaluate':
            import evaluator
            evaluator.setup()

        # Invalid choice!
        case _:
            raise ValueError( f'Invalid first argument `{sys.argv[1]}`' )
