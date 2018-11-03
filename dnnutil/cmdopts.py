import argparse
from pathlib import Path


__all__ = ['basic_parser', 'config_parser']


def basic_parser(description='', **kwargs):
    '''Returns an argparse.ArgumentParser commandline parser, pre-populated
    with common arguments used when training deep networks. The argument
    parsers come with sane defaults, many of which can be changed by passing
    **kwargs. Further arguments can be added to the parser later through calls
    to parser.add_argument.

    Args:
        description (str): Optional description of the program, which will
            be passed the constructor of the ArgumentParser.

    Kwargs:
        lr (float): The default learning rate.
        batch_size (int): The default batch size.
        epochs (int): The default number of training epochs.
        run_dir (str): Path to directory for storing runs.
    '''
    lr = 1e-3 if 'lr' not in kwargs else kwargs['lr']
    batch_size = 16 if 'batch_size' not in kwargs else kwargs['batch_size']
    epochs = 50 if 'epochs' not in kwargs else kwargs['epochs']

    parser = argparse.ArgumentParser(description)

    a = parser.add_argument('--lr', type=float, default=lr, 
        help=f'Learning rate. Default: {lr}.')
    a = parser.add_argument('--batch-size', type=int, default=batch_size,
        help=f'Batch size. Default: {batch_size}.')
    a = parser.add_argument('--epochs', type=int, default=epochs,
        help=f'Number of epochs to train for. Default: {epochs}.')
    a = parser.add_argument('--start', type=int, default=0,
        help='Epoch number to start from. Useful for resuming training. '
             'Default: 0.')
    a = parser.add_argument('--model', metavar='CHECKPOINT', default='',
        help='Load model checkpoint from file CHECKPOINT.')
    a = parser.add_argument('--test', action='store_true', default=False,
        help='Runs inference on the model.')
    a = parser.add_argument('--rid', type=int, default=0,
        help='Integer representing run id. The run id can be used to keep '
             'seperate logs and checkpoints. Default: 0')
    a = parser.add_argument('--note', default='',
        help='Any notes about the model or run. Can be used to keep track of '
             'different runs, using different hyperparameters, etc.')
    a = parser.add_argument('--resume', action='store_true',
        help='Flag indicating a desire to restore values based on the last '
             'epoch of training. Actual saving and restoring of state must '
             'be handled by i.e. a Manager object.')

    return parser


def config_parser(description='', **kwargs):
    '''Returns a commandline parser for use with configuration files.

    Returns an argparse.ArgumentParser set up for use with
    configuration files.
    '''

    parser = argparse.ArgumentParser(description)
    a = parser.add_argument('--lr', type=float,
        help='Learning rate.')
    a = parser.add_argument('--batch-size', type=int,
        help='Batch size.')
    a = parser.add_arguemtn('-e', '--epochs', type=int,
        help='Number of epochs to train for.')
    subparsers = parser.add_subparsers()

    parse_start = subparsers.add_parser('start', aliases=['s'],
        help='Start a new training run.')
    parse_start.add_argument('config', help='Configuration file for the run.')

    parse_cont = subparsers.add_parser('continue', aliases=['cont', 'c'],
        help='Continue a completed or interrupted training run.')
    a = parse_cont.add_argument('-i', '--rid', type=int, default=-1,
        help='Integer representing the run id.')

    return parser

