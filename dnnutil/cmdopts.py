import argparse
from pathlib import Path
import sys


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
    rundir = kwargs.get('run_dir', 'runs/')
    parser = _ConfigParser(description)
    a = parser.add_argument('--lr', type=float,
        help='Learning rate.')
    a = parser.add_argument('--batch-size', type=int,
        help='Batch size.')
    a = parser.add_argument('-e', '--epochs', type=int,
        help='Number of epochs to train for.')
    a = parser.add_argument('--model', default='',
        help='Path to model checkpoint')
    a = parser.add_argument('-D', '--run-dir', default=rundir,
        help='Path to root directory where runs should be saved.')
    
    subparsers = parser.add_subparsers()

    parse_start = subparsers.add_parser('start', aliases=['s'],
        help='Start a new training run. Creates a new run directory and '
             'begins training a model according to the provided config file.')
    parse_start.add_argument('config', help='Configuration file for the run.')
    parse_start.set_defaults(setup='start', rid=0)

    parse_cont = subparsers.add_parser('continue', aliases=['cont', 'c'],
        help='Continue a completed or interrupted training run.')
    a = parse_cont.add_argument('-i', '--rid', type=valid_continue_id,
        default=-1,
        help='Integer representing the run id. Use -1 for the most recently '
             'created run, or a positive integer to select a different run. '
             'Default: -1')
    parse_cont.set_defaults(setup='continue')

    return parser


def valid_continue_id(s):
    val = int(s)
    if val == 0 or val < -1:
        raise argparse.ArgumentTypeError(
            '{} is not a valid run id to continue from')
    return val


class _ConfigParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        if args is None:
            args = sys.argv[1:]
        if len(args) == 0:
            self.print_help()
            sys.exit(1)
        ns = super(_ConfigParser, self).parse_args(args, namespace)
        return ns

