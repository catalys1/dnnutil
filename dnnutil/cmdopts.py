import argparse
from pathlib import Path


def _next_rid():
    rid = 1
    p = Path('runs')
    if p.is_dir():
        runs = list(p.iterdir())
        if len(runs) > 0:
            last = max(int(x.name) for x in runs if x.name.isnumeric())
            rid = last + 1
    return rid


class RunSetter(argparse.Action):
    
    def __init_(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError('nargs not allowed')
        super(RunSetter, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        rid = values
        if rid == 0:
            rid = _next_rid()
        setattr(namespace, self.dest, rid)


class TrainArgParser(argparse.ArgumentParser):

    def parse_args(self, args=None, namespace=None):
        args = super(TrainArgParser, self).parse_args(args, namespace)
        for a in self._actions:
            if isinstance(a, RunSetter) and getattr(args, a.dest) == a.default:
                rid = _next_rid()
                setattr(args, a.dest, rid)
        return args
                

def basic_parser(description='', **kwargs):
    '''Returns an argparse.ArgumentParser commandline parser, pre-populated
    with common arguments used when training deep networks. The argument
    parsers come with sane defaults, many of which can be changed by passing
    **kwargs. Further arguments can be added to the parser later through calls
    to parser.add_argument.

    Args:
        description (str): Optional description of the program, which will
            be passed the constructor of the ArgumentParser.

    Valid kwargs:
        lr (float): The default learning rate.
        batch_size (int): The default batch size.
        epochs (int): The default number of training epochs.
        run_dir (str): Path to directory for storing runs.
    '''
    lr = 1e-3 if 'lr' not in kwargs else kwargs['lr']
    batch_size = 16 if 'batch_size' not in kwargs else kwargs['batch_size']
    epochs = 50 if 'epochs' not in kwargs else kwargs['epochs']

    parser = TrainArgParser(description)

    a = parser.add_argument('--lr', type=float, default=lr, 
        help='Learning rate')
    a = parser.add_argument('--batch_size', type=int, default=batch_size,
        help='Batch size')
    a = parser.add_argument('--epochs', type=int, default=epochs,
        help='Epoch number to train to')
    a = parser.add_argument('--start', type=int, default=0,
        help='Epoch number to start from. Useful for resuming training.')
    a = parser.add_argument('--model', metavar='CHECKPOINT', default='',
        help='Load model checkpoint from file CHECKPOINT')
    a = parser.add_argument('--test', action='store_true', default=False,
        help='Runs inference on the model')
    a = parser.add_argument('--rid', type=int, default=0, action=RunSetter,
        help='Positive int representing run id. The run id can be used to keep '
             'seperate logs and checkpoints')
    a = parser.add_argument('--note', default='',
        help='Any notes about the model or run. Can be used to keep track of '
             'different runs, using different hyperparameters, etc.')

    return parser
