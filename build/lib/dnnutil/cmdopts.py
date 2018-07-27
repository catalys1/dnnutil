import argparse


def basic_parser(description='' **kwargs):
    '''Returns a commandline parser for common arguments. Some defualt
    argument settings can be set using **kwargs.

    Valid kwargs:
        lr : float - the default learning rate
        batch_size : int - the default batch size
        epochs : int - the default number of training epochs
    '''
    lr = 1e-3 if 'lr' not in kwargs else kwargs['lr']
    batch_size = 16 if 'batch_size' not in kwargs else kwargs['batch_size']
    epochs = 50 if 'epochs' not in kwargs else kwargs['epochs']

    parser = argparse.ArgumentParser(description)

    parser.add_argument('--lr', type=float, default=lr, 
        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=batch_size,
        help='Batch size')
    parser.add_argument('--epochs', type=int, default=epochs,
        help='Epoch number to train to')
    parser.add_argument('--start', type=int, default=0,
        help='Epoch number to start from. Useful for resuming training.')
    parser.add_argument('--model', metavar='CHECKPOINT', default='',
        help='Load model checkpoint from file CHECKPOINT')
    parser.add_argument('--test', action='store_true', default=False,
        help='Runs inference on the model')
    parser.add_argument('--rid', type=int, default=1,
        help='Positive int representing run id. The run id can be used to keep '
             'seperate logs and checkpoints')
    parser.add_argument('--note', default='',
        help='Any notes about the model or run. Can be used to keep track of '
             'different runs, using different hyperparameters, etc.'

    return parser
