from pathlib import Path


__all__ = ['TextLog']


class TextLog(object):
    '''TextLog(filepath, console=True, create_ok=True)

    The TextLog logger logs per-epoch information in a human-readable way
    to a simple text file, and optionally to the command line.

    The output has the following form:

        "Epoch E (lr=LR, time=T): Train [loss (acc)]  Val [loss (acc)]  "

    where E is the epoch number, LR is the current learning rate, and T is
    running time for both training and evaluation.

    Args:
        filepath (str or Path): Path to the log file.
        console (bool): Whether to print log information to the commandline.
            Default: True.
        create_ok (bool): If True, allows the creation of any missing
            folders along the path to logfile. Default: True.
    '''
    def __init__(self, filepath, console=True, create_ok=True):
        self.filepath = Path(filepath)
        self.console = console

        if create_ok:
            self.filepath.parent.mkdir(parents=True, exist_ok=True)


    def log(self, epoch, time, train_loss=0, train_acc=0,
            test_loss=0, test_acc=0, lr=None):
        '''Append to the log file, and optionally print to the console.
        Accepts the current epoch number, execution time for the epoch,
        values for the training and test loss and accuracy, and optionally
        the value of the learning rate that was used.

        Args:
            epoch (int): Epoch number.
            time (float): Elapsed time over the epoch.
            train_loss (float): Loss measured across the training set.
            train_acc (float): Accuracy measured across the training set.
            test_loss (float): Loss measured across the test/validation set.
            test_acc (float): Accuracy measured across the test/validation set.
            lr (float): The learning rate.
        '''
        lr = f'{lr:1.0e}' if lr is not None else '-'
        s = (
            f'EPOCH {epoch:<3d} (lr={lr}, time={time:.1f}): '
            f'Train [{train_loss:.4f} ({train_acc * 100:02.2f})]  '
            f'Val [{test_loss:.4f} ({test_acc * 100:02.2f})]  '
        )

        with self.filepath.open('a') as f:
            f.write(s)
            f.write('\n')

        if self.console:
            print('\r', end='')
            print(s)

    def text(self, text):
        '''Append arbitrary text to the log file. Useful for keeping notes
        when starting a new run or resuming a stopped run.

        Args:
            text (str): Text to add to the log file.
        '''
        with self.filepath.open('a') as f:
            f.write(text)
            f.write('\n')

