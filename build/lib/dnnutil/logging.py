

class TextLog(object):

    def __init__(self, filepath, console=True):
        self.filepath = filepath
        self.console = console

    def log(epoch, time, train_loss=0, train_acc=0, test_loss=0, test_acc=0, lr=None):
        lr = '{lr:1.0e}' if lr is not None else '-'
        s = (
            f'EPOCH {epoch} (lr={lr}, time={t:.1f}): '
            f'Train [{train_loss:.4f} ({train_acc * 100:02.2f})]  '
            f'Val [{test_loss:.4f} ({test_acc * 100:02.2f})]  '
        )

        with open(self.filepath, 'a') as f:
            f.write(s)
            f.write('\n')

        if self.console:j
            print('\r')
            print(s)
