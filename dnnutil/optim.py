import torch


class EpochSetLR(torch.optim.lr_scheduler._LRScheduler):
    '''Sets the initial learning rate of each parameter group to decay by an
    epoch-specific lambda. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (torch.nn.Optimizer): Wrapped optimizer.
        epoch_lrs (iterable): An iterable of 2-tuples, where the first item
            is an epoch number and the second item is a decay value.
        last_epoch (int): The index of the last epoch. Default: -1.

    Example:
        >>> # The learning rate will decay by a factor of 0.5 at epoch 100,
        >>> # and by 0.2 at epoch 200
        >>> scheduler = EpochSetLR(optim, [[100, 0.5], [200, 0.2]])
        >>> for epoch in range(300):
        >>>     scheduler.step()
        >>>     train(...)
        >>>     validate(...)
    '''
    def __init__(self, optimizer, epoch_lrs, last_epoch=-1):
        self.epoch_lrs = dict(epoch_lrs)
        self.current_lrs = None
        super(EpochSetLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self.current_lrs:
            self.current_lrs = [x for x in self.base_lrs]
        if self.last_epoch in self.epoch_lrs:
            self.current_lrs = [lr * self.epoch_lrs[self.last_epoch] for lr in self.current_lrs]
        return self.current_lrs
