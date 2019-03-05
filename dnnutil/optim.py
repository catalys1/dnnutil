import torch
import math


__all__ = ['EpochSetLR', 'CosineAnealingLR']


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


class CosineAnealingLR(torch.optim.lr_scheduler._LRScheduler):
    '''Implements cosine anealed learning rate scheduler with restarts.
    This learning rate scheduler is meant to operate at the mini-batch
    level, so the `epoch` argument to step() should be the current
    iteration number.

    Args:
        optimizer (torch.nn.Optimizer): reference to an optimizer object.
        lr_max (float): the maximum learning rate within a cycle.
        stepsize (int): number of iterations in a cycle.
        last_epoch (int): the index of the last iteration. Default: -1.

    Example:
        >>> scheduler = CosineAnealingLR(optim, 0.2, 10)
        >>> for i in range(11):
        >>>     scheduler.step(i + 1)
        >>>     print(optim.param_groups[0]['lr'])
        0.2
        0.19510565162951538
        0.18090169943749476
        ...
        0.2
    '''
    def __init__(self, optimizer, lr_max, stepsize, last_epoch=-1):
        self.lr_max = lr_max
        self.stepsize = stepsize
        super(CosineAnealingLR, self).__init__(optimizer, last_epoch)

    def _f(self, i):
        '''Compute the LR for a given iteration'''
        a = self.lr_max
        stepsize = self.stepsize
        lr = a / 2 * (
             math.cos(math.pi * ((i - 1) % stepsize) / stepsize) + 1)
        return lr

    def get_lr(self):
        lr = self._f(self.last_epoch)
        return [lr for _ in self.base_lrs]
        


def lr_sweep(lr_min, lr_max, batch_size=None, epochs=1):
    pass
