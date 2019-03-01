import torch
import numpy as np
import dnnutil.network as network


__all__ = ['calculate_accuracy', 'Trainer', 'ClassifierTrainer', 'AutoencoderTrainer']


def calculate_accuracy(prediction, label, axis=1):
    '''calculate_accuracy(prediction, label)
    
    Computes the mean accuracy over a batch of predictions and corresponding
    ground-truth labels.

    Args:
        prediction (Tensor): A batch of predictions. Assumed to have shape
            [batch-size, nclasses, [d0, d1, ...]].
        label (LongTensor): A batch of labels. Assumed to have shape
            [batch-size, [d0, d1, ...]]). The number of dimensions should be
            one less than prediction.

    Returns:
        accuracy (Tensor): A single-element Tensor containing the percent of
            correct predictions in the batch as a value between 0 and 1.
    '''
    return torch.eq(prediction.argmax(axis), label).float().mean().item()


class Trainer(object):
    '''Trainer(net, optim, loss_fn, accuracy_metric=None)
    
    Base class for all network trainers. Network trainer classes provide 
    methods to facilitate training and testing deep network models. The goal
    is to encapsulate the common functionality, to reduce the boilerplate
    code that needs to be repeated across projects.

    Args:
        net (torch.nn.Module): An instance of a network that inherits from
            torch.nn.Module.
        optim (torch.optim.Optimizer): An instance of an optimizer that
            inherits from torch.optim.Optimizer.
        loss_fn (callable): A callable that calculates and returns a loss
            value. The loss value should be a single-element Tensor.
        accuracy_metric (callable): A callabel that calculates and returns
            an accuracy value. Usually this will be a floating point number
            in [0, 1].
    '''
    def __init__(self, net, optim, loss_fn, accuracy_metric=None):
        self.net = net
        self.loss_fn = loss_fn
        self.optim = optim
        if accuracy_metric is not None:
            self.measure_accuracy = accuracy_metric
        else:
            self.measure_accuracy = calculate_accuracy

        self.train_loss = 0.
        self.train_acc = 0.
        self.test_loss = 0.
        self.test_acc = 0.
        
    def _set_train_stats(self, stats):
        '''TODO:docs
        '''
        self.train_loss = stats[0]
        self.train_acc = stats[1]

    def _set_test_stats(self, stats):
        '''TODO:docs
        '''
        self.test_loss = stats[0]
        self.test_acc = stats[1]

    def get_stats(self):
        '''TODO:docs
        '''
        return (self.train_loss, self.train_acc,
                self.test_loss, self.test_acc)

    def train(self, dataloader, epoch):
        '''Train the Trainer's network.

        Args:
            dataloader (torch.utils.data.DataLoader): An instance of a
                DataLoader, which will provide access to the training data.
            epoch (int): The current epoch.

        Returns:
            loss (float): The mean loss over the epoch.
            accuracy (float): The mean accuracy over the epoch (in [0, 1]).
        '''
        self.net.train()
        stats = self._run_epoch(dataloader, epoch)
        self._set_train_stats(stats)
        return stats

    def eval(self, dataloader, epoch):
        '''Evaluate the Trainer's network.

        Args:
            dataloader (torch.utils.data.DataLoader): An instance of a
                DataLoader, which will provide access to the testing data.
            epoch (int): The current epoch.
        Returns:
            loss (float): The mean loss over the epoch.
            accuracy (float): The mean accuracy over the epoch (in [0, 1]).
        '''
        self.net.eval()
        stats = self._run_epoch(dataloader, epoch)
        self._set_test_stats(stats)
        return stats
        
    def _run_epoch(self, dataloader, epoch):
        '''Perform a single epoch of either training or evaluation.

        Args:
            dataloader (torch.utils.data.DataLoader): An instance of a
                DataLoader, which will provide access to the testing data.
            epoch (int): The current epoch.
        Returns:
            loss (float): The mean loss over the epoch.
            accuracy (float): The mean accuracy over the epoch (in [0, 1]).
        '''
        N = len(dataloader.batch_sampler)
        msg = 'train' if self.net.training else 'test'
        func = self.train_batch if self.net.training else self.test_batch
        loss = []
        acc = []
        for i, batch in enumerate(dataloader):
            batch_loss, batch_acc = func(batch)
                
            loss.append(batch_loss)
            acc.append(batch_acc)

            print(f'\rEPOCH {epoch}: {msg} batch {i:04d}/{N}{" "*10}',
                  end='', flush=True)

        loss = np.mean(loss)
        acc = np.mean(acc)

        return loss, acc
        
    def train_batch(self, batch):
        '''Train the Trainer's network on a single training batch.
        '''
        raise NotImplementedError()

    def test_batch(self, batch):
        '''Test the Trainer's network on a single testing batch.
        '''
        raise NotImplementedError()


class ClassifierTrainer(Trainer):
    '''ClassifierTrainer(net, optim, loss_fn, accuracy_metric=None)
    
    Trainer for training a network to do image classification.

    Args:
        net (torch.nn.Module): An instance of a network that inherits from
            torch.nn.Module.
        optim (torch.optim.Optimizer): An instance of an optimizer that
            inherits from torch.optim.Optimizer.
        loss_fn (callable): A callable that calculates and returns a loss
            value. The loss value should be a single-element Tensor.
        accuracy_metric (callable): A callabel that calculates and returns
            an accuracy value. Usually this will be a floating point number
            in [0, 1].
    '''
    def train_batch(self, batch):
        '''Train the Trainer's network on a single training batch.

        Args:
            batch (iterable): A 2-tuple of (images, labels). Images is a 4-d
                Tensor of shape (BxCxHxW), and labels is a Tensor of 2 or more
                dimensions (BxLx*) which matches images in the first (batch)
                dimension. The exact dimensionality of labels will depend on
                the application and loss function chosen, but often consists
                of integer class-indexes.
        Returns:
            loss (float): The mean loss over the batch.
            accuracy (float): The mean accuracy over the batch (in [0, 1]).
        '''
        self.optim.zero_grad()

        imgs, labels = network.tocuda(batch)

        predictions = self.net(imgs)
        loss = self.loss_fn(predictions, labels)

        loss.backward()
        self.optim.step()

        loss = loss.item()
        with torch.no_grad():
            accuracy = self.measure_accuracy(predictions, labels)
        return loss, accuracy

    @torch.no_grad()
    def test_batch(self, batch):
        '''Evaluate the Trainer's network on a single testing batch.

        Args:
            batch (iterable): A 2-tuple of (images, labels). Images is a 4-d
                Tensor of shape (BxCxHxW), and labels is a Tensor of 2 or more
                dimensions (BxLx*) which matches images in the first (batch)
                dimension. The exact dimensionality of labels will depend on
                the application and loss function chosen, but often consists
                of integer class-indexes.
        Returns:
            loss (float): The mean loss over the batch.
            accuracy (float): The mean accuracy over the batch (in [0, 1]).
        '''
        imgs, labels = network.tocuda(batch)
        predictions = self.net(imgs)
        loss = self.loss_fn(predictions, labels).item()
        accuracy = self.measure_accuracy(predictions, labels)
        return loss, accuracy


class AutoencoderTrainer(Trainer):
    '''AutoencoderTrainer(net, optim, loss_fn)

    Trainer for training an autoencoder network.

    Args:
        net (torch.nn.Module): An instance of a network that inherits from
            torch.nn.Module.
        optim (torch.optim.Optimizer): An instance of an optimizer that
            inherits from torch.optim.Optimizer.
        loss_fn (callable): A callable that calculates and returns a loss
            value. The loss value should be a single-element Tensor.
    '''
    def __init__(self, net, optim, loss_fn):
        super(AutoencoderTrainer, self).__init__(
            net, optim, loss_fn, None)
        delattr(self, 'measure_accuracy')

    def train_batch(self, batch):
        '''Train the Trainer's network on a single training batch.

        Args:
            batch (iterable): A 2-tuple of (images, labels). Images is a 4-d
                Tensor of shape (BxCxHxW), and labels is a Tensor of 2 or more
                dimensions (BxLx*) which matches images in the first (batch)
                dimension. The exact dimensionality of labels will depend on
                the application and loss function chosen, but often consists
                of integer class-indexes.
        Returns:
            loss (float): The mean loss over the batch.
        '''
        self.optim.zero_grad()

        imgs = network.tocuda(batch)

        predictions = self.net(imgs)
        loss = self.loss_fn(predictions, imgs)

        loss.backward()
        self.optim.step()

        loss = loss.item()

        return loss

    @torch.no_grad()
    def test_batch(self, batch):
        '''Evaluate the Trainer's network on a single testing batch.

        Args:
            batch (iterable): A 2-tuple of (images, labels). Images is a 4-d
                Tensor of shape (BxCxHxW), and labels is a Tensor of 2 or more
                dimensions (BxLx*) which matches images in the first (batch)
                dimension. The exact dimensionality of labels will depend on
                the application and loss function chosen, but often consists
                of integer class-indexes.
        Returns:
            loss (float): The mean loss over the batch.
        '''
        imgs = network.tocuda(batch)
        predictions = self.net(imgs)
        loss = self.loss_fn(predictions, imgs).item()
        return loss

    def _run_epoch(self, dataloader, epoch):
        '''Perform a single epoch of either training or evaluation.

        Args:
            dataloader (torch.utils.data.DataLoader): An instance of a
                DataLoader, which will provide access to the testing data.
            epoch (int): The current epoch.
        Returns:
            loss (float): The mean loss over the epoch.
        '''
        N = int(np.ceil(len(dataloader.dataset) / dataloader.batch_size))
        msg = 'train' if self.net.training else 'test'
        func = self.train_batch if self.net.training else self.test_batch
        loss = []
        for i, batch in enumerate(dataloader):
            batch_loss = func(batch)
            loss.append(batch_loss)

            print(f'\rEPOCH {epoch}: {msg} batch {i:04d}/{N}{" "*10}',
                  end='', flush=True)

        loss = np.mean(loss)

        return loss

