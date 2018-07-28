# Deep Neural Net Utilities for PyTorch

A package containing utilities for handling common deep-network tasks in PyTorch.

Some of the included features are:

- An extendable argument parser pre-loaded with common options
- Objects for abstracting
  - Training and validation loops
  - Logging
  - Checkpointing
  - Saving models and outputs across multiple training runs

The purpose of the package is to reduce the amount of code that has to be written and re-written (and which changes very little) from project to project. Hopefully this allows the data scientist to focus on the model and the data instead of boilerplate code.

## Example

Here's a full example of how dnnutil can be used. Train/test loops, logging, checkpointing, commandline arguments are all handled by dnnutil functions and classes. The deep lerning practitioner just needs to plug in his model and dataset.

```python
import torch
import dnnutil
import mymodel
import mydataset
import time


def accuracy(prediction, label):
    return torch.eq(prediction.argmax(1), label).float().mean()


def main():
    parser = dnnutil.basic_parser()
    args = parser.parse_args()

    train_data = mydataset.Dataset(train=True)
    test_data = mydataset.Dataset(train=False)

    kwargs = {'batch_size': args.batch_size, 'num_workers': 8}
    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, shuffle=False, **kwargs)

    net = dnnutil.load_model(mymodel.Model, args.model)
    optim = torch.optim.SGD(net.parameters(), args.lr, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()

    manager = dnnutil.Manager(run_num=args.rid)
    manager.set_description('An example using dnnutil')
    trainer = dnnutil.ClassifierTrainer(net, optim, loss_fn, accuracy)
    logger = dnnutil.TextLog(manager.run_dir / 'log.txt')
    checkpointer = dnnutil.Checkpointer(manager.run_dir, period=5)

    for e in range(args.start, args.epochs):
        start = time.time()

        train_loss, train_acc = trainer.train(train_loader, e)
        test_loss, test_acc = trainer.eval(test_load, e)

        t = time.time() - start
        lr = optim.param_groups[-1]['lr']
        log.log(e, t, train_loss, train_acc, test_loss, test_acc, lr)
        check.checkpoint(net, test_loss, e)


if __name__ == '__main__':
    main()
```
