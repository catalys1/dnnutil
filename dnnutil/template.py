import torch
import dnnutil
import time
# Make sure to import whatever you need for your data and network


def setup_data(args):
    # OVERRIDE: setup train and test data, create DataLoaders for them. Return
    # the DataLoaders.
    #
    # train_data = mydataset.Dataset(train=True)
    # test_data = mydataset.Dataset(train=False)

    # kwargs = {'batch_size': args.batch_size, 'num_workers': 8}
    # train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, **kwargs)
    # test_loader = torch.utils.data.DataLoader(test_data, shuffle=False, **kwargs)
    
    train_loader = test_loader = None
    return train_loader, test_loader


def setup_network(args):
    # OVERRIDE: Create an instance of your network and return it
    # net = dnnutil.load_model(mymodel.Model, args.model)
    net = None
    return net


def main():
    parser = dnnutil.basic_parser()
    args = parser.parse_args()
    
    # manager keeps track of where to save/load state for a particular run
    run_dir = './runs/'
    manager = dnnutil.Manager(root=run_dir, run_num=args.rid)
    manager.set_description(args.note)
    args = manager.load_state(args, restore_lr=False)

    train_loader, test_loader = setup_data(args)
    net = setup_network(args)
    # Change optim, loss_fn, and accuracy as needed
    optim = torch.optim.SGD(net.parameters(), args.lr, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()
    accuracy = dnnutil.calculate_accuracy

    # trainer handles the details of training/eval, logger keeps a log of the
    # training, checkpointer handles saving model weights
    trainer = dnnutil.ClassifierTrainer(net, optim, loss_fn, accuracy)

    for e in range(args.start, args.start + args.epochs):
        start = time.time()

        trainer.train(train_loader, e)
        trainer.eval(test_load, e)
        stats = trainer.get_stats()

        t = time.time() - start
        lr = optim.param_groups[-1]['lr']
        manager.epoch_save(net, e, t, lr, *stats)


if __name__ == '__main__':
    main()

