import torch
from pathlib import Path
import json
import shutil
from dnnutil.network import save_model
from dnnutil.logger import TextLog
from dnnutil.config import configure


__all__ = ['Checkpointer', 'RunException', 'Manager', 'ConfigManager']


class Checkpointer(object):
    '''The Checkpointer class handles saving model checkpoints. It can be set
    to save model weights periodically, keeping multiple copies or a single
    most recent copy, as well as keeping a seperate checkpoint of the
    best-so-far weights according to a metric.

    Args:
        checkpoint_dir (str or Path): Directory to save checkpoints in.
        save_multi (bool): When True, a new checkpoint file will be created
            each time a checkpoint is saved. Otherwise the previous checkpoint
            will be overwritten.
        period (int): How often to save a new checkpoint (in epochs).
            Default: 1.
        save_best (bool): If True, saves a seperate checkpoint that is updated
            with the best model weights so far. Default: True.
        metric (str): Metric used to measure model performance. Can be one of
            {"loss", "accuracy"}. Default: "loss".
    '''
    def __init__(self, checkpoint_dir, save_multi=False, period=1, 
                 save_best=True, metric='loss'):
        self.path = Path(checkpoint_dir)

        self.save_multi = save_multi
        if self.save_multi:
            self.period = period

        self.save_best = save_best
        if metric == 'loss':
            self.best = 1e8
            self.compare = lambda x, y: x < y
        elif metric == 'accuracy':
            self.best = 0
            self.compare = lambda x, y: x > y

    def checkpoint(self, model, val=None, epoch=None):
        '''Save model weights in a checkpoint file.

        Args:
            model (torch.nn.Module): The network for which to save weights.
            val (float): Value of the metric being used to determine the best
                model. Default: None.
            epoch (int): Most recently processed epoch. Used to postscript
                checkpoint files when save_multi is True. Default: None.
        '''
        model_path = self.path / 'model_weights'
        save_model(model, model_path)
        if self.save_best:
            if val is not None and self.compare(val, self.best):
                self.best = val
                best_path = self.path / 'best_model'
                shutil.copy(model_path, best_path)
                # save_model(model, self.path / 'best_model')
        if self.save_multi:
            if (epoch is not None) and (epoch % self.period == 0):
                curr_path = self.path / f'model_weights_{epoch}'
                shutil.copy(model_path, curr_path)
                # save_model(model, self.path / f'model_weights_{epoch}')


class RunException(Exception):
    '''RunException is raised when trying to resume from a run that does not
    exist.
    '''
    pass


class _Manager(object):
    '''Base class for managers.
    '''
    def __init__(self, root='./runs/', run_num=None, **kwargs):
        self.root = Path(root)

        create_ok = kwargs.pop('create_ok', True)
        if create_ok:
            self.root.mkdir(parents=True, exist_ok=True)

        enforce_exists = kwargs.pop('enforce_exists', True)
        if run_num is None:
            self.run_dir = self.root
        else:
            self.set_run(run_num, enforce_exists)

        self.checkpointer = Checkpointer(self.run_dir, **kwargs)
        self.logger = TextLog(self.run_dir / 'log.txt')

    def set_run(self, run_num=0, enforce_exists=True):
        '''Set up the directory for this run.

        Args:
            run_num (int): The run id number. If 0, it gets overwritten with
                the next available run id. Default: 0.
            enforce_exists (bool): If True, and run_id is not 0, then raises
                a RunException if the run directory does not already exist.
                Default: True.
        '''
        if run_num == 0:
            run_num = self._next_rid()
        elif run_num == -1:
            run_num = self._last_rid()
        elif enforce_exists:
            run_dir = self.root.joinpath(str(run_num))
            if not run_dir.is_dir():
                raise RunException(f'No existing run at {str(run_dir)}')

        # Create the run directory
        run_dir = self.root.joinpath(str(run_num))
        run_dir.mkdir(exist_ok=True)
        
        self.run_dir = run_dir

    def _next_rid(self):
        # Returns a new run id, which is 1 greater than the largest existant
        rid = 1
        runs = [int(x.name) for x in self.root.iterdir() if x.name.isnumeric()]
        if len(runs) > 0:
            last = max(runs)
            rid = last + 1
        return rid

    def _last_rid(self):
        runs = [int(x.name) for x in self.root.iterdir() if x.name.isnumeric()]
        if len(runs) > 0:
            last = max(runs)
        else:
            raise RunException('No previous run exists')
        return last

    def log(self, epoch, time, train_loss=0, train_acc=0, test_loss=0,
            test_acc=0, lr=None):
        '''TODO
        '''
        self.logger.log(epoch, time, train_loss, train_acc,
                        test_loss, test_acc, lr)

    def log_text(self, text):
        '''TODO
        '''
        self.logger.text(text)

    def checkpoint(self, model, val=None, epoch=None):
        '''TODO
        '''
        self.checkpointer.checkpoint(model, val, epoch)

    def epoch_save(self, net, epoch, time, lr, train_loss=0, train_acc=0,
                   test_loss=0, test_acc=0):
        '''Save and log at the end of an epoch.

        Convenience method for taking care of all the end-of-epoch
        bookkeeping: writing log info, saving state, saving model
        checkpoints.

        Args:
            TODO
        '''
        self.log(epoch, time, train_loss, train_acc, test_loss, test_acc, lr)
        self.save_state(epoch, lr)
        self.checkpoint(net, test_loss, epoch)

    def save_state(self, epoch_num, lr, **kwargs):
        '''Save some minimal state about the run so that it can be easily
        continued in the case of interruption or desire to train longer.

        Args:
            epoch_num (int): Index of last epoch of training completed.
            lr (float): Last learning rate used.

        Kwargs:
            None currently.
        '''
        kwargs.update(epoch_num=epoch_num, lr=lr)
        with open(self.run_dir / '.state.json', 'w') as f:
            json.dump(kwargs, f)


class Manager(_Manager):
    '''The Manager class provides an interface for managing where output files
    are stored for a given run. This makes it easy to keep all output, logs,
    checkpoints, etc. together in one location -- particularly when running
    multiple experiments. 
    
    The output of different runs are saved in numbered folders (starting from
    1). These are saved under a root runs folder.

    Args:
        root (str or Path): Path to root folder where run data will be saved.
            Default: "./runs/".
        run_num (int): A numeric identifier for the run. If 0, it will be set
            to 1 greater than the largest existing run id. If None, no new run
            will be created -- this will have to be done later with a call to
            set_run. Default: None.

    Kwargs:
        enforce_exists (bool): If True and run_num is something other than 0
            or None, a RunException will be raised if a run for run_num does
            not already exist. Default: True.
        create_ok (bool): If True, allows the Manager to create the directory
            given by `root`, including all intermediate directories, if any.
            Default: True.

    Additional keyword arguments will be passed to the constructor of
    Checkpointer.
    '''
    def set_description(self, description='', overwrite=False):
        '''Writes a description to description.txt in the run folder.

        Args:
            description (str): A textual description of this run. This can
                be anything that you would want to know or remember about this
                run, such as how it differs from other runs. If the description
                is empty or None, the user will be prompted to enter
                a description at the prompt. Default: "" (empty).
            overwrite (bool): If True, overwrite description.txt (if it
                exists). Otherwise, append to it. Default: False.
        '''
        mode = 'w' if overwrite else 'a'
        with self.run_dir.joinpath('description.txt').open(mode) as fp:
            if not description:
                description = input('Enter a description:\n')
            if description:
                print(description, file=fp)

    def load_state(self, args, match_model_epoch=False, use_best=False,
                   restore_lr=True):
        '''Restore the state of training at the time it finished or was
        interrupted. If args.resume is False, this simply return args.
        Otherwise, it changes the values in args to reflect the saved state,
        then returns it. Note that this only affects args - this will not
        restore the state of the momentum variables in the optimizer, for
        instance.

        Args:
            args (Namespace): A Namespace object, presumably created by the
                argument parser.
            match_model_epoch (bool): When True, tries to set the starting
                epoch based on the filename of the most recently saved weight
                checkpoint.
            use_best (bool): Restore the model weights from the best-so-far
                checkpoint. This is mutually exclusive with match_model_epoch,
                which will be ignored when use_best is set True.
            restore_lr (bool): When True, the learning rate will be set to the
                lr saved in the state file. Otherwise, it will remain set to
                whatever was passed in through args.
        '''
        if not hasattr(args, 'resume') or not args.resume:
            return args

        f = self.run_dir / '.state.json'
        if not f.is_file():
            raise RunException(f'Attempted to load state at {str(f)}, but no '
                'saved state exists')
        with open(f, 'r') as f:
            state = json.load(f)

        try:
            model_weight_files = sorted(
                self.run_dir.glob('model_weights*'),
                key=lambda x: int(x.name.split('_')[-1])
            )
            weight_file = model_weight_files[-1]
        except ValueError as e:
            # Error raised when the file has no numeric postscript
            weight_file = self.run_dir / 'model_weights'

        args.start = state['epoch_num'] + 1

        if use_best:
            weight_file = self.run_dir / 'best_model'
        elif match_model_epoch:
            if weight_file.name.split('_')[-1].is_numeric():
                n = int(weight_file.name.split('_')[-1])
                args.start = n + 1

        args.model = str(weight_file)

        if restore_lr:
            args.lr = state['lr']
        
        return args


class ConfigManager(_Manager):
    '''A manager class for config-file based experiments.

    This manager handles the same general tasks as Manager, such as
    keeping track of different experimental runs including logging and
    checkpointing models, and saving state so that experiments can be
    picked up where they left off. Furthermore, this manager also sets
    up a configuration and tracks the config file.

    Args:
        root (str or Path): Path to root folder where run data will be saved.
            Default: "./runs/".
        run_num (int): A numeric identifier for the run. If 0, it will be set
            to 1 greater than the largest existing run id. If None, no new run
            will be created -- this will have to be done later with a call to
            set_run. Default: None.

    Kwargs:
        enforce_exists (bool): If True and run_num is something other than 0
            or None, a RunException will be raised if a run for run_num does
            not already exist. Default: True.
        create_ok (bool): If True, allows the Manager to create the directory
            given by `root`, including all intermediate directories, if any.
            Default: True.

    Additional keyword arguments will be passed to the constructor of
    Checkpointer.
    '''
    def setup(self, args):
        '''Setup the experiment.

        Returns an object with the configuration data for an experiment.

        Args:
            args (namespace): A namespace object, presumably created by
                an argparse.ArgumentParser. Should contain a field
                called setup, which takes on of the following values:
                'start', for initializing a new experiment; 'continue',
                for resuming an experiment; or a callable that will be
                executed if not one of the other choices.

        Returns:
            Configuration or None: An object with the configuration
                if args.setup was one of 'start' or 'continue',
                oterhwise None.
        '''
        if args.setup == 'start':
            shutil.copy(args.config, self.run_dir / 'config.json')
            cfg = configure(args.config)
            self.start_state(cfg, args)
            return cfg
        elif args.setup == 'continue':
            return self.load_state(args)
        else:
            raise ValueError('{} is not a supported setup mode'.format(args.setup))

    def start_state(self, cfg, args):
        '''
        '''
        self._cfg2args(cfg, args)
        args.start = 0

    def _cfg2args(self, cfg, args):
        for k in cfg.hp:
            if not hasattr(args, k) or getattr(args, k) is None:
                setattr(args, k, cfg.hp[k])


    def load_state(self, args):
        '''Load state necessary to resume a previously halted
        experiment.

        Updates (in place) the input args dictionary with the
        previously saved state. Also loads the configuration from the
        saved configuration file. This function should only be called
        if the state and config file were previously saved in the run
        directory.

        Args:
            args (namespace): A namespace, presumably generated by an
                Argparse.ArgumentParser. This will be edited in place
                with the previously saved arguments.

        Returns:
            dict: The configuration dictionary.
        '''
        f = self.run_dir / '.state.json'
        if not f.is_file():
            raise RunException(f'Attempted to load state at {str(f)}, but no '
                'saved state exists')
        with open(f, 'r') as f:
            state = json.load(f)

        weight_file = self.run_dir / 'model_weights'
        args.model = str(weight_file)
        args.start = state['epoch_num'] + 1
        if 'lr' not in args or args.lr == None:
            args.lr = state['lr']

        cfg = configure(self.run_dir / 'config.json')
        self._cfg2args(cfg, args)
        
        return cfg

