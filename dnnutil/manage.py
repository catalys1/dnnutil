import torch
from pathlib import Path
import json
import re
from dnnutil.network import save_model


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
        if self.save_best:
            if val is not None and self.compare(val, self.best):
                self.best = val
                save_model(model, self.path / 'best_model')
        if self.save_multi:
            if (epoch is not None) and (epoch % self.period == self.period - 1):
                save_model(model, self.path / f'model_weights_{epoch}')
        else:   
            save_model(model, self.path / f'model_weights')


class RunException(Exception):
    '''RunException is raised when trying to resume from a run that does not
    exist.
    '''
    pass


class Manager(object):
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
    '''
    def __init__(self, root='./runs/', run_num=None, **kwargs):
        self.root = Path(root)

        create_ok = kwargs.get('create_ok', True)
        if create_ok:
            self.root.mkdir(parents=True, exist_ok=True)

        if run_num is not None:
            enforce_exists = kwargs.get('enforce_exists', True)
            self.set_run(run_num, enforce_exists)

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

        args.epochs += args.start
        args.model = str(weight_file)
        args.lr = state['lr']
        
        return args

