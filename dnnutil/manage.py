import torch
from pathlib import Path
from dnnutil.network import save_model


class Checkpointer(object):
    '''TODO
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
        '''TODO
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

    def set_run(self, run_num=-1, enforce_exists=True):
        '''TODO
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

    def set_description(self, description='', overwrite=False):
        '''TODO
        '''
        mode = 'w' if overwrite else 'a'
        with self.run_dir.joinpath('description.txt').open(mode) as fp:
            if not description:
                description = input('Enter a description:\n')
            fp.write(description)

    def _next_rid(self):
        # Returns a new run id, which is 1 greater than the largest existant
        rid = 1
        runs = [int(x.name) for x in self.root.iterdir() if x.name.isnumeric()]
        if len(runs) > 0:
            last = max(runs)
            rid = last + 1
        return rid

