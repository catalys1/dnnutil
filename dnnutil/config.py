import torch
import importlib
import json
from types import SimpleNamespace


__all__ = ['configure', 'config_string', 'get_dataloaders']


def configure(config_file):
    '''TODO:docs
    '''
    cfg = _Configuration(config_file)
    return cfg


class _Configuration(object):
    '''TODO:docs
    '''
    def __init__(self, config_file):
        self.load(config_file)

    def load(self, config_file):
        cf = json.load(open(config_file))

        # Network
        model_def = cf['network']
        model_pkg = model_def.get('package', None)
        model_mod = importlib.import_module(model_def['module'], model_pkg)
        model_class = getattr(model_mod, model_def['model'])
        model_kwargs = model_def['kwargs']

        # Dataset
        data_def = cf['dataset']
        if data_def['module']:
            data_pkg = data_def.get('package', None)
            data_mod = importlib.import_module(data_def['module'], data_pkg)
            dataset = getattr(data_mod, data_def['dataset'])
            data_kwargs = data_def['kwargs']
        else:
            dataset = None
            data_kwargs = {}

        # Loss
        loss_def = cf['loss']
        if loss_def['module']:
            loss_pkg = loss_def.get('package', None)
            loss_mod = importlib.import_module(loss_def['module'], loss_pkg)
            loss = getattr(loss_mod, loss_def['loss'])
            loss_kwargs = loss_def['kwargs']
        else:
            loss = None
            loss_kwargs = {}

        # Optimizer
        optim_def = cf.get('optimizer', False)
        if optim_def:
            opt_pkg = optim_def.get('package', None)
            opt_mod = importlib.import_module(optim_def['module'], opt_pkg)
            opt = getattr(opt_mod, optim_def['optim'])
            opt_kwargs = optim_def['kwargs']
        else:
            opt = None
            opt_kwargs = {}

        # Trainer
        trainer_def = cf['trainer']
        if trainer_def['module']:
            t_pkg = trainer_def.get('package', None)
            trainer_mod = importlib.import_module(trainer_def['module'], t_pkg)
            trainer = getattr(trainer_mod, trainer_def['trainer'])
        else:
            trainer = None

        # Hyperparameters
        params  = cf['hyperparams']
        
        # Other
        other = cf.get('other', {})

        self.model = SimpleNamespace(model=model_class, args=model_kwargs)
        self.data = SimpleNamespace(data=dataset, args=data_kwargs)
        self.loss = SimpleNamespace(loss=loss, args=loss_kwargs)
        self.optim = SimpleNamespace(optim=opt, args=opt_kwargs)
        self.trainer = trainer
        self.hp = params
        self.other = other


    def __repr__(self):
        s = 'Configuration{{\n{}\n}}'.format(
            '\n'.join('{}: {!r}'.format(k, v) for k, v in self.__dict__.items()))
        return s

    def __str__(self):
        return repr(self)


def config_string(config):
    '''TODO:docs
    '''
    if type(config) is str:
        config = json.load(open(config))
    j = config

    description = j.pop('description')

    details = ''
    hp = j.pop('hyperparams', None)
    for k, v in j.items():
        v.pop('package', None)
        v.pop('module', None)
        kwargs = v.pop('kwargs', None)
        others = list(v.values())
        if others and others[0]:
            details += f'    {k}: {others[0]} {kwargs}\n'
    details += f'    {hp}\n'

    return description, details


def get_dataloaders(data, batch_size, num_workers=8, collate=None):
    '''TODO:docs
    '''
    args = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    if collate is not None:
        args['collate_fn'] = collate

    trl = torch.utils.data.DataLoader(data.train(), shuffle=True, **args)
    tre = torch.utils.data.DataLoader(data.test(), **args)

    return trl, tre

