import importlib
import json
from types import SimpleNamespace


__all__ = ['configure']


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
        model_mod = importlib.import_module(model_def['module'])
        model_class = getattr(model_mod, model_def['model'])
        model_kwargs = model_def['kwargs']

        # Dataset
        data_def = cf['dataset']
        data_mod = importlib.import_module(data_def['module'])
        dataset = getattr(data_mod, data_def['dataset'])
        data_kwargs = data_def['kwargs']

        # Loss
        loss_def = cf['loss']
        loss_mod = importlib.import_module(loss_def['module'])
        loss = getattr(loss_mod, loss_def['loss'])
        loss_kwargs = loss_def['kwargs']

        # Trainer
        trainer_def = cf['trainer']
        trainer_mod = importlib.import_module(trainer_def['module'])
        trainer = getattr(trainer_mod, trainer_def['trainer'])

        # Hyperparameters
        params  = cf['hyperparams']
        
        # Other
        other = cf.get('other', {})

        self.model = SimpleNamespace(model=model_class, args=model_kwargs)
        self.data = SimpleNamespace(data=dataset, args=data_kwargs)
        self.loss = SimpleNamespace(loss=loss, args=loss_kwargs)
        self.trainer = trainer
        self.hp = params
        self.other = other


    def __repr__(self):
        s = 'Configuration{{\n{}\n}}'.format(
            '\n'.join('{}: {!r}'.format(k, v) for k, v in self.__dict__.items()))
        return s

    def __str__(self):
        return repr(self)

