from .cmdopts import basic_parser
from .logger import TextLog
from .network import tocuda, load_model, save_model
from .training import (Trainer, ClassifierTrainer, AutoencoderTrainer,
                       calculate_accuracy)
from .manage import Checkpointer, Manager
from .optim import EpochSetLR


name = 'dnnutil'
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

template = __path__[0] + '/template.txt'
config = __path__[0] + '/config.json'
