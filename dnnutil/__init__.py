from .cmdopts import *
from .logger import *
from .network import *
from .training import *
from .manage import *
from .optim import *


name = 'dnnutil'
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

template = __path__[0] + '/template.py'
config = __path__[0] + '/config.json'
