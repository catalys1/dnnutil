import torch
from pathlib import Path
import shutil


__all__ = ['tocuda', 'load_model', 'save_model']


def tocuda(seq):
    '''Convert all Tensors in seq to CUDA Tensors.
    
    Args:
        seq (Tensor or iterable): A Tensor or sequence of Tensors.

    Returns:
        A sequence of CUDA Tensors, matching the input sequence.
    '''
    if torch.cuda.is_available():
        if not isinstance(seq, torch.Tensor):
            return [x.cuda() for x in seq]
        else:
            return seq.cuda()
    return seq


def load_model(model_class, checkpoint=None, strict=False, cpu=False, **kwargs):
    '''Instantiates a deep network model and optionally loads weights from
    a checkpoint file. If cuda GPUs are available, the model will be converted
    via .cuda(). If multiple GPUs are available, the model will be loaded on
    each one through torch.nn.DataParallel.

    Args:
        model_class (torch.nn.Module): An uninstantiated model class. The
            network will be created as: "net = model_class(**kwargs)".
        checkpoint (str): Path to a checkpoint file. The file should have
            been created through a call to torch.save, and the saved
            paramters should match model_class. If None, no checkpoint is
            loaded. Default: None.
        strict (bool): Boolean indicating whether parameter name-matching
            should be performed strictly. This value is passed to
            net.load_state_dict in the event that a checkpoint is to be
            loaded. Default: False.
        cpu (bool): Boolean used to force the model to be on the CPU, even
            if a GPU is available. Default: False.

    Any other named arguments will be passed to model_class constructor.

    Returns:
        net (torch.nn.Module): The instantiated network, with weights
            optionally restored from a model checkpoint.
    '''
    print('Loading network...')
    net = model_class(**kwargs)
    cuda = torch.cuda.is_available() and not cpu
    if cuda:
        net = net.cuda()
    if checkpoint:
        if cuda:
            params = torch.load(checkpoint)
        else:
            # Convert weights saved from GPU to compatible CPU weights
            params = torch.load(checkpoint, map_location=lambda s, l: s)
        net.load_state_dict(params, strict=strict)
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    return net


def save_model(net, path):
    '''Save a checkpoint of a models weights.

    Args:
        net (torch.nn.Module): An instantiated network. The networks
            parameters will be saved in a checkpoint file.
        path (str): Path to the save file.
    '''
    params = net.state_dict()
    # If the model is distributed over multiple GPUs, then saving the
    # parameters directly will result in a checkpoint that can only be
    # loaded by a distributed network. By stripping off the 'module.' from
    # the parameter names, the weights can be loaded onto a non-distributed
    # model (which can then be distributed if desired).
    if next(iter(params.keys())).startswith('module.'):
        from collections import OrderedDict
        new_dict = OrderedDict()
        for k, v in params.items():
            new_dict[k[7:]] = v
        params = new_dict
    torch.save(params, path)

