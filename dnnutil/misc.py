import torch
import numpy as np


__all__ = ['make_img']


def make_img(tensor, pad=1, rows=None):
    '''Turn a batched tensor into a grid of images.

    Given a tensor of shape [B, C, H, W], creates a new tensor of shape
    [R*(H+2p), C*(W+2p), C], where p=pad. R and C are integers, chosen such
    that floor(sqrt(B)) <= R <= C <= ceil(sqrt(B)) and abs(R * C - B) is
    minimized. In other words, the resulting tensor forms a grid of images,
    where the grid is either square or there is one more column than rows.
    If B is not square, there will be some empty grid cells. This default
    behaviour can be overrided by providing the rows argument.

    Args:
        tensor (torch.Tensor): 4D tensor containing a batch of images.
        pad (int): number of pixels to pad around each image in the grid.
        rows (int): number of rows in the image grid. If None (default),
            will be approximately equal to sqrt(B), where B is the first
            dimension of tensor.

    Returns:
        Image grid (numpy.ndarray)
    '''
    img = tensor.cpu().numpy()

    n = tensor.shape[0]
    if rows is None:
        sqn = n**.5
        r = c = int(sqn)
        if r * c < n:
            c += 1
        if r * c < n:
            r += 1
    else:
        r = rows
        cc = n / rows
        c = int(cc)
        if cc - c > 0:
            c += 1

    ip = r * c - n
    p = pad
    padding = ((0, ip), (0, 0), (p, p), (p, p))
    args = dict(mode='constant', constant_values=1)

    img = np.pad(img, padding, **args)
    cc, hh, ww = img.shape[1:]
    img = img.reshape(r, c, cc, hh, ww)
    img = img.transpose(0, 3, 1, 4, 2).reshape(r * hh, c * ww, cc)
    padding = ((p, p), (p, p), (0, 0))
    img = np.pad(img, padding, **args)
    img = img.squeeze()

    return img

