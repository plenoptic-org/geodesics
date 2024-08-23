import torch
from torch import Tensor
import numpy as np

def unwrap(p: Tensor, dim: int = -1) -> Tensor:
    """Unwrap by taking the complement of large deltas with respect to the period.

    This is a pytorch implementation of numpy's unwrap method, based off the
    function in
    https://discuss.pytorch.org/t/np-unwrap-function-in-pytorch/34688

    This is equivalent to numpy's implementation with period=pi/2 and
    discont=period/2, and the goal is to allow for reasonable computation of
    distance on angles.

    Parameters
    ----------
    p
        Input tensor.
    dim
        Dimension along which unwrap will operate.

    Returns
    -------
    unwrapped
        Unwrapped tensor, same shape as `p`

    """
    # goal is to add a single 0 at the beginning of whichever dimension we're
    # unwrapping along, see
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
    # for why pad is constructed this way
    pad = p.ndim * [0, 0]
    if dim >= p.ndim or dim < -p.ndim:
        raise ValueError("dim must lie within [-p.ndim, p.ndim-1], but got "
                         f"dim={dim} and p.ndim={p.ndim} instead!")
    elif dim >= 0:
        idx = 2 * (p.ndim - dim - 1)
    else:
        idx = 2 * abs(dim + 1)
    pad[idx] = 1
    dp = torch.nn.functional.pad(p.diff(dim=dim), pad)
    dp_m = ((dp+np.pi) % (2 * np.pi)) - np.pi
    dp_m[(dp_m == -np.pi) & (dp > 0)] = np.pi
    p_adj = dp_m - dp
    p_adj[dp.abs() < np.pi] = 0
    return p + p_adj.cumsum(dim)
