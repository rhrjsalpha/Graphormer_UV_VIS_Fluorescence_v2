import torch
from Graphormer.GP5.Custom_Loss.fast_dtw import fastdtw
from collections import defaultdict

import torch
from Graphormer.GP5.Custom_Loss.fast_dtw import fastdtw
import torch.nn as nn
#from Graphormer.GP.Custom_Loss.soft_dtw.sdtw import SoftDTW

def fastdtw_loss(pred, target, radius=1, dist=None):
    """
    FastDTW-based loss function.

    Args:
        pred (torch.Tensor): Predicted tensor.
        target (torch.Tensor): Target tensor.
        radius (int, optional): Radius for FastDTW. Defaults to 1.
        dist (function, optional): Distance metric. Defaults to None.

    Returns:
        torch.Tensor: Computed DTW distance as a loss value.
    """
    # Ensure pred and target are tensors
    pred = torch.as_tensor(pred, dtype=torch.float)
    target = torch.as_tensor(target, dtype=torch.float)

    # Compute FastDTW distance
    distance, _ = fastdtw(pred, target, radius=radius, dist=dist)

    # Safely convert distance to a tensor with requires_grad
    if isinstance(distance, torch.Tensor):
        loss = distance.clone().detach().requires_grad_(True)
    else:
        loss = torch.tensor(distance, dtype=torch.float, requires_grad=True)

    return loss


