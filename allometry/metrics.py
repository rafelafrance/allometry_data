"""Loss functions for training."""

import torch
from torch import nn


class DiceLoss(nn.Module):
    """Calculate the dice loss function for binary arrays."""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        """Calculate the dice loss function for binary arrays."""
        y_pred = torch.flatten(y_pred)
        y_true = torch.flatten(y_true)
        inter = torch.sum(y_pred * y_true)
        union = torch.sum(y_pred + y_true)
        score = (2.0 * inter + self.eps) / (union + self.eps)
        return 1.0 - score
