"""Loss functions for training."""

import torch
from torch import nn


class BinaryDiceLoss(nn.Module):
    """Calculate the dice loss function for binary classification."""

    def __init__(self, eps=1.0):
        super().__init__()
        self.eps = eps

    def forward(self, pred, y):
        """Calculate the dice loss function for binary arrays."""
        pred = torch.flatten(pred)
        y = torch.flatten(y)

        inter = (pred * y).sum()

        score = (2.0 * inter + self.eps) / (pred.sum() + y.sum() + self.eps)
        return 1.0 - score
