"""Loss functions for training."""

import torch
from torch import nn

THRESH = 0.96


class BinaryDiceLoss(nn.Module):
    """Calculate the dice loss function for binary classification."""

    def __init__(self, eps=1.0):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        """Calculate the dice loss function for binary arrays."""
        y_pred = torch.sigmoid(y_pred)

        y_pred = torch.flatten(y_pred)
        y_true = torch.flatten(y_true)

        inter = (y_pred * y_true).sum()
        score = (2.0 * inter + self.eps) / (y_pred.sum() + y_true.sum() + self.eps)
        return 1.0 - score
