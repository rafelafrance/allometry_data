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
        inter = torch.dot(y_pred.view(-1), y_true.view(-1))
        union = torch.sum(y_pred) + torch.sum(y_true)
        score = (2.0 * inter + self.eps) / (union + self.eps)
        return 1.0 - score
