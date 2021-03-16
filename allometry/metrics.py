"""Loss functions for training."""

import torch
from torch import nn

THRESH = 0.96


class BinaryDiceLoss(nn.Module):
    """Calculate the dice loss function for binary classification."""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        """Calculate the dice loss function for binary arrays."""
        y_pred = torch.flatten(y_pred).ge(THRESH).to(torch.float32)
        y_true = torch.flatten(y_true).ge(THRESH).to(torch.float32)
        inter = torch.sum(y_pred * y_true)
        union = torch.sum(y_pred + y_true)
        score = (2.0 * inter + self.eps) / (union + self.eps)
        return 1.0 - score
