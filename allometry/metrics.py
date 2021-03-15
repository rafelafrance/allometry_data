"""Loss functions for training."""

from torch import nn


class DiceLoss(nn.Module):
    """Calculate the dice loss function for binary arrays."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def loss(self, y_pred, y_true):
        """Calculate the dice loss function for binary arrays."""
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        intersection = (y_pred * y_true).sum()
        return (1.0 - ((2.0 * intersection + self.smooth)
                       / (y_pred.sum() + y_true.sum() + self.smooth)))
