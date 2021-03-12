"""Loss functions for training."""


def dice_loss(actual, expect, smooth=1.0):
    """Calculate the dice loss function for binary arrays."""
    actual = actual.view(-1)
    expect = expect.view(-1)
    intersect = (actual * expect).sum()
    return 1.0 - ((2.0 * intersect + smooth) / (actual.sum() + expect.sum() + smooth))
