"""Common functions for training, testing, and running a model."""

import torch
from torch import nn
from torchvision import models

from allometry.const import CHARS


def load_state(state, model):
    """Load a saved model."""
    start = 1
    if state:
        state = torch.load(state)
        model.load_state_dict(state)
        if model.state_dict().get('epoch'):
            start = model.state_dict()['epoch'] + 1
    return start


def get_model(model_name):
    """Get the model to use."""
    model = None
    if model_name == 'resnet50':
        model = models.resnet50()
    elif model_name == 'resnet101':
        model = models.resnet50()

    if model_name.startswith('resnet'):
        model.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(model.fc.in_features, len(CHARS))

    # print(model)
    return model
