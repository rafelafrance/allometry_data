"""Common functions for training, testing, and running a model."""

import torch
from torch import nn
from torchvision import models

from allometry.const import CHARS


MODELS = {
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'resnet50': models.resnet50,
    'resnet101': models.resnet101,
    'resnext50': models.resnext50_32x4d,
    'resnext101': models.resnext101_32x8d,
}


def load_model_state(model_dir, model_state, model):
    """Load a saved model."""
    start = 1
    if model_state:
        path = model_dir / model_state
        state = torch.load(path)
        model.load_state_dict(state)
        if model.state_dict().get('epoch'):
            start = model.state_dict()['epoch'] + 1
    return start


def get_model(model_arch):
    """Get the model to use."""
    model = MODELS[model_arch]()

    if model_arch.startswith('resne'):
        model.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(model.fc.in_features, len(CHARS))

    # print(model)
    return model
