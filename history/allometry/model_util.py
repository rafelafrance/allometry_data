"""Common functions for training, testing, and running a model."""
import torch

from history.allometry.autoencoder import Autoencoder


def get_model(model_name):
    """Get the model to use."""
    if model_name == 'unet':
        model = torch.hub.load(
            'mateuszbuda/brain-segmentation-pytorch',
            'unet',
            in_channels=1,
            out_channels=1,
            init_features=32,
            pretrained=False)

    elif model_name == 'deeplabv3':
        model = torch.hub.load(
            'pytorch/vision:v0.9.0',
            'deeplabv3_resnet101',
            pretrained=False)

    else:
        model = Autoencoder()

    return model


def load_state(state, model):
    """Load a saved model."""
    start = 1
    if state:
        state = torch.load(state)
        model.load_state_dict(state)
        if model.state_dict().get('epoch'):
            start = model.state_dict()['epoch'] + 1
    return start
