#!/usr/bin/env python3
"""Look at at results from the model."""

import argparse
import logging
import textwrap
from os.path import join
from random import seed

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from allometry.autoencoder import Autoencoder
from allometry.datasets import ImageFileDataset
from allometry.metrics import BinaryDiceLoss
from allometry.util import finished, started


def test(args):
    """Train the neural net."""
    logging.info('Starting testing')

    if args.seed is not None:
        torch.manual_seed(args.seed)
        seed(args.seed)

    model = get_model(args)
    load_state(args, model)

    device = torch.device(args.device)
    model.to(device)

    criterion = BinaryDiceLoss()

    test_loader = get_loaders(args)

    losses = test_batches(args, model, device, criterion, test_loader)

    test_log(losses)


def test_batches(args, model, device, criterion, loader):
    """Run the validating phase of the epoch."""
    losses = []
    model.eval()
    for data in tqdm(loader):
        x, y, name = data
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(False):
            pred = model(x)
            batch_loss = criterion(pred, y)
            losses.append(batch_loss.item())
        save_predictions(args, x, y, pred, name)
    return losses


def save_predictions(args, x, y, pred, name):
    """Save predictions for analysis"""
    if args.prediction_dir:
        for x, y, pred, name in zip(x, y, pred, name):
            path = join(args.prediction_dir, 'x', name)
            save_image(x, path)

            path = join(args.prediction_dir, 'y', name)
            save_image(y, path)

            path = join(args.prediction_dir, 'pred', name)
            save_image(pred, path)


def test_log(losses):
    """Clean up after the validation epoch."""
    avg_loss = np.mean(losses)
    logging.info(f'Average test loss {avg_loss:0.6f}')


def get_loaders(args):
    """Get the data loaders."""
    test_split, *_ = ImageFileDataset.split_files(
        args.x_dir, args.y_dir, args.test_split)

    size = (args.width, args.height)

    test_dataset = ImageFileDataset(test_split, size=size)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    return test_loader


def get_model(args):
    """Get the model to use."""
    if args.model == 'unet':
        model = torch.hub.load(
            'mateuszbuda/brain-segmentation-pytorch',
            'unet',
            in_channels=1,
            out_channels=1,
            init_features=32,
            pretrained=False)

    elif args.model == 'deeplabv3':
        model = torch.hub.load(
            'pytorch/vision:v0.9.0',
            'deeplabv3_resnet101',
            pretrained=False)

    else:
        model = Autoencoder()

    return model


def load_state(args, model):
    """Load a saved model."""
    if args.state:
        state = torch.load(args.state)
        model.load_state_dict(state)


def parse_args():
    """Process command-line arguments."""
    description = """Train a denoising autoencoder used to cleanup dirty label
        images."""
    arg_parser = argparse.ArgumentParser(
        description=textwrap.dedent(description),
        fromfile_prefix_chars='@')

    arg_parser.add_argument(
        '--test-split', '-t', type=int, required=True,
        help="""How many records to use for testing.""")

    arg_parser.add_argument(
        '--x-dir', '-X', help="""Read X images from this directory.""")

    arg_parser.add_argument(
        '--y-dir', '-Y', help="""Read Y images from this directory.""")

    arg_parser.add_argument(
        '--prediction-dir', '-P', help="""Save model predictions here.""")

    arg_parser.add_argument(
        '--state', '-s', required=True, help="""Load this state dict for testing.""")

    arg_parser.add_argument(
        '--model', '-m', choices=['autoencoder', 'unet'], default='autoencoder',
        help="""What model architecture to use. (default: %(default)s)
            U-Net and DeepLabV3-ResNet101 are untrained versions from PyTorch Hub.""")

    arg_parser.add_argument(
        '--device', '-d',
        help="""Which GPU or CPU to use. Options are 'cpu', 'cuda:0', 'cuda:1' etc.
            We'll try to default to either 'cpu' or 'cuda:0' depending on the
            availability of a GPU.""")

    arg_parser.add_argument(
        '--batch-size', '-b', type=int, default=16,
        help="""Input batch size. (default: %(default)s)""")

    arg_parser.add_argument(
        '--workers', '-w', type=int, default=4,
        help="""Number of workers for loading data. (default: %(default)s)""")

    arg_parser.add_argument(
        '--width', '-W', type=int, default=512,
        help="""Crop the images to this width. (default: %(default)s)""")

    arg_parser.add_argument(
        '--height', '-H', type=int, default=512,
        help="""Crop the images to this height. (default: %(default)s)""")

    arg_parser.add_argument(
        '--seed', '-S', type=int, help="""Create a random seed.""")

    args = arg_parser.parse_args()

    if not args.device:
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    return args


if __name__ == '__main__':
    started()

    ARGS = parse_args()
    test(ARGS)

    finished()
