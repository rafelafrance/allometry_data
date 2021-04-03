#!/usr/bin/env python3
"""Look at at results from a model."""

import argparse
import logging
import textwrap
from os import makedirs
from os.path import join
from pathlib import Path
from random import seed

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from allometry.image_parts import ImageParts
from allometry.metrics import BinaryDiceLoss
from allometry.model_util import get_model, load_state
from allometry.util import finished, started


def test(args):
    """Test the neural net."""
    make_dirs(args)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        seed(args.seed)

    model = get_model(args.model)
    load_state(args.state, model)

    device = torch.device(args.device)
    model.to(device)

    criterion = BinaryDiceLoss()
    loader = get_loader(args)
    losses = batches(model, device, criterion, loader, args.prediction_dir)

    test_log(losses)


def batches(model, device, criterion, loader, prediction_dir):
    """Test the model."""
    losses = []
    model.eval()
    for data in tqdm(loader):
        x, y, name = data
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(False):
            pred = model(x)
            batch_loss = criterion(pred, y)
            losses.append(batch_loss.item())
        save_predictions(prediction_dir, x, y, pred, name)
    return losses


def save_predictions(prediction_dir, x_s, y_s, preds, names):
    """Save predictions for analysis."""
    if prediction_dir:
        for x, y, pred, name in zip(x_s, y_s, preds, names):
            path = join(prediction_dir, 'X', name)
            save_image(x, path)

            path = join(prediction_dir, 'Y', name)
            save_image(y, path)

            path = join(prediction_dir, 'pred', name)
            save_image(pred, path)


def test_log(losses):
    """Clean up after the validation epoch."""
    avg_loss = np.mean(losses)
    logging.info(f'Average test loss {avg_loss:0.6f}')


def get_loader(args):
    """Get the data loaders."""
    pairs = ImageParts.get_files(args.test_dir)
    size = (args.height, args.width)
    dataset = ImageParts(pairs, size=size)
    return DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)


def make_dirs(args):
    """Create output directories."""
    if args.prediction_dir:
        makedirs(Path(args.prediction_dir) / 'X', exist_ok=True)
        makedirs(Path(args.prediction_dir) / 'Y', exist_ok=True)
        makedirs(Path(args.prediction_dir) / 'pred', exist_ok=True)


def parse_args():
    """Process command-line arguments."""
    description = """Test a denoising autoencoder used to cleanup allometry sheets."""

    arg_parser = argparse.ArgumentParser(
        description=textwrap.dedent(description),
        fromfile_prefix_chars='@')

    arg_parser.add_argument(
        '--test-dir', '-T', required=True,
        help="""Read test images from the X & Y subdirectories under this one.""")

    arg_parser.add_argument(
        '--prediction-dir', '-P', help="""Save model predictions here.""")

    arg_parser.add_argument(
        '--state', '-s', required=True, help="""Load this state dict for testing.""")

    arg_parser.add_argument(
        '--model', '-m', choices=['autoencoder', 'unet'], default='autoencoder',
        help="""What model architecture to use. (default: %(default)s)""")

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
