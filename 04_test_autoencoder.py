#!/usr/bin/env python3
"""Look at at results from the model."""

import argparse
import logging
import textwrap
from os.path import join

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from tqdm import tqdm

# from allometry.metrics import BinaryDiceLoss
from allometry.autoencoder import Autoencoder
from allometry.datasets import ImageFileDataset
from allometry.util import finished, started


def test(args):
    """Train the neural net."""
    logging.info('Starting testing')

    model = Autoencoder()
    load_model(args, model)

    device = torch.device(args.device)
    model.to(device)

    # criterion = BinaryDiceLoss()
    criterion = nn.BCEWithLogitsLoss()

    test_loader = get_loaders(args)

    losses = test_batches(args, model, device, criterion, test_loader)

    test_log(losses)


def test_batches(args, model, device, criterion, loader):
    """Run the validating phase of the epoch."""
    losses = []
    model.eval()
    for data in tqdm(loader):
        x, y_true, width, height, name = data
        x, y_true = x.to(device), y_true.to(device)
        with torch.set_grad_enabled(False):
            y_pred = model(x)
            batch_loss = criterion(y_pred, y_true)
            losses.append(batch_loss.item())
        save_predictions(args, y_pred, name)
    return losses


def save_predictions(args, y_pred, name):
    """Save predictions for analysis"""
    if args.prediction_dir:
        for i, pred in enumerate(y_pred):
            path = join(args.prediction_dir, name[i])
            image = ToPILImage()(pred)
            image.save(path)


def test_log(losses):
    """Clean up after the validation epoch."""
    avg_loss = np.mean(losses)
    logging.info(f'Average test loss {avg_loss:0.6f}')


def get_loaders(args):
    """Get the data loaders."""
    test_split, *_ = ImageFileDataset.split_files(
        args.dirty_dir, args.clean_dir, args.test_split)

    test_dataset = ImageFileDataset(test_split, resize=(512, 512))

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )

    return test_loader


def load_model(args, model):
    """Load a saved model."""
    if args.load_model:
        state = torch.load(args.load_model)
        model.load_state_dict(state)


def parse_args():
    """Process command-line arguments."""
    description = """Train a denoising autoencoder used to cleanup dirty label
        images."""
    arg_parser = argparse.ArgumentParser(
        description=textwrap.dedent(description),
        fromfile_prefix_chars='@')

    arg_parser.add_argument(
        '--test-split', '-t', type=float, required=True,
        help="""How many records to use for testing. If the argument is
            greater than 1 than it's treated as a count. If 1 or less then it
            is treated as a fraction.""")

    arg_parser.add_argument(
        '--clean-dir', '-C', help="""Read clean images from this directory.""")

    arg_parser.add_argument(
        '--dirty-dir', '-D', help="""Read dirty images from this directory.""")

    arg_parser.add_argument(
        '--prediction-dir', '-R', help="""Save model predictions here.""")

    arg_parser.add_argument(
        '--load-model', '-L', required=True,
        help="""Load this state dict to restart the model.""")

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

    args = arg_parser.parse_args()

    if not args.device:
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    return args


if __name__ == '__main__':
    started()

    ARGS = parse_args()
    test(ARGS)

    finished()
