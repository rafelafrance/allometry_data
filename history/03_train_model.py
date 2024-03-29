#!/usr/bin/env python3
"""Train a model."""

import argparse
import logging
import textwrap
from datetime import date
from os import makedirs
from pathlib import Path
from random import seed

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from allometry.image_parts import ImageParts
from allometry.metrics import BinaryDiceLoss
from allometry.model_util import get_model, load_state
from allometry.util import finished, started


def train(args):
    """Train the neural net."""
    make_dirs(args)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        seed(args.seed)

    name = f'{args.model}_{date.today().isoformat()}'
    name = f'{name}_{args.suffix}' if args.suffix else name

    writer = SummaryWriter(args.runs_dir)

    model = get_model(args.model)
    epoch_start = load_state(args.state, model)
    epoch_end = epoch_start + args.epochs

    device = torch.device(args.device)
    model.to(device)

    criterion = BinaryDiceLoss()

    losses = []
    train_loader, valid_loader = get_loaders(args)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_loss = 9999.9

    for epoch in range(epoch_start, epoch_end):
        train_batches(model, device, criterion, losses, train_loader, optimizer)
        msg = train_log(writer, losses, epoch)
        losses = []

        valid_batches(model, device, criterion, losses, valid_loader, args.seed)
        avg_loss = valid_log(writer, losses, epoch, msg, best_loss)
        losses = []

        best_loss = save_state(model, epoch, best_loss, avg_loss, name, args.state_dir)

    writer.flush()
    writer.close()


def train_batches(model, device, criterion, losses, loader, optimizer):
    """Run the training phase of the epoch."""
    model.train()
    for data in loader:
        x, y, *_ = data
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            pred = model(x)
            batch_loss = criterion(pred, y)
            losses.append(batch_loss.item())
            batch_loss.backward()
            optimizer.step()


def valid_batches(model, device, criterion, losses, loader, seed_):
    """Run the validating phase of the epoch."""
    model.eval()

    # Use the same validation images for each epoch i.e. same augmentations
    rand_state = ImageParts.get_state(seed_)

    for data in loader:
        x, y, name = data
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(False):
            pred = model(x)
            batch_loss = criterion(pred, y)
            losses.append(batch_loss.item())

    # Return to the current state of the training random number generator
    ImageParts.set_state(rand_state)


def train_log(writer, losses, epoch):
    """Clean up after the training epoch."""
    avg_loss = np.mean(losses)
    writer.add_scalar("Loss/train", avg_loss, epoch)
    return f'training {avg_loss:0.6f}'


def valid_log(writer, losses, epoch, msg, best_loss):
    """Clean up after the validation epoch."""
    avg_loss = np.mean(losses)
    flag = '*' if avg_loss < best_loss else ''
    writer.add_scalar("Loss/valid", avg_loss, epoch)
    logging.info(
        f'Epoch: {epoch: 3d} Average loss {msg} validation {avg_loss:0.6f} {flag}')
    return avg_loss


def save_state(model, epoch, best_loss, avg_loss, name, state_dir):
    """Save the model if the current validation score is better than the best one."""
    # TODO save optimizer too
    model.state_dict()['epoch'] = epoch
    model.state_dict()['avg_loss'] = avg_loss

    if avg_loss < best_loss:
        path = state_dir / f'best_{name}.pth'
        torch.save(model.state_dict(), path)
        best_loss = avg_loss

    return best_loss


def get_loaders(args):
    """Get the data loaders."""
    train_pairs = ImageParts.get_files(args.train_dir)
    valid_pairs = ImageParts.get_files(args.valid_dir)

    size = (args.height, args.width)

    train_dataset = ImageParts(train_pairs, size=size)
    valid_dataset = ImageParts(valid_pairs, size=size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=args.workers,
    )

    return train_loader, valid_loader


def make_dirs(args):
    """Create output directories."""
    if args.state_dir:
        makedirs(args.state_dir, exist_ok=True)
    if args.runs_dir:
        makedirs(args.runs_dir, exist_ok=True)


def parse_args():
    """Process command-line arguments."""
    description = """Train a denoising autoencoder used to cleanup allometry sheets."""
    arg_parser = argparse.ArgumentParser(
        description=textwrap.dedent(description),
        fromfile_prefix_chars='@')

    arg_parser.add_argument(
        '--train-dir', '-T', required=True,
        help="""Read training images from the X & Y subdirectories under this one.""")

    arg_parser.add_argument(
        '--valid-dir', '-V', required=True,
        help="""Read validation images from the X & Y subdirectories under this one.""")

    arg_parser.add_argument(
        '--state-dir', '-s', help="""Save best models to this directory.""")

    arg_parser.add_argument(
        '--runs-dir', '-R', help="""Save tensor board logs to this directory.""")

    arg_parser.add_argument(
        '--device', '-d',
        help="""Which GPU or CPU to use. Options are 'cpu', 'cuda:0', 'cuda:1' etc.
            We'll try to default to either 'cpu' or 'cuda:0' depending on the
            availability of a GPU.""")

    arg_parser.add_argument(
        '--model', '-m', choices=['autoencoder', 'unet'], default='autoencoder',
        help="""What model architecture to use. (default: %(default)s)""")

    arg_parser.add_argument(
        '--suffix',
        help="""Add this to the model name to differentiate it from other runs.""")

    arg_parser.add_argument(
        '--epochs', '-e', type=int, default=100,
        help="""How many epochs to train. (default: %(default)s)""")

    arg_parser.add_argument(
        '--learning-rate', '--lr', '-l', type=float, default=0.0001,
        help="""Initial learning rate. (default: %(default)s)""")

    arg_parser.add_argument(
        '--batch-size', '-b', type=int, default=16,
        help="""Input batch size. (default: %(default)s)""")

    arg_parser.add_argument(
        '--width', '-W', type=int, default=512,
        help="""Crop the images to this width. (default: %(default)s)""")

    arg_parser.add_argument(
        '--height', '-H', type=int, default=512,
        help="""Crop the images to this height. (default: %(default)s)""")

    arg_parser.add_argument(
        '--workers', '-w', type=int, default=4,
        help="""Number of workers for loading data. (default: %(default)s)""")

    arg_parser.add_argument(
        '--state', '-L', help="""Load this state dict to restart the model.""")

    arg_parser.add_argument(
        '--seed', '-S', type=int, help="""Create a random seed.""")

    args = arg_parser.parse_args()

    if args.state_dir:
        args.state_dir = Path(args.state_dir)

    if not args.device:
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    return args


if __name__ == '__main__':
    started()

    ARGS = parse_args()
    train(ARGS)

    finished()
