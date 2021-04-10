#!/usr/bin/env python3
"""Run a model on real allometry sheets."""

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
from torch import nn
from torch.utils.data import DataLoader

from allometry.model_util import MODELS, get_model, load_model_state
from allometry.training_data import TrainingData
from allometry.util import finished, started


def test(args):
    """Train the neural net."""
    make_dirs(args)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        seed(args.seed)

    name = f'{args.model}_{date.today().isoformat()}'
    name = f'{name}_{args.suffix}' if args.suffix else name

    model = get_model(args.model)
    epoch_start = load_model_state(args.state_dir, args.state, model)
    epoch_end = epoch_start + args.epochs

    device = torch.device(args.device)
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    losses = []
    train_loader, valid_loader = get_loaders(args)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_loss = 9999.9

    for epoch in range(epoch_start, epoch_end):
        train_batches(model, device, criterion, losses, train_loader, optimizer)
        msg = train_log(losses)
        losses = []

        valid_batches(model, device, criterion, losses, valid_loader, args.seed)
        avg_loss = valid_log(losses, epoch, msg, best_loss)
        losses = []

        best_loss = save_state(
            model, epoch, best_loss, avg_loss, name, args.state_dir, args.save_modulo)


def train_batches(model, device, criterion, losses, loader, optimizer):
    """Run the training phase of the epoch."""
    model.train()
    for x, y in loader:
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
    rand_state = TrainingData.get_state(seed_)

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(False):
            pred = model(x)
            batch_loss = criterion(pred, y)
            losses.append(batch_loss.item())

    # Return to the current state of the training random number generator
    TrainingData.set_state(rand_state)


def get_loaders(args):
    """Get the data loaders."""
    train_dataset = TrainingData(args.train_size)
    valid_dataset = TrainingData(args.valid_size)

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


def train_log(losses):
    """Clean up after the training epoch."""
    avg_loss = np.mean(losses)
    return f'training {avg_loss:0.8f}'


def valid_log(losses, epoch, msg, best_loss):
    """Clean up after the validation epoch."""
    avg_loss = np.mean(losses)
    flag = '*' if avg_loss < best_loss else ''
    logging.info(f'Epoch: {epoch:3d} Loss {msg} validation {avg_loss:0.8f} {flag}')
    return avg_loss


def save_state(model, epoch, best_loss, avg_loss, name, state_dir, save_modulo):
    """Save the model if the current validation score is better than the best one."""
    # TODO save optimizer too
    model.state_dict()['epoch'] = epoch
    model.state_dict()['avg_loss'] = avg_loss

    if avg_loss < best_loss:
        path = state_dir / f'best_{name}.pth'
        torch.save(model.state_dict(), path)
        best_loss = avg_loss

    if save_modulo:
        path = state_dir / f'last_{name}_modulo_{epoch % save_modulo}.pth'
        torch.save(model.state_dict(), path)

    return best_loss


def make_dirs(args):
    """Create output directories."""
    if args.state_dir:
        makedirs(args.state_dir, exist_ok=True)
    if args.runs_dir:
        makedirs(args.runs_dir, exist_ok=True)


def parse_args():
    """Process command-line arguments."""
    description = """Train a model to recognize characters on allometry sheets."""
    arg_parser = argparse.ArgumentParser(
        description=textwrap.dedent(description),
        fromfile_prefix_chars='@')

    arg_parser.add_argument(
        '--input-image', '-I', required=True,
        type=Path, help="""The image to convert.""")

    arg_parser.add_argument(
        '--output-data', '-O', required=True, type=Path,
        help="""The image to convert.""")

    arg_parser.add_argument(
        '--model-dir', '-m', type=Path, help="""Save models to this directory.""")

    arg_parser.add_argument(
        '--model-state', '-M',
        help="""Load this model state to continue training the model. The file must
            be in the --model-dir.""")

    arg_parser.add_argument(
        '--model', '-m', default='resnet50', choices=list(MODELS.keys()),
        help="""What model architecture to use. (default: %(default)s)""")

    default = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    arg_parser.add_argument(
        '--device', '-d', default=default,
        help="""Which GPU or CPU to use. Options are 'cpu', 'cuda:0', 'cuda:1' etc.
            (default: %(default)s)""")

    arg_parser.add_argument(
        '--batch-size', '-b', type=int, default=16,
        help="""Input batch size. (default: %(default)s)""")

    arg_parser.add_argument(
        '--workers', '-w', type=int, default=4,
        help="""Number of workers for loading data. (default: %(default)s)""")

    args = arg_parser.parse_args()

    return args


if __name__ == '__main__':
    started()

    ARGS = parse_args()
    test(ARGS)

    finished()
