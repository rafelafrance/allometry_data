#!/usr/bin/env python3
"""Train a denoising autoencoder."""

import argparse
import logging
import textwrap

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from allometry.autoencoder import Autoencoder
from allometry.datasets import ImageFileDataset
from allometry.metrics import DiceLoss
from allometry.util import finished, started


def train(args):
    """Train the neural net."""
    logging.info('Starting training')

    model = Autoencoder()

    device = torch.device(args.device)
    model.to(device)

    loss = DiceLoss()

    train_losses = []
    valid_losses = []
    train_loader, valid_loader = get_loaders(args)

    best_valid = 0.0

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        train_run(model, device, loss, train_losses, train_loader, optimizer)
        train_log(epoch)

        valid_run(model, device, loss, valid_losses, valid_loader)
        valid_log(epoch)


def train_run(model, device, loss, losses, loader, optimizer):
    """Run the training phase of the epoch."""
    model.train()
    for data in tqdm(loader):
        x, y_true, *_ = data
        x, y_true = x.to(device), y_true.to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            y_pred = model(x)
            loss = loss.loss(y_pred, y_true)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()


def valid_run(model, device, loss, losses, loader):
    """Run the validating phase of the epoch."""
    model.eval()
    for data in tqdm(loader):
        x, y_true, *_ = data
        x, y_true = x.to(device), y_true.to(device)
        with torch.set_grad_enabled(False):
            y_pred = model(x)
            loss = loss.loss(y_pred, y_true)
            losses.append(loss.item())


def train_log(epoch):
    """Clean up after the training epoch."""
    ...


def valid_log(epoch):
    """Clean up after the validation epoch."""
    ...


def get_loaders(args):
    """Get the data loaders."""
    train_split, valid_split = ImageFileDataset.split_files(
        args.dirty_dir, args.clean_dir, 0.6, 0.4)

    train_dataset = ImageFileDataset(train_split, resize=(512, 512))
    valid_dataset = ImageFileDataset(valid_split, resize=(512, 512))

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
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )

    return train_loader, valid_loader


def parse_args():
    """Process command-line arguments."""
    description = """Train a denoising autoencoder used to cleanup dirty label 
        images."""
    arg_parser = argparse.ArgumentParser(
        description=textwrap.dedent(description),
        fromfile_prefix_chars='@')

    arg_parser.add_argument(
        '--clean-dir', '-C', help="""Save the clean images to this directory.""")

    arg_parser.add_argument(
        '--dirty-dir', '-D', help="""Save the dirty images to this directory.""")

    arg_parser.add_argument(
        '--device', '-d',
        help="""Which GPU or CPU to use. Options are 'cpu', 'cuda:0', 'cuda:1' etc.
            We'll try to default to either 'cpu' or 'cuda:0' depending on the
            availability of a GPU.""")

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
        '--workers', '-w', type=int, default=4,
        help="""Number of workers for loading data. (default: %(default)s)""")

    args = arg_parser.parse_args()

    if not args.device:
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    return args


if __name__ == '__main__':
    started()

    ARGS = parse_args()
    train(ARGS)

    finished()
