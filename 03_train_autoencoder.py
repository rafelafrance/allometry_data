#!/usr/bin/env python3
"""Train a denoising autoencoder."""

import argparse
import logging
import textwrap
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from allometry.autoencoder import Autoencoder
from allometry.datasets import ImageFileDataset
from allometry.util import finished, started


def train(args):
    """Train the neural net."""
    logging.info('Starting training')
    writer = SummaryWriter()

    model = Autoencoder()
    load_model(args, model)

    device = torch.device(args.device)
    model.to(device)

    # criterion = DiceLoss()
    criterion = nn.BCEWithLogitsLoss()

    losses = []
    train_loader, valid_loader = get_loaders(args)

    best_valid = 9999.0

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        train_batches(model, device, criterion, losses, train_loader, optimizer)
        msg = train_log(writer, losses, epoch)
        losses = []

        valid_batches(model, device, criterion, losses, valid_loader)
        curr_valid = valid_log(writer, losses, epoch, msg)
        losses = []

        save_model(args, model, epoch, best_valid, curr_valid)

    writer.flush()
    writer.close()


def train_batches(model, device, criterion, losses, loader, optimizer):
    """Run the training phase of the epoch."""
    model.train()
    for data in tqdm(loader):
        x, y_true, *_ = data
        x, y_true = x.to(device), y_true.to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            y_pred = model(x)
            batch_loss = criterion(y_pred, y_true)
            losses.append(batch_loss.item())
            batch_loss.backward()
            optimizer.step()


def valid_batches(model, device, criterion, losses, loader):
    """Run the validating phase of the epoch."""
    model.eval()
    for data in tqdm(loader):
        x, y_true, *_ = data
        x, y_true = x.to(device), y_true.to(device)
        with torch.set_grad_enabled(False):
            y_pred = model(x)
            batch_loss = criterion(y_pred, y_true)
            losses.append(batch_loss.item())


def train_log(writer, losses, epoch):
    """Clean up after the training epoch."""
    avg_loss = np.mean(losses)
    writer.add_scalar("Loss/train", avg_loss, epoch)
    return f'training {avg_loss:0.6f}'


def valid_log(writer, losses, epoch, msg):
    """Clean up after the validation epoch."""
    avg_loss = np.mean(losses)
    writer.add_scalar("Loss/valid", avg_loss, epoch)
    logging.info(f'Epoch: {epoch: 3d} Average losses {msg} validation {avg_loss:0.6f}')
    return avg_loss


def save_model(args, model, epoch, best_valid, curr_valid):
    """Save the model if the current validation score is better than the best one."""
    # TODO save optimizer too
    if curr_valid < best_valid and epoch % args.save_every == 0:
        path = args.model_dir / f'best_autoencoder{epoch}.pth'
        model.state_dict()['epoch'] = epoch
        torch.save(model.state_dict(), path)


def load_model(args, model):
    """Load a saved model."""
    if args.load_model:
        state = torch.load(args.load_model)
        model.load_state_dict(state)


def get_loaders(args):
    """Get the data loaders."""
    train_split, valid_split = ImageFileDataset.split_files(
        args.dirty_dir, args.clean_dir, args.train_split, args.valid_split)

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
        '--train-split', '-t', type=float, required=True,
        help="""How many records to use for training. If the argument is
            greater than 1 than it's treated as a count. If 1 or less then it
            is treated as a fraction.""")

    arg_parser.add_argument(
        '--valid-split', '-v', type=float, required=True,
        help="""How many records to use for validation. If the argument is
            greater than 1 than it's treated as a count. If 1 or less then it
            is treated as a fraction.""")

    arg_parser.add_argument(
        '--clean-dir', '-C', help="""Read clean images from this directory.""")

    arg_parser.add_argument(
        '--dirty-dir', '-D', help="""Read dirty images from this directory.""")

    arg_parser.add_argument(
        '--model-dir', '-M', help="""Save best models to this directory.""")

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
        '--save-every', '-i', type=int, default=10,
        help="""Check every -i iterations to see if we should save a snapshot of the
            model. It only saves if the validation loss is less than the previous
            best validation loss. (default: %(default)s)""")

    arg_parser.add_argument(
        '--workers', '-w', type=int, default=4,
        help="""Number of workers for loading data. (default: %(default)s)""")

    arg_parser.add_argument(
        '--load-model', '-L', help="""Load this state dict to restart the model.""")

    args = arg_parser.parse_args()

    if args.model_dir:
        args.model_dir = Path(args.model_dir)

    if not args.device:
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    return args


if __name__ == '__main__':
    started()

    ARGS = parse_args()
    train(ARGS)

    finished()
