#!/usr/bin/env python3
"""Train a denoising autoencoder."""

import argparse
import logging
import textwrap
from os.path import join
from pathlib import Path
from random import seed

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from allometry.autoencoder import Autoencoder
from allometry.datasets import ImageFileDataset
from allometry.metrics import BinaryDiceLoss
from allometry.util import finished, started


def train(args):
    """Train the neural net."""
    if args.seed is not None:
        torch.manual_seed(args.seed)
        seed(args.seed)

    writer = SummaryWriter(args.runs_dir)

    model = get_model(args)
    epoch_start = load_state(args, model)
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

        valid_batches(args, model, device, criterion, losses, valid_loader, epoch)
        avg_loss = valid_log(writer, losses, epoch, msg, best_loss)
        losses = []

        best_loss = save_state(args, model, epoch, best_loss, avg_loss)

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


def valid_batches(args, model, device, criterion, losses, loader, epoch):
    """Run the validating phase of the epoch."""
    model.eval()
    for data in loader:
        x, y, name = data
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(False):
            pred = model(x)
            batch_loss = criterion(pred, y)
            losses.append(batch_loss.item())
        save_predictions(args, x, y, pred, name, epoch)


def save_predictions(args, x, y, pred, name, epoch):
    """Save predictions for analysis"""
    if args.prediction_dir:
        for x_, y_, pred_, name_ in zip(x, y, pred, name):
            path = join(args.prediction_dir, 'x', f'{epoch}_{name_}')
            save_image(x_, path)

            path = join(args.prediction_dir, 'y', f'{epoch}_{name_}')
            save_image(y_, path)

            path = join(args.prediction_dir, 'pred', f'{epoch}_{name_}')
            save_image(pred_, path)


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


def save_state(args, model, epoch, best_loss, avg_loss):
    """Save the model if the current validation score is better than the best one."""
    # TODO save optimizer too

    model.state_dict()['epoch'] = epoch
    model.state_dict()['avg_loss'] = avg_loss

    if args.save_every and epoch % args.save_every == 0:
        path = args.state_dir / f'save_{args.model}_{epoch}.pth'
        torch.save(model.state_dict(), path)

    if args.save_best and avg_loss < best_loss:
        path = args.state_dir / f'best_{args.model}.pth'
        torch.save(model.state_dict(), path)
        best_loss = avg_loss

    return best_loss


def load_state(args, model):
    """Load a saved model."""
    start = 1
    if args.load_state:
        state = torch.load(args.load_state)
        model.load_state_dict(state)
        start = model.state_dict()['epoch'] + 1
    return start


def get_loaders(args):
    """Get the data loaders."""
    train_pairs = ImageFileDataset.get_files(args.train_dir)
    valid_pairs = ImageFileDataset.get_files(args.valid_dir)

    size = (args.height, args.width)

    train_dataset = ImageFileDataset(train_pairs, size=size)
    valid_dataset = ImageFileDataset(valid_pairs, size=size)

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
        '--prediction-dir', '-P', help="""Save model predictions here.
            For debugging.""")

    arg_parser.add_argument(
        '--device', '-d',
        help="""Which GPU or CPU to use. Options are 'cpu', 'cuda:0', 'cuda:1' etc.
            We'll try to default to either 'cpu' or 'cuda:0' depending on the
            availability of a GPU.""")

    arg_parser.add_argument(
        '--model', '-m', choices=['autoencoder', 'unet'], default='autoencoder',
        help="""What model architecture to use. (default: %(default)s)
            U-Net and DeepLabV3-ResNet101 are untrained versions from PyTorch Hub.""")

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
        '--save-every', '-i', type=int,
        help="""Check every -i iterations to see if we should save a snapshot of the
            model. (default: %(default)s)""")

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
        '--load-state', '-L',
        help="""Load this state dict to restart the model.""")

    arg_parser.add_argument(
        '--no-save-best', '-B', action='store_false', dest='save_best',
        help="""Save the model with the best validation score.""")

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