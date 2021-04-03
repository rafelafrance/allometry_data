#!/usr/bin/env python3
"""Clean real allometry sheets."""

import argparse
import textwrap
from os import makedirs
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from allometry.model_util import get_model, load_state
from allometry.util import finished, started
from allometry.allometry_sheets import AllometrySheets


def clean(args):
    """Cut, clean, and stitch allometry sheets."""
    make_dirs(args)
    model = get_model(args.model)
    load_state(args.state, model)
    device = torch.device(args.device)
    model.to(device)
    loader, dataset = get_loader(args)

    batches(model, device, loader, dataset)
    dataset.stitch_images()


def batches(model, device, loader, dataset):
    """Clean the images."""
    model.eval()
    for data in tqdm(loader):
        x, name, box = data
        x = x.to(device)
        with torch.set_grad_enabled(False):
            pred = model(x)
        dataset.save_predictions(pred, name, box)


def get_loader(args):
    """Get the data loaders."""
    size = (args.height, args.width)
    dataset = AllometrySheets(
        args.allometry_dir, args.clean_dir, crop_size=size, rotate=args.rotate)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)
    return loader, dataset


def make_dirs(args):
    """Create output directories."""
    args.clean_dir = Path(args.clean_dir) / Path(args.allometry_dir).name
    makedirs(args.clean_dir, exist_ok=True)


def parse_args():
    """Process command-line arguments."""
    description = """Use a denoising autoencoder to cleanup allometry sheets."""

    arg_parser = argparse.ArgumentParser(
        description=textwrap.dedent(description),
        fromfile_prefix_chars='@')

    arg_parser.add_argument(
        '--allometry-dir', '-A', help="""Get allometry sheets from here.""")

    arg_parser.add_argument(
        '--clean-dir', '-C', help="""Save clean sheets here.""")

    arg_parser.add_argument(
        '--state', '-s', required=True, help="""Load this state dict for cleaning.""")

    arg_parser.add_argument(
        '--model', '-m', choices=['autoencoder', 'unet'], default='autoencoder',
        help="""What model architecture to use. (default: %(default)s)
            U-Net is an untrained version from PyTorch Hub.""")

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
        '--rotate', '-R', type=int, default=0,
        help="""Rotate images by this many degrees counter-clockwise.
            (default: %(default)s)""")

    args = arg_parser.parse_args()

    if not args.device:
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    return args


if __name__ == '__main__':
    started()

    ARGS = parse_args()
    clean(ARGS)

    finished()
