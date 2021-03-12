#!/usr/bin/env python3
"""Train a denoising autoencoder."""

import argparse
import logging
import textwrap

# from tqdm import tqdm

from allometry.autoencoder import Autoencoder
from allometry.util import started, finished


def train(args):
    """Train the neural net."""
    logging.info('Starting training')
    model = Autoencoder(n_channels=1)

    for epoch in range(args.epochs):
        model.train()


def parse_args():
    """Process command-line arguments."""
    description = """Train a denoising autoencoder used to cleanup dirty images."""
    arg_parser = argparse.ArgumentParser(
        description=textwrap.dedent(description),
        fromfile_prefix_chars='@')

    arg_parser.add_argument(
        '--epochs', '-e', type=int, default=10,
        help="""How many epochs to train. (default: %(default)s)""")

    args = arg_parser.parse_args()

    return args


if __name__ == '__main__':
    started()

    ARGS = parse_args()
    train(ARGS)

    finished()
