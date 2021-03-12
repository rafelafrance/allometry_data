#!/usr/bin/env python3
"""Train a denoising autoencoder."""

import argparse
import logging
import textwrap

import torch
import torch.optim as optim
from tqdm import trange

from allometry.autoencoder import Autoencoder
from allometry.loss import dice_loss
from allometry.util import started, finished


def train(args):
    """Train the neural net."""
    logging.info('Starting training')

    model = Autoencoder()
    device = torch.device(args.device)
    model.to(device)

    train_loader = ''
    valid_loader = ''

    best_valid = 0.0

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    step = 0

    for epoch in trange(args.epochs):
        for phase in ('train', 'valid'):
            model.train() if phase == 'train' else model.eval()


def parse_args():
    """Process command-line arguments."""
    description = """Train a denoising autoencoder used to cleanup dirty images."""
    arg_parser = argparse.ArgumentParser(
        description=textwrap.dedent(description),
        fromfile_prefix_chars='@')

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

    args = arg_parser.parse_args()

    if not args.device:
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    return args


if __name__ == '__main__':
    started()

    ARGS = parse_args()
    train(ARGS)

    finished()
