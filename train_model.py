#!/usr/bin/env python3
"""Train a model to recognize digits on allometry sheets."""

import argparse
import logging
import textwrap
from datetime import date
from os import makedirs
from pathlib import Path
from random import randint, seed

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

from allometry.model_util import MODELS, get_model, load_model_state
from allometry.training_data import TrainingData
from allometry.util import Score, finished, started


def train(args):
    """Train the neural net."""
    make_dirs(args)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        seed(args.seed)

    name = f'{args.model_arch}_{date.today().isoformat()}'
    name = f'{name}_{args.suffix}' if args.suffix else name

    model = get_model(args.model_arch)
    epoch_start = load_model_state(args.model_dir / args.model_state, model)
    epoch_end = epoch_start + args.epochs

    device = torch.device(args.device)
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    train_loader, score_loader = get_loaders(args)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_score = Score()
    best_loss = Score()

    for epoch in range(epoch_start, epoch_end):
        score = Score()

        train_batches(model, device, criterion, train_loader, optimizer, score)
        score_batches(model, device, criterion, score_loader, score)

        log_score(score, best_score, epoch)
        best_score, best_loss = save_state(
            model, args.model_dir, name, epoch, score, best_score, best_loss)


def train_batches(model, device, criterion, loader, optimizer, score):
    """Run the training phase of the epoch."""
    model.train()

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            pred = model(x)
            loss = criterion(pred, y)
            score.train_losses.append(loss.item())
            loss.backward()
            optimizer.step()


def score_batches(model, device, criterion, loader, score):
    """Run the validating phase of the epoch."""
    model.eval()

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(False):
            pred = model(x)
            loss = criterion(pred, y)
            _, idx = torch.max(pred.data, 1)
            score.score_losses.append(loss.item())
            score.total.append(y.size(0))
            score.correct_1.append((idx == y).sum().item())


def log_score(score, best_score, epoch):
    """Clean up after the scoring epoch."""
    acc_flag = '*' if score.better_than(best_score) else ''
    score_flag = '*' if score.avg_score_loss < best_score.avg_score_loss else ' '

    logging.info(f'Epoch: {epoch:3d} Average loss '
                 f'(train: {score.avg_train_loss:0.8f},'
                 f' score: {score.avg_score_loss:0.8f}) {score_flag} '
                 f'Accuracy: {score.top_1:6.4f} % {acc_flag}')


def save_state(model, model_dir, name, epoch, score, best_score, best_loss):
    """Save the model if the current score is better than the best one."""
    model.state_dict()['epoch'] = epoch

    if score.better_than(best_score):
        path = model_dir / f'best_{name}.pth'
        torch.save(model.state_dict(), path)
        best_score = score

    if score.avg_score_loss < best_loss.avg_score_loss:
        path = model_dir / f'best_loss_{name}.pth'
        torch.save(model.state_dict(), path)
        best_loss = score

    return best_score, best_loss


def get_loaders(args):
    """Get the data loaders."""
    train_dataset = TrainingData(args.train_size)
    score_dataset = TrainingData(args.score_size)

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.workers,
        worker_init_fn=lambda w: np.random.seed(np.random.get_state()[1][0] + w),
    )

    score_loader = DataLoader(
        score_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        worker_init_fn=lambda w: np.random.seed(args.seed + w),
    )

    return train_loader, score_loader


def make_dirs(args):
    """Create output directories."""
    if args.model_dir:
        makedirs(args.model_dir, exist_ok=True)
    # if args.runs_dir:
    #     makedirs(args.runs_dir, exist_ok=True)


def parse_args():
    """Process command-line arguments."""
    description = """Train a model to recognize characters on allometry sheets."""
    arg_parser = argparse.ArgumentParser(
        description=textwrap.dedent(description),
        fromfile_prefix_chars='@')

    arg_parser.add_argument(
        '--train-size', type=int, default=4096,
        help="""Train this many characters per epoch. (default: %(default)s)""")

    arg_parser.add_argument(
        '--score-size', type=int, default=512,
        help="""Train this many characters per epoch. (default: %(default)s)""")

    arg_parser.add_argument(
        '--model-dir', type=Path, help="""Save models to this directory.""")

    arg_parser.add_argument(
        '--model-state',
        help="""Load this model state to continue training the model. The file must
            be in the --model-dir.""")

    arg_parser.add_argument(
        '--model-arch', default='resnet50', choices=list(MODELS.keys()),
        help="""What model architecture to use. (default: %(default)s)""")

    arg_parser.add_argument(
        '--suffix',
        help="""Add this to the saved model name to differentiate it from
            other runs.""")

    default = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    arg_parser.add_argument(
        '--device', default=default,
        help="""Which GPU or CPU to use. Options are 'cpu', 'cuda:0', 'cuda:1' etc.
            (default: %(default)s)""")

    arg_parser.add_argument(
        '--epochs', type=int, default=100,
        help="""How many epochs to train. (default: %(default)s)""")

    arg_parser.add_argument(
        '--learning-rate', type=float, default=0.0001,
        help="""Initial learning rate. (default: %(default)s)""")

    arg_parser.add_argument(
        '--batch-size', type=int, default=16,
        help="""Input batch size. (default: %(default)s)""")

    # arg_parser.add_argument(
    #     '--top-k', type=int, default=5,
    #     help="""Get the top K predictions. (default: %(default)s)""")

    arg_parser.add_argument(
        '--workers', type=int, default=4,
        help="""Number of workers for loading data. (default: %(default)s)""")

    arg_parser.add_argument(
        '--seed', type=int, help="""Create a random seed.""")

    # arg_parser.add_argument(
    #     '--runs-dir', help="""Save tensor board logs to this directory.""")

    args = arg_parser.parse_args()

    # Wee need something for the scoring pass
    args.seed = args.seed if args.seed is not None else randint(0, 4_000_000_000)

    return args


if __name__ == '__main__':
    started()

    ARGS = parse_args()
    train(ARGS)

    finished()
