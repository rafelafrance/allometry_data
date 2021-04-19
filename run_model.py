#!/usr/bin/env python3
"""Run a model on real allometry sheets and save the raw output."""

import argparse
import json
import logging
import textwrap
from os import makedirs
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from allometry.allometry_sheet import AllometrySheet
from allometry.characters import IDX_TO_CHAR
from allometry.model_util import MODELS, get_model, load_model_state
from allometry.util import finished, started


def main(args):
    """Train the neural net."""
    make_dirs(args)

    model = get_model(args.model_arch)
    load_model_state(args.trained_model, model)

    device = torch.device(args.device)
    model.to(device)

    model.eval()

    for image_path in sorted(args.sheet_dir.glob(f'*.{args.image_suffix}')):
        logging.info(f'{image_path}')

        sheet = []

        loader = get_loader(args, image_path)

        for x, boxes in loader:
            x = x.to(device)
            with torch.set_grad_enabled(False):
                pred = model(x)
                scores, indices = torch.topk(pred, args.top_k)

            scores, indices = scores.to('cpu'), indices.to('cpu')

            save_batch(sheet, boxes, indices, scores)

        save_sheet(args.jsonl_dir, image_path, sheet)


def save_batch(sheet, boxes, indices, scores):
    """Save the current batch of characters."""
    keys = ['left', 'top', 'right', 'bottom']
    for box, score, index in zip(boxes, scores, indices):
        sheet.append({
            'chars': [IDX_TO_CHAR[i] for i in index.tolist()],
            'box': {k: v for k, v in zip(keys, box.tolist())},
            'scores': score.tolist(),
        })


def save_sheet(jsonl_dir, image_path, sheet):
    """Save the entire sheet as a JSON lines file."""
    path = jsonl_dir / (image_path.stem + '.jsonl')
    with open(path, 'w') as output_file:
        for rec in sheet:
            json.dump(rec, output_file)
            output_file.write('\n')


def get_loader(args, path):
    """Get the data loader for the sheet."""
    dataset = AllometrySheet(path, rotate=args.rotate)

    sheet_loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.workers)

    return sheet_loader


def make_dirs(args):
    """Create output directories."""
    if args.jsonl_dir:
        makedirs(args.jsonl_dir, exist_ok=True)


def parse_args():
    """Process command-line arguments."""
    description = """Train a model to recognize characters on allometry sheets."""
    arg_parser = argparse.ArgumentParser(
        description=textwrap.dedent(description),
        fromfile_prefix_chars='@')

    arg_parser.add_argument(
        '--sheet-dir', required=True, type=Path,
        help="""The directory containing the images of the allometry sheets to convert.
            They should all have the same orientation.""")

    arg_parser.add_argument(
        '--image-suffix', default='tif', help="""The suffix for the images.""")

    arg_parser.add_argument(
        '--rotate', type=int, default=0,
        help="""Rotate all images counterclockwise (in degrees) to orient
            them. Typically, you will use 0, 90, 180, or 270 degrees. Note that this
            means that all of the images in the --image-dir should all have the same
            original orientation. This is only to get image oriented properly, the
            program will fine tune the rotation (deskew) later.
            (default: %(default)s)""")

    arg_parser.add_argument(
        '--jsonl-dir', required=True, type=Path,
        help="""Where to put the output JSONL data.""")

    arg_parser.add_argument(
        '--trained-model', required=True, help="""Path to the model to use.""")

    arg_parser.add_argument(
        '--model-arch', default='resnet50', choices=list(MODELS.keys()),
        help="""What model architecture to use. (default: %(default)s)""")

    default = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    arg_parser.add_argument(
        '--device', default=default,
        help="""Which GPU or CPU to use. Options are 'cpu', 'cuda:0', 'cuda:1' etc.
            (default: %(default)s)""")

    arg_parser.add_argument(
        '--batch-size', type=int, default=16,
        help="""Input batch size. (default: %(default)s)""")

    arg_parser.add_argument(
        '--top-k', type=int, default=5,
        help="""Save the top K predictions. (default: %(default)s)""")

    arg_parser.add_argument(
        '--workers', type=int, default=4,
        help="""Number of workers for loading data. (default: %(default)s)""")

    args = arg_parser.parse_args()

    return args


if __name__ == '__main__':
    started()

    ARGS = parse_args()
    main(ARGS)

    finished()
