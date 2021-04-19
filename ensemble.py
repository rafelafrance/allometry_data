#!/usr/bin/env python3
"""Combine all of the models predictions into one value for each character.

This is an ensemble of all of the previous prediction. I'm taking the top-5 predictions
for each letter and their scores and using some sort of weighted average to arrive at
a single prediction for each character.

WARNING: This is a very weak ensemble in that is may suffer from "group think".
That is, even though the models were mostly trained on different data they are
all really variant of the same theme. It would be better to use completely
different models like random forests (and something else) instead of all neural nets.
"""

import argparse
import json
import logging
import textwrap
from collections import defaultdict
from os import makedirs
from pathlib import Path

from allometry.util import finished, started


def get_ensembles(args: argparse.Namespace) -> None:
    """Create the ensemble."""
    make_dirs(args)

    sheets = get_sheets(args)
    for sheet in sheets:
        logging.info(f'{sheet}')

        chars = get_chars(args, sheet)
        check_chars(chars)

        with open(args.ensemble_dir / sheet, 'w') as output_file:

            for char in chars.values():
                best = best_guess(char)
                rec = json.dumps(best)

                output_file.write(rec)
                output_file.write('\n')


def best_guess(guesses: list) -> dict:
    """Reduce all predictions for a character into a single 'best' prediction."""
    scores = defaultdict(int)

    for guess in guesses:
        # for char, score, weight in zip(guess['chars'], guess['scores']):
        for char, weight in zip(guess['chars'], range(5, 0, -1)):
            scores[char] += weight

    scores = sorted(scores.items(), key=lambda i: -i[1])
    scores = list(zip(*scores))

    best = {
        'chars': scores[0],
        'box': guesses[0]['box'],
        'scores': scores[1],
    }

    return best


def get_chars(args: argparse.Namespace, sheet) -> dict[list]:
    """Get all characters for every model run for the sheet."""
    chars = defaultdict(list)
    for jsonl_dir in args.jsonl_dir:
        path = jsonl_dir / sheet
        with open(path) as jsonl_file:
            for ln in jsonl_file.readlines():
                rec = json.loads(ln)
                box = rec['box']
                chars[(box['top'], box['left'])].append(rec)

    return chars


def check_chars(chars):
    """Check that the segmentation is the same for all sheets."""
    count = 0
    for value in chars.values():
        if not count:
            count = len(value)
        if len(value) != count:
            logging.error('All sheets were not segmented identically.')


def get_sheets(args: argparse.Namespace) -> list[str]:
    """Find files that match across all input jsonl directories."""
    paths = [{p.name for p in d.glob('*.jsonl')} for d in args.jsonl_dir]
    names = paths[0]
    for jsonl_dir, name_set in zip(args.jsonl_dir[1:], paths[1:]):
        names &= name_set
        if len(names) != len(paths[0]):
            logging.info(f'Mismatched files in --jsonl-dir {jsonl_dir}')
    return sorted(names)


def make_dirs(args: argparse.Namespace):
    """Create output directories."""
    if args.ensemble_dir:
        makedirs(args.ensemble_dir, exist_ok=True)


def parse_args() -> argparse.Namespace:
    """Process command-line arguments."""
    description = """Combine model predictions."""
    arg_parser = argparse.ArgumentParser(
        description=textwrap.dedent(description),
        fromfile_prefix_chars='@')

    arg_parser.add_argument(
        '--jsonl-dir', required=True, type=Path, action='append',
        help="""The directory containing the raw data in JSONL format.""")

    arg_parser.add_argument(
        '--ensemble-dir', required=True, type=Path,
        help="""Where to put the ensemble output.""")

    arg_parser.add_argument(
        '--repeat-bias', type=float, default=1.0,
        help="""Weigh guesses so that characters that appear in multiple models
            will be favored over ones in a single guess.""")

    args = arg_parser.parse_args()

    return args


if __name__ == '__main__':
    started()

    ARGS = parse_args()
    get_ensembles(ARGS)

    finished()
