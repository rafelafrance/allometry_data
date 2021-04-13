#!/usr/bin/env python3
"""Transform the raw output from run_model.py into something researcher can use."""

import argparse
import json
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Generator

from allometry.util import finished, started

INSIDE = 30
OUTSIZE = 40


def transform(args):
    """Transform the data."""
    formatter = FORMATS[args.format]

    jsonl_paths = args.jsonl_dir.glob('*.jsonl')
    for jsonl_path in sorted(jsonl_paths):
        with open(jsonl_path) as jsonl_file:
            data = (json.loads(ln) for ln in jsonl_file.readlines())
        lines = to_lines(data)
        words = to_words(lines)
        formatter(words)
        return


def to_lines(data: Generator) -> dict[int, list[dict]]:
    """Organize the data by lines of text."""
    lines: dict[int, list[dict]] = defaultdict(list)
    [lines[c['box']['top']].append(c) for c in data]

    return lines


def to_words(lines: dict[int, list]) -> dict[int, dict[str, dict]]:
    """Convert lines of text into words."""
    words = {}
    for ln, chars in lines.items():
        chars = sorted(chars, key=lambda c: c['box']['left'])

        words[ln] = {}

        prev_right = -INSIDE * 2

        groups = []
        for char in chars:
            if char['box']['left'] - prev_right > INSIDE:
                groups.append([])
            groups[-1].append(char)
            prev_right = char['box']['right']

        for group in groups:
            word = ''.join(c['char'] for c in group)
            words[ln][word] = group

    return words


def as_text(words):
    """Convert the JSONL data into raw text."""
    for ln, word_list in words.items():
        print(' '.join(word_list.keys()))


def parse_args():
    """Process command-line arguments."""
    description = """Convert raw model output into something researchers can use."""
    arg_parser = argparse.ArgumentParser(
        description=textwrap.dedent(description),
        fromfile_prefix_chars='@')

    arg_parser.add_argument(
        '--jsonl-dir', required=True, type=Path,
        help="""The directory containing the raw data in JSONL format.""")

    arg_parser.add_argument(
        '--output-dir', type=Path,
        help="""Where to put the transformed output. The default is to output to
            the stdout.""")

    arg_parser.add_argument(
        '--format', action='append', default='text', choices=list(FORMATS.keys()),
        help="""The output format. (default: %(default)s)""")

    args = arg_parser.parse_args()

    return args


if __name__ == '__main__':
    started()

    FORMATS = {
        'tsv': None,
        'csv': None,
        'text': as_text,
        'html': None,
    }

    ARGS = parse_args()
    transform(ARGS)

    finished()
