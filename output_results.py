#!/usr/bin/env python3
"""Transform the raw output from the models into something researchers can use."""

import argparse
import json
import textwrap
from collections import defaultdict
from os import makedirs
from pathlib import Path
from typing import Generator

from allometry.util import finished, started

INSIDE = 30
OUTSIDE = 42
HEIGHT = 48


def output(args):
    """Output the sheets."""
    make_dirs(args)

    jsonl_paths = args.jsonl_dir.glob('*.jsonl')
    for jsonl_path in sorted(jsonl_paths):
        with open(jsonl_path) as jsonl_file:
            data = (json.loads(ln) for ln in jsonl_file.readlines())
        lines = chars_by_line(data)

        if args.text_dir:
            as_text(args, jsonl_path, lines)


def chars_by_line(data: Generator) -> dict[int, list[dict]]:
    """Organize the data by lines of text."""
    lines: dict[int, list[dict]] = defaultdict(list)
    [lines[c['box']['top']].append(c) for c in data]
    return lines


def as_text(args, jsonl_path, lines):
    """Convert the JSONL data into raw text."""
    output_path = args.text_dir / (jsonl_path.stem + '.txt')
    with open(output_path, 'w') as output_file:

        prev_ln = 0

        for ln, chars in lines.items():

            eol = round((ln - prev_ln) / HEIGHT) - 1
            for _ in range(eol):
                output_file.write('\n')

            prev_ln = ln

            text = []
            prev_left = 0

            for char in chars:
                curr_left = char['box']['left']
                space = ' ' * (round((curr_left - prev_left) / OUTSIDE) - 1)
                if space:
                    text.append(space)
                text.append(char['chars'][0])
                prev_left = curr_left

            line = ''.join(text)

            output_file.write(line)
            output_file.write('\n')
        output_file.write('\n')


def make_dirs(args):
    """Create output directories."""
    if args.text_dir:
        makedirs(args.text_dir, exist_ok=True)


def parse_args():
    """Process command-line arguments."""
    description = """Output the raw model output into a form researchers can use."""
    arg_parser = argparse.ArgumentParser(
        description=textwrap.dedent(description),
        fromfile_prefix_chars='@')

    arg_parser.add_argument(
        '--jsonl-dir', required=True, type=Path,
        help="""The directory containing the raw data in JSONL format.""")

    arg_parser.add_argument(
        '--text-dir', type=Path, help="""Where to put the text output.""")

    args = arg_parser.parse_args()

    return args


if __name__ == '__main__':
    started()

    ARGS = parse_args()
    output(ARGS)

    finished()
