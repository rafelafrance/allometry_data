#!/usr/bin/env python3
"""Transform the raw output from the models into something researchers can use."""

import argparse
import csv
import json
import logging
import string
import textwrap
from collections import defaultdict
from os import makedirs
from pathlib import Path
from typing import Generator

import enchant

from allometry.const import SPELL
from allometry.util import finished, started

Rows = list[list[str]]

INSIDE = 30
OUTSIDE = 42
HEIGHT = 48

LETTERS = set(list(string.ascii_uppercase))
NUMBERS = set(list(string.digits) + list('.-'))

DICT = enchant.DictWithPWL('en_US', str(SPELL))


def output(args: argparse.Namespace) -> None:
    """Output the sheets."""
    make_dirs(args)

    jsonl_paths = args.jsonl_dir.glob('*.jsonl')
    for jsonl_path in sorted(jsonl_paths):
        logging.info(f'{jsonl_path}')

        with open(jsonl_path) as jsonl_file:
            data = (json.loads(ln) for ln in jsonl_file.readlines())
        chars = chars_by_line(data)
        rows = lines_of_text(chars)
        rows = post_process(rows)

        if args.text_dir:
            output_text(args.text_dir, jsonl_path, rows, 'txt')

        if args.tsv_dir:
            output_csv(args.tsv_dir, jsonl_path, rows, 'tsv')


def post_process(rows: Rows) -> Rows:
    """Try and fix spelling errors and problems with numbers."""
    new_rows: Rows = []
    for row in rows:
        new_row = []
        for cell in row:
            words = []
            for word in cell.split():
                half = len(word) // 2
                letters = sum(map(word.count, LETTERS)) > half
                numbers = sum(map(word.count, NUMBERS)) > half

                new_word = word

                if len(word) <= 2:
                    pass
                elif numbers:
                    new_word = word.replace('O', '0')
                elif letters and not DICT.check(word):
                    suggest = DICT.suggest(word)
                    new_word = suggest[0] if suggest else word

                words.append(new_word)
            new_row.append(' '.join(words))

        new_rows.append(new_row)
    return new_rows


def chars_by_line(data: Generator) -> dict[int, list[dict]]:
    """Organize the data by lines of text."""
    lines: dict[int, list[dict]] = defaultdict(list)
    [lines[c['box']['top']].append(c) for c in data]
    return lines


def lines_of_text(line_chars) -> Rows:
    """Convert the JSONL data into raw text."""
    lines: list[str] = []

    prev_ln = 0

    for ln, chars in line_chars.items():
        eol = round((ln - prev_ln) / HEIGHT) - 1
        [lines.append('') for _ in range(eol)]

        prev_ln = ln

        text = []
        prev_left = 0

        for char in chars:
            curr_left = char['box']['left']

            spaces = round((curr_left - prev_left) / OUTSIDE) - 1
            if spaces == 1:
                text.append(' ')
            elif spaces > 1:
                text.append('\t')

            text.append(char['chars'][0])
            prev_left = curr_left

        line = ''.join(text)
        lines.append(line.strip())

    rows = [ln.split('\t') for ln in lines]
    return rows


def output_csv(
        csv_dir: Path,
        jsonl_path: Path,
        rows: Rows,
        ext: str,
        delimiter: str = '\t',
) -> None:
    """Convert the JSONL data into a CSV file."""
    output_path = csv_dir / (jsonl_path.stem + f'.{ext}')
    with open(output_path, 'w') as output_file:
        writer = csv.writer(output_file, delimiter=delimiter)
        for row in rows:
            writer.writerow(row)


def output_text(
        text_dir: Path,
        jsonl_path: Path,
        rows: Rows,
        ext: str,
) -> None:
    """Convert the JSONL data into raw text."""
    output_path = text_dir / (jsonl_path.stem + f'.{ext}')
    with open(output_path, 'w') as output_file:
        for row in rows:
            output_file.write('\t'.join(row))
            output_file.write('\n')


def make_dirs(args):
    """Create output directories."""
    if args.text_dir:
        makedirs(args.text_dir, exist_ok=True)
    if args.tsv_dir:
        makedirs(args.tsv_dir, exist_ok=True)


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

    arg_parser.add_argument(
        '--tsv-dir', type=Path, help="""Where to put the TSV output.""")

    arg_parser.add_argument(
        '--spell-check', action='store_true', help="""Try to correct spelling.""")

    args = arg_parser.parse_args()

    return args


if __name__ == '__main__':
    started()

    ARGS = parse_args()
    output(ARGS)

    finished()
