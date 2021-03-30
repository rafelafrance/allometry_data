#!/usr/bin/env python3
"""Generate images from the text data."""

import argparse
import json
import textwrap
from os import makedirs
from pathlib import Path
from random import seed, random

from tqdm import tqdm

from allometry.font_util import choose_augment
from allometry.page_image import x_snow_filter, y_image, ablate_pixels

WIDTH = 4500
HEIGHT = 3440


def build_page(args, page):
    """Build images from the data."""
    with open(page) as data_file:
        data = json.load(data_file)

    name = page.stem + '.jpg'

    aug = choose_augment()

    y = y_image(data, aug.font, aug.font_size, WIDTH, HEIGHT)
    if args.y_dir:
        y.save(args.y_dir / name, 'JPEG')

    # if random() < 0:
    #     x = ablate_pixels(y)
    # else:
    #     x = x_snow_filter(y, aug.snow_fract, aug.filter)
    x = ablate_pixels(y)
    if args.x_dir:
        x.save(args.x_dir / f'{Path(aug.font).stem}_{name}', 'JPEG')
        # x.save(args.x_dir / name, 'JPEG')


def make_dirs(args):
    """Create output directories."""
    if args.x_dir:
        makedirs(args.x_dir, exist_ok=True)
    if args.y_dir:
        makedirs(args.y_dir, exist_ok=True)

    if args.remove_images:
        if args.y_dir:
            for path in args.y_dir.glob('*.jpg'):
                path.unlink()
        if args.x_dir:
            for path in args.x_dir.glob('*.jpg'):
                path.unlink()


def generate_images(args):
    """Generate the images for the pages."""
    make_dirs(args)

    if args.seed is not None:
        seed(args.seed)

    existing = {p.stem for p in args.y_dir.glob('*.jpg')}
    existing = set() if args.remove_images else existing

    pages = [p for p in args.text_dir.glob('*.json') if p.stem not in existing]
    pages = sorted(pages)

    if args.count:
        pages = pages[:args.count]

    for page in tqdm(pages):
        build_page(args, page)


def parse_args():
    """Process command-line arguments."""
    description = """Generate images from text."""
    arg_parser = argparse.ArgumentParser(
        description=textwrap.dedent(description),
        fromfile_prefix_chars='@')

    arg_parser.add_argument(
        '--text-dir', '-t', required=True,
        help="""Where is the text data stored.""")

    arg_parser.add_argument(
        '--x-dir', '-X', help="""Save the x images to this directory.""")

    arg_parser.add_argument(
        '--y-dir', '-Y', help="""Save the y images to this directory.""")

    arg_parser.add_argument(
        '--remove-images', '-R', action='store_true',
        help="""Should we clear all of the existing images in the --x-dir &
            --y-dir.""")

    arg_parser.add_argument(
        '--count', '-c', type=int,
        help="""How many images to create. If omitted then it will process
            all of the files in the --text-dir.""")

    arg_parser.add_argument(
        '--seed', '-S', type=int,
        help="""Create a random seed for python. Note: SQLite3 does not
            use seeds. (default: %(default)s)""")

    args = arg_parser.parse_args()
    if args.text_dir:
        args.text_dir = Path(args.text_dir)
    if args.y_dir:
        args.y_dir = Path(args.y_dir)
    if args.x_dir:
        args.x_dir = Path(args.x_dir)

    return args


if __name__ == '__main__':
    ARGS = parse_args()
    generate_images(ARGS)
