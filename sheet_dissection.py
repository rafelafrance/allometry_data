#!/usr/bin/env python3
"""Save the allometry sheet dissection for inspection."""

import argparse
import logging
import textwrap
from os import makedirs
from pathlib import Path

from PIL import ImageDraw

from allometry.allometry_sheet import AllometrySheet
from allometry.util import finished, started

BOX_COLOR = (0, 255, 255)


def dissect(args):
    """Dissect the sheets."""
    make_dirs(args)

    image_paths = sorted(args.sheet_dir.glob(f'*.{args.image_suffix}'))
    if args.filter:
        image_paths = [p for p in image_paths if str(p).find(args.filter) > -1]

    for image_path in image_paths:
        logging.info(f'{image_path}')

        sheet = AllometrySheet(image_path, args.rotate)

        dissected = sheet.binary.convert('RGB')
        draw = ImageDraw.Draw(dissected)

        for i in range(len(sheet)):
            _, box = sheet[i]
            box = box.tolist()
            draw.rectangle(box, outline=BOX_COLOR)

        if args.show_rows:
            width = sheet.binary.size[0]
            for top, bottom in sheet.rows:
                draw.line((0, top, width, top), width=1, fill='red')
                draw.line((0, bottom, width, bottom), width=1, fill='yellow')

        dissect_path = args.dissection_dir / (image_path.stem + '.jpg')
        dissected.save(dissect_path, 'JPEG')


def make_dirs(args):
    """Create output directories."""
    if args.dissection_dir:
        makedirs(args.dissection_dir, exist_ok=True)


def parse_args():
    """Process command-line arguments."""
    description = """Save the results of image dissection."""
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
        '--dissection-dir', required=True, type=Path,
        help="""Where to put the dissected images.""")

    arg_parser.add_argument(
        '--filter', help="""Filter images.""")

    arg_parser.add_argument(
        '--show-rows', action='store_true', help="""Show rows.""")

    args = arg_parser.parse_args()

    return args


if __name__ == '__main__':
    started()

    ARGS = parse_args()
    dissect(ARGS)

    finished()
