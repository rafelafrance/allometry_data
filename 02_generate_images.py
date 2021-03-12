#!/usr/bin/env python3
"""Generate images from the text data."""

import argparse
import json
import textwrap
from pathlib import Path
from random import choice, randint, random, randrange

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from tqdm import tqdm

from allometry.consts import CLEAN_DIR, DIRTY_DIR
from allometry.font_util import choose_font

WIDTH = 4500
HEIGHT = 3440


def clean_image(data, font, font_size):
    """Generate a clean image from the text."""
    text = data['page']

    font = ImageFont.truetype(font=font, size=font_size)
    size = font.getsize_multiline(text)

    image = Image.new(mode='L', size=(WIDTH, HEIGHT), color='white')

    x, y = translate_text(size)

    txt = rotate_text(font, size, text)

    image.paste(txt, (x, y))

    return image


def rotate_text(font, size, text):
    """Rotate the text."""
    txt = Image.new('L', size=size, color='white')
    draw = ImageDraw.Draw(txt)
    draw.text((0, 0), text, font=font, fill='black')
    theta = randrange(0, 2, 1) if random() < 0.5 else randrange(358, 360, 1)
    txt = txt.rotate(theta, expand=True, fillcolor='white')
    return txt


def translate_text(size):
    """Translate the text."""
    dx = WIDTH - size[0]
    dy = HEIGHT - size[1]
    x = (dx // 2) + (randint(0, dx // 4) * choice([1, -1]))
    y = (dy // 2) + (randint(0, dy // 4) * choice([1, -1]))
    return x, y


def add_snow(data, snow_fract, low=128, high=255):
    """Add random pixels to the image to create snow."""
    shape = data.shape
    data = data.flatten()
    how_many = int(data.size * snow_fract)
    mask = np.random.choice(data.size, how_many)
    data[mask] = np.random.randint(low, high)
    data = data.reshape(shape)
    return data


def filter_image(image, image_filter):
    """Use filters to extend the effect of the added snow."""
    image = image.filter(ImageFilter.UnsharpMask())

    if image_filter == 'max':
        image = image.filter(ImageFilter.MaxFilter())
    elif image_filter == 'custom-max':
        image = custom_filter(image)
        image = image.filter(ImageFilter.MaxFilter())
    elif image_filter == 'custom-median':
        image = custom_filter(image)
        image = image.filter(ImageFilter.MedianFilter())
    elif image_filter == 'mode':
        image = image.filter(ImageFilter.ModeFilter())

    return image


def custom_filter(image):
    """This filter seems to degrade the image in realistic way."""
    image = image.filter(ImageFilter.Kernel(
        size=(3, 3),
        kernel=(-1, 0, 1, 0, -5, 0, 1, 0, -1)
    ))
    return image


def dirty_image(image, snow_fract, image_filter):
    """Make the image look like the real data as much as possible."""
    dirty = np.array(image).copy()
    dirty = add_snow(dirty, snow_fract)
    dirty = Image.fromarray(dirty)

    dirty = filter_image(dirty, image_filter)

    return dirty


def build_page(args, page):
    """Build images from the data."""
    with open(page) as data_file:
        data = json.load(data_file)

    name = page.stem + '.jpg'

    font, font_size, image_filter, snow_fract = choose_font()

    clean = clean_image(data, font, font_size)
    if args.clean_dir:
        clean.save(CLEAN_DIR / name, 'JPEG')

    dirty = dirty_image(clean, snow_fract, image_filter)
    if args.dirty_dir:
        dirty.save(DIRTY_DIR / name, 'JPEG')

    return data


def generate_images(args):
    """Generate the images for the pages."""
    pages = sorted(Path(args.text_dir).glob('*.json'))
    if args.count:
        pages = pages[:args.count]

    if args.remove_images:
        if args.clean_dir:
            for path in Path(args.clean_dir).glob('*.jpg'):
                path.unlink()
        if args.dirty_dir:
            for path in Path(args.dirty_dir).glob('*.jpg'):
                path.unlink()

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
        '--count', '-c', type=int,
        help="""How many images to create. If omitted then it will process 
            all of the files in the --text-dir.""")

    arg_parser.add_argument(
        '--clean-dir', '-C', help="""Save the clean images to this directory.""")

    arg_parser.add_argument(
        '--dirty-dir', '-D', help="""Save the dirty images to this directory.""")

    arg_parser.add_argument(
        '--remove-images', '-R', action='store_true',
        help="""Should we clear all of the existing images in the clean &
            dirty directories."""
    )

    args = arg_parser.parse_args()

    return args


if __name__ == '__main__':
    ARGS = parse_args()
    generate_images(ARGS)
