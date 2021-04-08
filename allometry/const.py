"""Project-wide constants."""
from collections import namedtuple
from pathlib import Path


BBox = namedtuple('BBox', 'left top right bottom')
ImageSize = namedtuple('Size', 'width height')

ROOT_DIR = Path('.') if str(Path.cwd()).endswith('allometry_data') else Path('..')

DATA_DIR = ROOT_DIR / 'data'
ITIS_DIR = DATA_DIR / 'itis'
TEXT_DIR = DATA_DIR / 'text'

FONTS_DIR = ROOT_DIR / 'fonts'
FONTS = sorted(FONTS_DIR.glob('*/*.ttf'))

CHAR_IMAGE_SIZE = ImageSize(32 * 4, 48)

POINTS_TO_PIXELS = 1.333333
