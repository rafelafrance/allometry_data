"""Project-wide constants."""
import string
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

CHAR_IMAGE_SIZE = ImageSize(32, 48)

TINY_PUNCT = '.-,'
OTHER_PUNCT = """$%*()<=>+/:"""
CHARS = list(string.digits + string.ascii_uppercase + TINY_PUNCT + OTHER_PUNCT)
CHAR_TO_CLASS = {c: i for i, c in enumerate(CHARS)}
CLASS_TO_CHAR = {v: k for k, v in CHAR_TO_CLASS.items()}

POINTS_TO_PIXELS = 1.333333
