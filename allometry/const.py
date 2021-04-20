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

SPELL = ROOT_DIR / 'allometry' / 'spell.txt'

CONTEXT_SIZE = ImageSize(32 * 4, 48)

ON, OFF = 255, 0
