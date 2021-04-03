"""Project-wide constants."""

from pathlib import Path

ROOT_DIR = Path('.') if str(Path.cwd()).endswith('allometry_data') else Path('..')

DATA_DIR = ROOT_DIR / 'data'
ITIS_DIR = DATA_DIR / 'itis'
TEXT_DIR = DATA_DIR / 'text'

FONTS_DIR = ROOT_DIR / 'fonts'
FONTS = sorted(FONTS_DIR.glob('*/*.ttf'))

IMAGE_SIZE = (32, 48)
