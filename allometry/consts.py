"""Project-wide constants."""
from pathlib import Path

ROOT_DIR = Path('.') if str(Path.cwd()).endswith('allometry_data') else Path('..')

FONTS_DIR = ROOT_DIR / 'fonts'
DATA_DIR = ROOT_DIR / 'data'
ITIS_DIR = DATA_DIR / 'itis'
TEXT_DIR = DATA_DIR / 'text'
