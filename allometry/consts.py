"""Project-wide constants."""
from pathlib import Path

SEED = 981

DATA_DIR = Path('.') / 'data'
ITIS_DIR = DATA_DIR / 'itis'
CLEAN_DIR = DATA_DIR / 'clean'
DIRTY_DIR = DATA_DIR / 'dirty'
TEXT_DIR = DATA_DIR / 'text'
