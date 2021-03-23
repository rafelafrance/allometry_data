"""Common functions for dealing with fonts."""

from os.path import basename, splitext
from pathlib import Path
from random import choice, random
# from itertools import cycle

FONTS_DIR = Path('.') / 'fonts'

FONTS = FONTS_DIR.glob('*/*.ttf')
FONTS = sorted([str(f) for f in FONTS])

BOLD = [f for f in FONTS if f.casefold().find('bold') > -1]
REGULAR = [f for f in FONTS if f not in BOLD]

FONT_PARAMS = {
    'B612Mono-Bold': {},
    'B612Mono-BoldItalic': {},
    'B612Mono-Italic': {},
    'B612Mono-Regular': {},
    'CourierPrime-Bold': {},
    'CourierPrime-BoldItalic': {},
    'CourierPrime-Italic': {},
    'CourierPrime-Regular': {},
    'CutiveMono-Regular': {},
    'RobotoMono-Italic-VariableFont_wght': {'size': 32},
    'RobotoMono-VariableFont_wght': {'size': 32},
    'SyneMono-Regular': {},
    'VT323-Regular': {},
    'XanhMono-Italic': {},
    'XanhMono-Regular': {},
    'Kingthings_Trypewriter_2': {},
    'OCRB_Medium': {},
    'OCRB_Regular': {},
    'OcrB2': {},
}


# CURR = cycle(FONTS)


def choose_font():
    """Randomly select a font to use for the image."""
    font = choice(BOLD) if random() < 0.5 else choice(REGULAR)
    is_bold = font.casefold().find('bold') > -1

    name = splitext(basename(font))[0]

    params = FONT_PARAMS.get(name, {})

    size = params.get('size', 36)

    filter_ = 'custom-max' if is_bold else 'custom-median'
    filter_ = params.get('filter', filter_)

    # snow_fract = 0.1 if is_bold else 0.2
    snow_fract = params.get('snow_fract', 0.2)

    return font, size, filter_, snow_fract
