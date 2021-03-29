"""Common functions for dealing with fonts."""

from collections import namedtuple
from os.path import basename, splitext
# from random import choice, random

from allometry.consts import FONTS_DIR

from itertools import cycle

FONTS = FONTS_DIR.glob('*/*.ttf')
FONTS = sorted([str(f) for f in FONTS])

BOLD = [f for f in FONTS if f.casefold().find('bold') > -1]
REGULAR = [f for f in FONTS if f not in BOLD]

AugmentParams = namedtuple('AugmentParams', 'font font_size filter snow_fract')

FONT_PARAMS = {
    'B612Mono-Bold': {},
    'B612Mono-BoldItalic': {},
    'B612Mono-Italic': {},
    'B612Mono-Regular': {},
    'CourierPrime-Bold': {},
    'CourierPrime-BoldItalic': {},
    'CourierPrime-Italic': {},
    'CourierPrime-Regular': {},
    'CutiveMono-Regular': {'snow_fract': 0.25, 'filter': 'custom-max'},
    'Kingthings_Trypewriter_2': {},
    'OCRB_Medium': {},
    'OCRB_Regular': {},
    'OcrB2': {},
    'RobotoMono-Italic-VariableFont_wght': {'size': 32},
    'RobotoMono-VariableFont_wght': {'size': 32},
    'SyneMono-Regular': {},
    'VT323-Regular': {},
    'XanhMono-Italic': {'snow_fract': 0.25, 'filter': 'custom-max'},
    'XanhMono-Regular': {'snow_fract': 0.25, 'filter': 'custom-max'},
}


FONT = cycle(FONTS)


def choose_augment():
    """Randomly select a font to use for the image."""
    font = next(FONT)
    name = splitext(basename(font))[0]
    params = FONT_PARAMS.get(name, {})

    font_size = params.get('size', 36)

    filter_ = 'max'  # if is_bold else 'custom-max'
    filter_ = params.get('filter', filter_)

    snow_fract = params.get('snow_fract', 0.10)

    return AugmentParams(font, font_size, filter_, snow_fract)
