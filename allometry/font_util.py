"""Common functions for dealing with fonts."""

from collections import namedtuple
from itertools import cycle

from allometry.const import FONTS_DIR

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
    'OCRB_Medium': {},
    'OCRB_Regular': {},
    'OcrB2': {},
    'RobotoMono-Italic-VariableFont_wght': {'size': 32},
    'RobotoMono-VariableFont_wght': {'size': 32},
}

FONTS = sorted(FONTS_DIR.glob('*/*.ttf'))
FONTS = [f for f in FONTS if f.stem in FONT_PARAMS]
FONT = cycle(FONTS)


def choose_augment():
    """Select a font to use for the image."""
    font = next(FONT)
    params = FONT_PARAMS.get(font.stem, {})

    font_size = params.get('size', 36)

    filter_ = 'max'  # if is_bold else 'custom-max'
    filter_ = params.get('filter', filter_)

    snow_fract = params.get('snow_fract', 0.10)

    return AugmentParams(str(font), font_size, filter_, snow_fract)
