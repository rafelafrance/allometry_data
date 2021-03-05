"""Common functions for dealing with fonts."""

from os.path import basename, splitext
from pathlib import Path
from random import choice

FONTS_DIR = Path('.') / 'fonts'

FONTS = FONTS_DIR.glob('*/*.ttf')
FONTS = sorted([str(f) for f in FONTS])

FONT_PARAMS = {
    'B612Mono-Bold': {},
    'B612Mono-BoldItalic': {},
    'B612Mono-Italic': {},
    'B612Mono-Regular': {},
    'CourierPrime-Bold': {},
    'CourierPrime-BoldItalic': {},
    'CourierPrime-Italic': {},
    'CourierPrime-Regular': {},
    'CutiveMono-Regular': {'filter': 'custom-median', 'snow_fract': 0.1},
    'RobotoMono-Italic-VariableFont_wght': {'size': 32, 'snow_fract': 0.01},
    'RobotoMono-VariableFont_wght': {'size': 32, 'snow_fract': 0.01},
    'SyneMono-Regular': {'snow_fract': 0.01},
    'VT323-Regular': {},
    'XanhMono-Italic': {'filter': 'custom-median', 'snow_fract': 0.1},
    'XanhMono-Regular': {'filter': 'custom-median', 'snow_fract': 0.1},
    'Kingthings_Trypewriter_2': {'snow_fract': 0.01},
    'OCRB_Medium': {},
    'OCRB_Regular': {},
    'OcrB2': {},
}


def choose_font():
    """Randomly select a font to use for the image."""
    font = choice(FONTS)
    name = splitext(basename(font))[0]
    is_bold = name.lower().find('bold') > -1

    params = FONT_PARAMS.get(name, {})

    size = params.get('size', 36)

    filter_ = 'max' if is_bold else 'custom-max'
    filter_ = params.get('filter', filter_)

    snow_fract = 0.05 if is_bold else 0.04
    snow_fract = params.get('snow_fract', snow_fract)

    return font, size, filter_, snow_fract
