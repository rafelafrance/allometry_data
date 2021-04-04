"""Generate training data."""

import string
from random import choice, choices, getstate, randint, seed, setstate

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from torch.utils.data import Dataset

from allometry.const import CHARS, CHAR_TO_CLASS, FONTS, IMAGE_SIZE, OTHER_PUNCT, TINY_PUNCT

# How to handle each font
FONT_PARAMS = {
    '1979_dot_matrix': {'font_size_high': 40},
    'B612Mono-Bold': {'font_size_high': 48, 'filter': 'custom-median'},
    'B612Mono-Regular': {'font_size_high': 48},
    'CourierPrime-Bold': {'font_size_high': 48, 'filter': 'custom-median'},
    'CourierPrime-BoldItalic': {'font_size_high': 48, 'filter': 'custom-median'},
    'CourierPrime-Italic': {'font_size_high': 48},
    'CourierPrime-Regular': {'font_size_high': 48},
    'CutiveMono-Regular': {'font_size_high': 52},
    'DOTMATRI': {'font_size_high': 54},
    'DOTMBold': {'font_size_high': 48},
    'DottyRegular-vZOy': {'font_size_high': 72},
    'EHSMB': {'font_size_high': 48},
    'ELEKTRA_': {'font_size_high': 54},
    'Merchant Copy Doublesize': {'font_size_high': 44, 'filter': 'custom-median'},
    'Merchant Copy Wide': {'font_size_high': 48},
    'Merchant Copy': {'font_size_high': 72},
    'Minecart_LCD': {'font_size_high': 48},
    'OCRB_Medium': {'font_size_high': 48, 'filter': 'custom-median'},
    'OCRB_Regular': {'font_size_high': 48, 'filter': 'custom-median'},
    'OcrB2': {'font_size_high': 48, 'filter': 'custom-median'},
    'Ordre de Départ': {'font_size_high': 48},
    'RobotoMono-Italic-VariableFont_wght': {'font_size_high': 48},
    'RobotoMono-VariableFont_wght': {'font_size_high': 48},
    'SyneMono-Regular': {'font_size_high': 48},
    'VT323-Regular': {'font_size_high': 54},
    'XanhMono-Regular': {'font_size_high': 48},
    'fake-receipt': {'font_size_high': 48},
    'hydrogen': {'font_size_high': 54},
    'scoreboard': {'font_size_high': 48},
}

# These are used for biasing the random select of characters
WEIGHTS = [20] * len(string.digits)
WEIGHTS += [5] * len(string.ascii_uppercase)
WEIGHTS += [20] * len(TINY_PUNCT)
WEIGHTS += [1] * len(OTHER_PUNCT)


class TrainingData(Dataset):
    """Generate augmented training data."""

    def __init__(self, length):
        """Generate a dataset using pairs of images."""
        self.length = length

    def __len__(self):
        return self.length

    @staticmethod
    def get_state(seed_):
        """Get the current random state so we can return to it later."""
        rand_state = None
        if seed_ is not None:
            rand_state = getstate()
            seed(seed_)
        return rand_state

    @staticmethod
    def set_state(rand_state):
        """Continue with an existing random number generator."""
        if rand_state is not None:
            setstate(rand_state)

    def __getitem__(self, _) -> torch.uint8:
        char = choices(CHARS, WEIGHTS)[0]

        font_path = choice(FONTS)

        params = FONT_PARAMS.get(font_path.stem, {})

        size_high = params.get('font_size_high', 40)
        size_low = size_high - 4
        font_size = randint(size_low, size_high)

        font = ImageFont.truetype(str(font_path), size=font_size)
        size = font.getsize_multiline(char)

        max_left = IMAGE_SIZE[0] - size[0]
        left = randint(0, max_left) if max_left > 1 else 0

        max_top = IMAGE_SIZE[1] - size[1]
        top = randint(0, max_top) if max_top > 1 else 0

        image = Image.new('L', IMAGE_SIZE, color='black')

        draw = ImageDraw.Draw(image)
        draw.text((left, top), char, font=font, fill='white')

        snow = 0.05 if char in TINY_PUNCT else 0.1
        image = add_snow(image, snow)

        filter_ = params.get('filter', 'median')
        image = filter_image(image, filter_)

        image = image.point(lambda x: 255 if x > 128 else 0)

        data = TF.to_tensor(image)
        return data, CHAR_TO_CLASS[char]


def custom_filter(image):
    """Degrade image in realistic way."""
    image = image.filter(ImageFilter.Kernel(
        size=(3, 3), kernel=(1, 0, 1, 0, 0, 0, 1, 0, 1)))
    return image


def filter_image(image, image_filter):
    """Use filters to extend the effect of the added snow."""
    if image_filter == 'max':
        image = image.filter(ImageFilter.MaxFilter())
    elif image_filter == 'min':
        image = image.filter(ImageFilter.MinFilter())
    elif image_filter == 'median':
        image = image.filter(ImageFilter.MedianFilter())
    elif image_filter == 'custom-max':
        image = custom_filter(image)
        image = image.filter(ImageFilter.MaxFilter())
    elif image_filter == 'custom-min':
        image = custom_filter(image)
        image = image.filter(ImageFilter.MinFilter())
    elif image_filter == 'custom-median':
        image = custom_filter(image)
        image = image.filter(ImageFilter.MedianFilter())

    return image


def add_snow(image, fract=0.1):
    """Add snow (soot) to the image."""
    data = np.array(image).copy()

    shape = data.shape
    data = data.flatten()
    how_many = int(data.size * fract)
    mask = np.random.choice(data.size, how_many)
    data[mask] = 0
    data = data.reshape(shape)

    image = Image.fromarray(data)
    return image
