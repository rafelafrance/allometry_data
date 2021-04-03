"""Generate training data."""

import string
from random import choice, choices, randint

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from torch.utils.data import Dataset

from allometry.const import FONTS, IMAGE_SIZE


FONT_PARAMS = {
    '1979_dot_matrix': {'high': 40},
    'B612Mono-Bold': {'high': 48, 'filter': 'custom-median'},
    'B612Mono-Regular': {'high': 48},
    'CourierPrime-Bold': {'high': 48, 'filter': 'custom-median'},
    'CourierPrime-BoldItalic': {'high': 48, 'filter': 'custom-median'},
    'CourierPrime-Italic': {'high': 48},
    'CourierPrime-Regular': {'high': 48},
    'CutiveMono-Regular': {'high': 52},
    'DOTMATRI': {'high': 54},
    'DOTMBold': {'high': 48},
    'DottyRegular-vZOy': {'high': 72},
    'EHSMB': {'high': 48},
    'ELEKTRA_': {'high': 54},
    'Merchant Copy Doublesize': {'high': 44, 'filter': 'custom-median'},
    'Merchant Copy Wide': {'high': 48},
    'Merchant Copy': {'high': 72},
    'Minecart_LCD': {'high': 48},
    'OCRB_Medium': {'high': 48, 'filter': 'custom-median'},
    'OCRB_Regular': {'high': 48, 'filter': 'custom-median'},
    'OcrB2': {'high': 48, 'filter': 'custom-median'},
    'Ordre de Départ': {'high': 48},
    'RobotoMono-Italic-VariableFont_wght': {'high': 48, 'tiny': -4},
    'RobotoMono-VariableFont_wght': {'high': 48, 'tiny': -4},
    'SyneMono-Regular': {'high': 48},
    'VT323-Regular': {'high': 54},
    'XanhMono-Regular': {'high': 48},
    'fake-receipt': {'high': 48},
    'hydrogen': {'high': 54},
    'scoreboard': {'high': 48},
}

TINY_PUNCT = '.-,'
OTHER_PUNCT = """$%*()<=>?+/;:^"""
CHARS = list(string.digits + string.ascii_uppercase + TINY_PUNCT + OTHER_PUNCT)

WEIGHTS = [10] * len(string.digits)
WEIGHTS += [5] * len(string.ascii_uppercase)
WEIGHTS += [10] * len(TINY_PUNCT)
WEIGHTS += [1] * len(OTHER_PUNCT)


class TrainingData(Dataset):
    """Generate augmented training data."""

    def __init__(self, length):
        """Generate a dataset using pairs of images."""
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, _) -> torch.uint8:
        char = choices(CHARS, WEIGHTS)[0]

        font_path = choice(FONTS)

        params = FONT_PARAMS.get(font_path.stem, {})

        size_high = params.get('high', 40)
        size_low = size_high - 4
        font_size = randint(size_low, size_high)
        snow = params.get('snow', 0.2)
        filter_ = params.get('filter', 'median')
        tiny = params.get('tiny', 0)

        font = ImageFont.truetype(str(font_path), size=font_size)
        size = font.getsize_multiline(char)

        left, top = 0, 0
        if char in TINY_PUNCT:
            left, top = tiny, tiny
        else:
            if size[0] < IMAGE_SIZE[0] - 1:
                left = randint(1, (IMAGE_SIZE[0] - size[0] - 1))
            if size[1] < IMAGE_SIZE[1] - 1:
                top = randint(1, (IMAGE_SIZE[1] - size[1] - 1))

        image = Image.new('L', IMAGE_SIZE, color='black')

        draw = ImageDraw.Draw(image)
        draw.text((left, top), char, font=font, fill='white')

        if snow:
            image = add_snow(image, snow)

        if filter_:
            image = filter_image(image, filter_)

        image = image.point(lambda x: 255 if x > 128 else 0)

        data = TF.to_tensor(image)
        return data


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
