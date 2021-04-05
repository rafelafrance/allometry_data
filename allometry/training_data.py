"""Generate training data."""

import string
from random import choice, choices, getstate, randint, seed, setstate

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from torch.utils.data import Dataset

from allometry.const import (CHARS, CHAR_IMAGE, CHAR_TO_CLASS, FONTS, ImageSize,
                             OTHER_PUNCT, POINTS_TO_PIXELS, TINY_PUNCT)


class TrainingData(Dataset):
    """Generate augmented training data."""

    # How to handle each font
    font_params = {
        '1979_dot_matrix': {},
        'B612Mono-Bold': {'filter': 'custom-median'},
        'B612Mono-Regular': {},
        'CourierPrime-Bold': {'pt': 48, 'filter': 'custom-median'},
        'CourierPrime-Regular': {'pt': 48},
        'DOTMATRI': {},
        'DottyRegular-vZOy': {'pt': 72},
        'EHSMB': {},
        'ELEKTRA_': {'pt': 48},
        'Merchant Copy Doublesize': {'filter': 'custom-median'},
        'Merchant Copy': {'pt': 72},
        'OCRB_Medium': {'filter': 'custom-median'},
        'OCRB_Regular': {'filter': 'custom-median'},
        'Ordre de Départ': {},
        'RobotoMono-VariableFont_wght': {},
        'hydrogen': {'pt': 52},
        'scoreboard': {'pt': 52},
    }
    # These are used for biasing the random select of characters
    weights = [20] * len(string.digits)
    weights += [5] * len(string.ascii_uppercase)
    weights += [20] * len(TINY_PUNCT)
    weights += [1] * len(OTHER_PUNCT)

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
        # char = choice(CHARS)
        char = choices(CHARS, self.weights)[0]

        image = self.char_image(char)

        data = TF.to_tensor(image)
        return data, CHAR_TO_CLASS[char]

    def char_image(self, char):
        """Draw an image of the character."""
        font_path = choice(FONTS)

        params = self.font_params.get(font_path.stem, {})

        size_high = params.get('pt', int(CHAR_IMAGE.width * POINTS_TO_PIXELS))
        size_low = size_high - 2
        font_size = randint(size_low, size_high)

        font = ImageFont.truetype(str(font_path), size=font_size)
        size = font.getsize_multiline(char)
        size = ImageSize(size[0], size[1])

        image = Image.new('L', CHAR_IMAGE, color='black')

        left = (CHAR_IMAGE.width - size.width) // 2
        left = left if left > 0 else 0

        top = (CHAR_IMAGE.height - size.height) // 2
        top = top if top > 0 else 0

        draw = ImageDraw.Draw(image)
        draw.text((left, top), char, font=font, fill='white')

        image = add_soot(image, 0.2)
        filter_ = params.get('filter', 'median')

        image = filter_image(image, filter_)
        image = image.point(lambda x: 255 if x > 128 else 0)

        return image


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


def add_soot(image, fract=0.1):
    """Add soot (black pixels) to the image."""
    data = np.array(image).copy()

    shape = data.shape
    data = data.flatten()
    how_many = int(data.size * fract)
    mask = np.random.choice(data.size, how_many)
    data[mask] = 0
    data = data.reshape(shape)

    image = Image.fromarray(data)
    return image
