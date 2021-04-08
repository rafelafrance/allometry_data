"""Generate training data."""

import string
from random import choice, choices, getstate, randint, seed, setstate

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from torch.utils.data import Dataset

from allometry.const import (CHAR_IMAGE_SIZE, CHAR_TO_CLASS, CHARS, FONTS,
                             OTHER_PUNCT, POINTS_TO_PIXELS, TINY_PUNCT,
                             ImageSize)


class TrainingData(Dataset):
    """Generate augmented training data."""

    # How to handle each font
    font_params = {
        '1979_dot_matrix': {'.': '.•', 'pt': 40},
        'B612Mono-Bold': {'.': '.•●', 'pt': 48, 'filter': 'custom-median'},
        'B612Mono-Regular': {'.': '.•●', 'pt': 48},
        'CourierPrime-Bold': {'pt': 48, 'filter': 'custom-median'},
        'CourierPrime-BoldItalic': {'pt': 48, 'filter': 'custom-median', '.': '.••'},
        'CourierPrime-Italic': {'pt': 48, '.': '.••'},
        'CourierPrime-Regular': {'pt': 48, '.': '.••'},
        'CutiveMono-Regular': {'pt': 52, '.': '.·•'},
        'DOTMATRI': {'pt': 54},
        'DOTMBold': {'pt': 48, 'filter': 'custom-median'},
        'DottyRegular-vZOy': {'pt': 72},
        'EHSMB': {'pt': 48},
        'ELEKTRA_': {'pt': 54, '.': '.•'},
        'FiraMono-Bold': {'.': '.·•∙●', 'filter': 'custom-median'},
        'IBMPlexMono-Bold': {'.': '.·•', 'filter': 'custom-median'},
        'Merchant Copy Doublesize': {'pt': 44, 'filter': 'custom-median'},
        'Merchant Copy Wide': {'pt': 48, 'filter': 'custom-median'},
        'Merchant Copy': {'pt': 72},
        'Minecart_LCD': {'pt': 48},
        'OcrB2': {'pt': 48, 'filter': 'custom-median', '.': '.·•∙'},
        'Ordre de Départ': {'pt': 48},
        'OverpassMono-Bold': {'.': '.·•●'},
        'RobotoMono-Italic-VariableFont_wght': {'pt': 48, '.': '.·•'},
        'RobotoMono-VariableFont_wght': {'pt': 48, '.': '.·•'},
        'SourceCodePro-Black': {'.': '.·∙●', 'filter': 'custom-median'},
        'SourceCodePro-Bold': {'.': '.·∙●', 'filter': 'custom-median'},
        'SpaceMono-Bold': {'.': '.·•', 'filter': 'custom-median'},
        'SyneMono-Regular': {'pt': 48, '.': '.·•'},
        'VT323-Regular': {'pt': 54, '.': '.·•∙'},
        'XanhMono-Regular': {'pt': 48},
        'fake-receipt': {'pt': 48},
        'hydrogen': {'pt': 54},
        'scoreboard': {'pt': 48, '.': '.·•'},
    }

    # These are used for biasing the random select of characters
    weights = [25] * len(string.digits)
    weights += [5] * len(string.ascii_uppercase)
    weights += [40] * len(TINY_PUNCT)
    weights += [1] * len(OTHER_PUNCT)

    def __init__(self, length):
        """Generate a dataset."""
        self.length = length

    def __len__(self):
        """Return the length given in the constructor."""
        return self.length

    def __getitem__(self, _) -> tuple[torch.Tensor, int]:
        """Get a training image for a character and its target class."""
        char = choices(CHARS, self.weights)[0]
        font_path = choice(FONTS)

        image = self.single_char(char, font_path)

        data = TF.to_tensor(image)
        return data, CHAR_TO_CLASS[char]

    def single__getitem__(self, _) -> tuple[torch.Tensor, int]:
        """Get a training image for a single character and its target class."""
        # char = choice(CHARS)
        char = choices(CHARS, self.weights)[0]
        font_path = choice(FONTS)

        image = self.single_char(char, font_path)

        data = TF.to_tensor(image)
        return data, CHAR_TO_CLASS[char]

    def single_char(self, char, font_path, filter_='median'):
        """Draw an image of one character."""
        params = self.font_params.get(font_path.stem, {})

        tweak = 0  # self.char_tweak.get(char, 0)
        chars = params.get(char, char)
        char = chars[-1]

        size_high = params.get('pt', int(CHAR_IMAGE_SIZE.width * POINTS_TO_PIXELS))
        size_low = size_high - 2
        font_size = randint(size_low, size_high)

        font = ImageFont.truetype(str(font_path), size=font_size)
        size = font.getsize(char)
        size = ImageSize(size[0], size[1])

        image = Image.new('L', CHAR_IMAGE_SIZE, color='black')

        left = (CHAR_IMAGE_SIZE.width - size.width) // 2
        left = left if left > 0 else 0

        top = (CHAR_IMAGE_SIZE.height - size.height) // 2
        top = top if top > 0 else -tweak

        draw = ImageDraw.Draw(image)
        draw.text((left, top), char, font=font, fill='white')

        soot_fract = 0.1 if char in TINY_PUNCT else 0.25
        image = add_soot(image, soot_fract)

        filter_ = params.get('filter', filter_)
        image = filter_image(image, filter_)

        image = image.point(lambda x: 255 if x > 128 else 0)

        return image

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