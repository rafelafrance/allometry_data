"""Common functions for generating allometry images from text."""
from random import choice, randint

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


def clean_image(data, font, font_size, width, height):
    """Generate a clean image from the text."""
    text = data['page']

    font = ImageFont.truetype(font=font, size=font_size)
    size = font.getsize_multiline(text)

    image = Image.new(mode='L', size=(width, height), color='white')

    x, y = translate_text_params(size, width, height)

    txt = Image.new('L', size=size, color='white')
    draw = ImageDraw.Draw(txt)
    draw.text((0, 0), text, font=font, fill='black')

    image.paste(txt, (x, y))

    return image


def translate_text_params(size, width, height):
    """Translate the text."""
    dx = width - size[0]
    dy = height - size[1]
    x = (dx // 2) + (randint(0, dx // 4) * choice([1, -1]))
    y = (dy // 2) + (randint(0, dy // 4) * choice([1, -1]))
    return x, y


def add_snow(data, snow_fract, low=128, high=255):
    """Add random pixels to the image to create snow."""
    shape = data.shape
    data = data.flatten()
    how_many = int(data.size * snow_fract)
    mask = np.random.choice(data.size, how_many)
    data[mask] = np.random.randint(low, high)
    data = data.reshape(shape)
    return data


def filter_image(image, image_filter):
    """Use filters to extend the effect of the added snow."""
    image = image.filter(ImageFilter.UnsharpMask())

    if image_filter == 'max':
        image = image.filter(ImageFilter.MaxFilter())
    elif image_filter == 'custom-max':
        image = custom_filter(image)
        image = image.filter(ImageFilter.MaxFilter())
    elif image_filter == 'custom-median':
        image = custom_filter(image)
        image = image.filter(ImageFilter.MedianFilter())
    elif image_filter == 'mode':
        image = image.filter(ImageFilter.ModeFilter())

    return image


def custom_filter(image):
    """This filter seems to degrade the image in realistic way."""
    image = image.filter(ImageFilter.Kernel(
        size=(3, 3),
        kernel=(-1, 0, 1, 0, -5, 0, 1, 0, -1)
    ))
    return image


def dirty_image(image, snow_fract, image_filter):
    """Make the image look like the real data as much as possible."""
    dirty = np.array(image).copy()
    dirty = add_snow(dirty, snow_fract)
    dirty = Image.fromarray(dirty)

    dirty = filter_image(dirty, image_filter)

    return dirty
