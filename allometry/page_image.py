"""Common functions for generating allometry images from text."""
from random import choice, randint

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


def y_image(data, font, font_size, width, height):
    """Generate a target image from the text."""
    text = data['page']

    font = ImageFont.truetype(font=font, size=font_size)
    size = font.getsize_multiline(text)

    image = Image.new(mode='L', size=(width, height), color='white')

    col, row = translate_text_params(size, width, height)

    txt = Image.new('L', size=size, color='white')
    draw = ImageDraw.Draw(txt)
    draw.text((0, 0), text, font=font, fill='black')

    image.paste(txt, (col, row))

    return image


def translate_text_params(size, width, height):
    """Translate the text."""
    d_col = width - size[0]
    d_row = height - size[1]
    col = (d_col // 2) + (randint(0, d_col // 4) * choice([1, -1]))
    row = (d_row // 2) + (randint(0, d_row // 4) * choice([1, -1]))
    return col, row


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


def x_image(image, snow_fract, image_filter):
    """Make the image look like the real data as much as possible."""
    x = np.array(image).copy()
    x = add_snow(x, snow_fract)
    x = Image.fromarray(x)

    x = filter_image(x, image_filter)

    return x
