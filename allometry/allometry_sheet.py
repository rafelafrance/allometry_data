"""Common functions for dissecting allometry sheets to create a dataset."""

from collections import namedtuple
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from scipy import signal

from allometry.const import BBox, CONTEXT_SIZE, ON

Pair = namedtuple('Row', 'low high')


class AllometrySheet(Dataset):
    """A dataset for dissecting real allometry sheets.

    1) Prepare the image for dissection.
       Orient the image: This is just rotating by 0, 90, 180, or 270 degrees.
       Binarize the image: Convert the image into white text on a black background.
       Deskew the image: Fine tune the image rotation to get rows even with the edge.

    2) Find rows of text in the image.
       This gives us a top and bottom border for each row.

    3) Find all characters in each row of text.
       This gives us the left and right borders for each character.

    4) Use bounding boxes from steps 2 & 3 to get characters along with their
       "context" (surrounding characters).
    """

    def __init__(self, path: Path, rotate: int = 0, **kwargs):
        """Dissect a image of an allometry sheet and find all of its characters."""
        padding = kwargs.get('padding', 2)  # How many pixels border each character
        bin_threshold = kwargs.get('bin_threshold', 230)  # Binary threshold
        vert_dist = kwargs.get('vert_dist', 40)
        row_threshold = kwargs.get('row_threshold', 40)
        horiz_dist = kwargs.get('horiz_dist', 30)
        min_pixels = kwargs.get('min_pixels', 40)
        fat_row = kwargs.get('fat_row', 40)
        deskew_range = kwargs.get('deskew_range', (-0.2, 0.21, 0.01))

        self.image = Image.open(path).convert('L')

        # Orient the image by rotating 0, 90, 180, or 270 degrees
        if rotate != 0:
            self.image = self.image.rotate(rotate, expand=True, fillcolor='white')

        # Convert to a binary image with white text on black background
        self.binary = self.image.point(lambda x: 255 if x < bin_threshold else 0)

        self.binary = deskew_image(
            self.binary,
            padding=padding,
            fat_row=fat_row,
            vert_dist=vert_dist,
            row_threshold=row_threshold,
            deskew_range=deskew_range,
        )

        rows = find_rows(
            self.binary,
            padding=padding,
            vert_dist=vert_dist,
            row_threshold=row_threshold,
        )

        for row in rows:
            chars = find_chars(
                self.binary,
                row,
                padding=padding,
                horiz_dist=horiz_dist,
                min_pixels=min_pixels,
            )
            self.chars.extend(chars)

        self.chars = []

    def __len__(self) -> int:
        """Return the count of characters on the sheet."""
        return len(self.chars)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.IntTensor]:
        """Get a single character and its class."""
        image, box = self.char_image(idx)
        data = TF.to_tensor(image)
        return data, torch.IntTensor(box)

    def char_image(self, idx: int, *, char_width: int = 32) -> tuple[Image, BBox]:
        """Crop the character and its context and put it into an image."""
        image = Image.new('L', CONTEXT_SIZE, color='black')

        char_box = self.chars[idx]
        before = self.chars[idx - 1] if idx > 0 else None
        after = self.chars[idx + 1] if idx < len(self.chars) - 1 else None

        cropped = self.binary.crop(char_box)

        # Put the target character into the middle of the image
        left = (CONTEXT_SIZE.width - cropped.size[0]) // 2
        top = (CONTEXT_SIZE.height - cropped.size[1]) // 2

        image.paste(cropped, (left, top))

        if before and (char_box.left - before.right) < char_width:
            cropped = self.binary.crop(before)
            new_left = left - (char_box.left - before.left)
            image.paste(cropped, (new_left, top))

        if after and (after.left - char_box.right) < char_width:
            cropped = self.binary.crop(after)
            new_left = left + (after.left - char_box.left)
            image.paste(cropped, (new_left, top))

        return image, char_box


def deskew_image(
        binary_image: Image,
        *,
        deskew_range: tuple[float, float, float],
        fat_row: int = 60,
        row_threshold: int = 20,
        padding: int = 1,
        vert_dist: int = 40,
) -> Image:
    """Fine tune the rotation of a binary image."""
    rows = find_rows(
        binary_image,
        padding=padding,
        row_threshold=row_threshold, vert_dist=vert_dist,
    )
    fat_rows = sum(1 for r in rows if r.low - r.high > fat_row)

    if fat_rows == 0:
        return binary_image

    thin_rows = sum(1 for r in rows if r.low - r.high < fat_row)
    best = (thin_rows, binary_image, rows)

    for angle in np.arange(*deskew_range):
        rotated = rotate_image(binary_image, angle)
        rows = find_rows(rotated, row_threshold=row_threshold, padding=padding)
        thin_rows = sum(1 for r in rows if r.low - r.high < fat_row)
        if thin_rows > best[0]:
            best = (thin_rows, rotated, rows)

    return best[1], best[2]


def rotate_image(image: Image, angle: float) -> Image:
    """Rotate the image by a fractional degree."""
    theta = np.deg2rad(angle)
    cos, sin = np.cos(theta), np.sin(theta)
    data = (cos, sin, 0.0, -sin, cos, 0.0)
    rotated = image.transform(image.size, Image.AFFINE, data, fillcolor='black')
    return rotated


def find_rows(
        binary_image: Image,
        *,
        padding: int = 1,
        row_threshold: int = 40,
        vert_dist: int = 40,
) -> list[Pair]:
    """Find rows in the image."""
    data = np.array(binary_image) // ON

    proj = data.sum(axis=1)
    proj = proj < (binary_image.size[0] // row_threshold)
    proj = proj.astype(int) * ON

    proj[0] = 0
    proj[-1] = 0

    peaks = signal.find_peaks(proj, distance=vert_dist, plateau_size=1)

    tops = peaks[1]['right_edges']
    bots = peaks[1]['left_edges'][1:]
    pairs = [Pair(t-padding, b+padding) for t, b in zip(tops, bots)]

    return pairs


def find_chars(
        binary_image: Image,
        row: Pair,
        *,
        padding: int = 2,
        horiz_dist: int = 30,
        min_pixels: int = 40,
) -> list[BBox]:
    """Find all of the characters in a row."""
    line = binary_image.crop((0, row.low, binary_image.size[0], row.high))

    data = np.array(line) // ON

    proj = data.sum(axis=0)
    proj = proj == 0
    proj = proj.astype(int) * ON

    proj[0] = 0
    proj[-1] = 0

    peaks = signal.find_peaks(proj, distance=horiz_dist, plateau_size=1)

    lefts = peaks[1]['right_edges']
    rights = peaks[1]['left_edges'][1:]
    cols = [Pair(left-padding, right+padding) for left, right in zip(lefts, rights)]

    boxes: list[BBox] = []

    for col in cols:
        box = BBox(col.low, row.low, col.high, row.high)
        char = binary_image.crop(box)
        data = np.array(char) // ON
        pixels = np.sum(data)
        if pixels > min_pixels:
            boxes.append(box)

    return boxes
