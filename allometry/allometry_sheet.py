"""Common functions for dissecting allometry sheets to create a dataset."""

from collections import namedtuple
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from allometry.const import BBox, CONTEXT_SIZE, OFF, ON

Pair = namedtuple('Pair', 'low high')
Where = namedtuple('Where', 'line type')

PADDING = 2  # How many pixels to pad a character
BIN_THRESHOLD = 230  # A pixel is considered "on" if its values is at least this
ROW_THRESHOLD = 40  # Max number of "on" pixels for a row to be considered empty
COL_THRESHOLD = 0  # Max number of "on" pixels for a column to be considered empty
INSIDE = 4  # Only merge boxes if they are this close
OUTSIDE = 40  # Only merge boxes if they will not make a box this fat
DESKEW_RANGE = (-0.2, 0.21, 0.01)  # Rotations to check when deskewing
FAT_ROW = 60  # Minimize this when deskewing
MIN_PIXELS = 40  # A character box must have this many "on" pixels


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

    def __init__(self, path: Path, rotate: int = 0):
        """Dissect a image of an allometry sheet and find all of its characters."""
        self.image = Image.open(path).convert('L')

        # Orient the image by rotating 0, 90, 180, or 270 degrees
        if rotate != 0:
            self.image = self.image.rotate(rotate, expand=True, fillcolor='white')

        # Convert to a binary image with white text on black background
        self.binary = self.image.point(lambda x: ON if x < BIN_THRESHOLD else OFF)
        self.binary = deskew_image(self.binary)

        rows = find_rows(self.binary)
        self.chars: list[BBox] = find_chars(self.binary, rows)

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
        deskew_range: tuple[float, float, float] = DESKEW_RANGE,
        fat_row: int = FAT_ROW,
) -> Image:
    """Fine tune the rotation of a binary image."""
    rows = find_rows(binary_image)
    fat_rows = sum(1 for r in rows if r.low - r.high > fat_row)

    if fat_rows == 0:
        return binary_image

    thin_rows = sum(1 for r in rows if r.low - r.high < fat_row)
    best = (thin_rows, binary_image)

    for angle in np.arange(*deskew_range):
        rotated = rotate_image(binary_image, angle)
        rows = find_rows(rotated)
        thin_rows = sum(1 for r in rows if r.low - r.high < fat_row)
        if thin_rows > best[0]:
            best = (thin_rows, rotated)

    return best[1]


def rotate_image(image: Image, angle: float) -> Image:
    """Rotate the image by a fractional degree."""
    theta = np.deg2rad(angle)
    cos, sin = np.cos(theta), np.sin(theta)
    data = (cos, sin, 0.0, -sin, cos, 0.0)
    rotated = image.output(image.size, Image.AFFINE, data, fillcolor='black')
    return rotated


def find_rows(
        binary_image: Image,
        *,
        row_threshold: int = ROW_THRESHOLD,
) -> list[Pair]:
    """Find rows of text in the image."""
    return profile_projection(binary_image, threshold=row_threshold)


def find_chars(
        binary_image: Image,
        rows: list[Pair],
        *,
        col_threshold: int = COL_THRESHOLD,
) -> list[BBox]:
    """Find all characters in a row of text."""
    boxes = []
    width = binary_image.size[0]
    for row in rows:
        top, bottom = row
        row_image = binary_image.crop((0, top, width, bottom))

        pairs = profile_projection(row_image, axis=0, threshold=col_threshold)

        row_boxes = merge_boxes(pairs, row)
        row_boxes = remove_empty_boxes(binary_image, row_boxes)
        boxes.extend(row_boxes)

    return boxes


def remove_empty_boxes(
        binary_image: Image,
        row_boxes: list[BBox],
        *,
        min_pixels: int = MIN_PIXELS,
) -> list[BBox]:
    """Remove boxes with too few "on" pixels."""
    boxes: list[BBox] = []
    for box in row_boxes:
        char = binary_image.crop(box)
        data = np.array(char) // ON
        pixels = np.sum(data)
        if pixels > min_pixels:
            boxes.append(box)

    return boxes


def merge_boxes(
        pairs: list[Pair],
        row: Pair,
        *,
        inside: int = INSIDE,
        outside: int = OUTSIDE,
) -> list[BBox]:
    """Merge character boxes that are near to each other.

    We are finding boxes around each character. However, there are a lot of missing
    pixels in the characters and the profile projection method relies on finding
    blank columns to delimit the characters. This function merges nearby boxes to
    join broken characters into a single box.
    """
    top, bottom = row
    row_boxes = [BBox(pairs[0].low, top, pairs[0].high, bottom)]

    for left, right in pairs[1:]:
        prev_left, prev_right = row_boxes[-1].left, row_boxes[-1].right

        if (left - prev_right) <= inside and (right - prev_left) <= outside:
            row_boxes.pop()
            row_boxes.append(BBox(prev_left, top, right, bottom))
        else:
            row_boxes.append(BBox(left, top, right, bottom))

    return row_boxes


def profile_projection(
        bin_section: Image,
        threshold: int = 20,
        axis: int = 1,
        padding: int = PADDING,
) -> list[Pair]:
    """Characters in the image via a profile projection.
    Look for blank rows or columns to delimit a line of printed text or a character.
    """
    data = np.array(bin_section).copy() / 255

    proj = data.sum(axis=axis)
    proj = proj > threshold
    proj = proj.astype(int)

    prev = np.insert(proj[:-1], 0, 0)
    curr = np.insert(proj[1:], 0, 0)
    wheres = np.where(curr != prev)[0]
    wheres = wheres.tolist()

    splits = np.array_split(proj, wheres)

    wheres = wheres if wheres[0] == 0 else ([0] + wheres)
    wheres = [Where(w, s[0]) for w, s in zip(wheres, splits)]

    starts = [w.line - padding for w in wheres if w.type == 1]
    ends = [w.line + padding for w in wheres if w.type == 0][1:]
    pairs = [Pair(t - padding, b + padding) for t, b in zip(starts, ends)]

    return pairs
