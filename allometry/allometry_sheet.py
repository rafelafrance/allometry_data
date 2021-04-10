"""Common functions for dissecting allometry sheets to create a dataset."""

from collections import namedtuple
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from allometry.const import BBox, CONTEXT_SIZE

ThresholdBreak = namedtuple('ThresholdBreak', 'line type')
Row = namedtuple('Row', 'top bottom')
Col = namedtuple('Col', 'left right')


class AllometrySheet(Dataset):
    """A dataset for dissecting real allometry sheets.

    1) Prepare the image for dissection.
       Orient the image: This is just rotating by 0, 90, 180, 0r 270 degrees.
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
        padding = kwargs.get('padding', 1)  # How many pixels border each character
        fat_row = kwargs.get('fat_row', 60)  # Row with this many pixels are too fat

        self.image = Image.open(path).convert('L')

        # Orient the image by rotating 0, 90, 180, or 270 degrees
        if rotate != 0:
            self.image = self.image.rotate(rotate, expand=True, fillcolor='white')

        # Convert to a binary image with white text on black background
        bin_threshold = kwargs.get('bin_threshold', 230)  # Binary threshold
        self.binary = self.image.point(lambda x: 255 if x < bin_threshold else 0)

        # Fine tune the image rotation
        self.binary, rows = deskew_image(
            self.binary,
            deskew_range=kwargs.get('deskew_range', (-0.2, 0.21, 0.01)),
            fat_row=fat_row,
            padding=padding,
            row_threshold=kwargs.get('row_threshold', 20),
        )

        # Some rows will have multiple lines of text, split them at the thinnest point
        rows = split_fat_rows(
            self.binary,
            rows,
            fat_row=fat_row,
            split_radius=kwargs.get('split_radius', 5))

        # Now split rows of text into characters
        self.chars: list[BBox] = find_chars(
            self.binary,
            rows,
            padding=padding,
            box_width=kwargs.get('box_width', 8),
            col_threshold=kwargs.get('col_threshold', 0))

    def __len__(self) -> int:
        """Return the count of characters on the sheet."""
        return len(self.chars)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, BBox]:
        """Get a single character and its class."""
        image, box = self.char_image(idx)
        data = TF.to_tensor(image)
        return data, box

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
) -> tuple[Image, list[Row]]:
    """Deskew a binary image."""
    rows = find_rows(binary_image, row_threshold=row_threshold, padding=padding)
    fat_rows = sum(1 for r in rows if r.bottom - r.top > fat_row)

    if fat_rows == 0:
        return binary_image, rows

    thin_rows = sum(1 for r in rows if r.bottom - r.top < fat_row)
    best = (thin_rows, binary_image, rows)

    for angle in np.arange(*deskew_range):
        rotated = rotate_image(binary_image, angle)
        rows = find_rows(rotated, row_threshold=row_threshold, padding=padding)
        thin_rows = sum(1 for r in rows if r.bottom - r.top < fat_row)
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
        row_threshold: int = 20,
        padding: int = 1,
) -> list[Row]:
    """Find rows in the image."""
    threshold_breaks = chop_image(binary_image, threshold=row_threshold)
    tops, bottoms = line_pairs(threshold_breaks, padding)
    rows = [Row(t, b) for t, b in zip(tops, bottoms)]
    return rows


def split_fat_rows(
        binary_image: Image,
        rows: list[Row],
        *,
        fat_row: int = 60,
        split_radius: int = 5,
) -> list[Row]:
    """Split fat rows at the point where there are the fewest "on" pixels."""
    new_rows = []
    proj = None

    for row in rows:
        count = round((row.bottom - row.top) / fat_row)

        if count < 2:
            new_rows.append(row)

        else:
            if proj is None:
                data = np.array(binary_image).copy() / 255
                proj = data.sum(axis=1)

            if count == 2:
                mid = row.top + ((row.bottom - row.top) // 2)
                north = mid - split_radius
                south = mid + split_radius + 1
                split = np.argmin(proj[north:south])
                div = mid - split_radius + split
                new_rows.append(Row(row.top, div))
                new_rows.append(Row(div, row.bottom))

            else:
                raise ValueError

    return new_rows


def find_chars(
        binary_image: Image,
        rows: list[Row],
        *,
        padding: int = 1,
        box_width: int = 8,
        col_threshold: int = 0
) -> list[BBox]:
    """Find all characters in a line of text."""
    chars = []
    for row in rows:
        binary_row = binary_image.crop((0, row.top, binary_image.size[0], row.bottom))

        threshold_breaks = chop_image(binary_row, axis=0, threshold=col_threshold)

        lefts, rights = line_pairs(threshold_breaks, padding)
        cols = [Col(ll, rr) for ll, rr in zip(lefts, rights)]

        cols = split_fat_chars(binary_row, cols)

        boxes = merge_boxes(cols, row)
        boxes = [b for b in boxes if b.right - b.left >= box_width]
        chars.extend(boxes)

    return chars


def split_fat_chars(
        binary_row: Image,
        cols: list[Col],
        *,
        fat_col: int = 35,
        split_radius: int = 5,
) -> list[Col]:
    """Some characters blur into each other, we split these a the thinnest point."""
    new_cols = []
    proj = None

    for col in cols:
        count = round((col.right - col.left) / fat_col)

        if count < 2:
            new_cols.append(col)

        else:
            if proj is None:
                data = np.array(binary_row).copy() / 255
                proj = data.sum(axis=0)

            if count == 2:
                mid = col.left + ((col.right - col.left) // 2)
                east = mid - split_radius
                west = mid + split_radius
                split = np.argmin(proj[east:west])
                div = mid - split + split
                new_cols.append(Col(col.left, div))
                new_cols.append(Col(div, col.right))

            # else:
            #     raise ValueError

    return new_cols


def merge_boxes(
        cols: list[Col],
        row: Row,
        *,
        inside: int = 4,
        outside: int = 40,
) -> list[BBox]:
    """Merge character boxes that are near to each other.

    We are finding boxes around each character. However, there are a lot of missing
    pixels in the characters and the profile projection method relies on finding
    blank columns to delimit the characters. This function merges nearby boxes to
    join broken characters into a single box.
    """
    boxes: list[BBox] = [BBox(cols[0].left, row.top, cols[0].right, row.bottom)]

    for col in cols[1:]:
        prev_left, prev_right = boxes[-1].left, boxes[-1].right

        if (col.left - prev_right) <= inside and (col.right - prev_left) <= outside:
            boxes.pop()
            boxes.append(BBox(prev_left, row.top, col.right, row.bottom))
        else:
            boxes.append(BBox(col.left, row.top, col.right, row.bottom))

    return boxes


def chop_image(
        bin_image: Image,
        axis: int = 1,
        threshold: int = 20,
) -> list[ThresholdBreak]:
    """Chop the image into rows or columns."""
    data = np.array(bin_image).copy() / 255
    proj = data.sum(axis=axis)
    proj = proj > threshold
    proj = proj.astype(int)

    prev = np.insert(proj[:-1], 0, 0)
    curr = np.insert(proj[1:], 0, 0)
    where = np.where(curr != prev)[0]
    where = where.tolist()

    splits = np.array_split(proj, where)

    where = where if where[0] == 0 else ([0] + where)
    where = [ThresholdBreak(w, s[0]) for w, s in zip(where, splits)]

    return where


def line_pairs(
        threshold_breaks: list[ThresholdBreak],
        padding: int = 1,
) -> tuple[list[int], list[int]]:
    """Convert to pairs of lines (top, bottom) or (left, right)."""
    starts = [w.line - padding for w in threshold_breaks if w.type == 1]
    ends = [w.line + padding for w in threshold_breaks if w.type == 0][1:]
    return starts, ends
