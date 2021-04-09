"""Common functions for dissecting allometry sheets to create a dataset."""

from collections import namedtuple
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from allometry.const import BBox, CONTEXT_SIZE, ImageSize

Where = namedtuple('Where', 'line type')
Row = namedtuple('Row', 'top bottom')
Col = namedtuple('Col', 'left right')


class AllometrySheet(Dataset):
    """A dataset for dissecting real allometry sheets."""

    def __init__(self, path: Path, rotate: int = 0, **kwargs):
        """Dissect a image of an allometry sheet and get all of its characters."""
        self.path = path  # Path to the image
        self.rotate = rotate  # Rotate the image before processing

        self.padding = kwargs.get('padding', 1)  # How many pixels border each character
        self.bin_threshold = kwargs.get('bin_threshold', 230)  # Binary threshold
        self.row_threshold = kwargs.get('row_threshold', 20)  # Max pixels for empty row
        self.col_threshold = kwargs.get('col_threshold', 0)  # Max pixels for empty col
        self.box_width = kwargs.get('box_width', 8)  # A box must be this wide
        self.fat_row = kwargs.get('fat_row', 60)  # How many pixels for a fat row
        self.deskew_range = kwargs.get('deskew_range', (-0.2, 0.21, 0.01))  # Angles
        self.split_radius = 5  # Split fat rows at lowest profile within this span
        self.char_size = kwargs.get('char_size', ImageSize(32, 48))

        self.image = Image.open(path).convert('L')
        if rotate:
            self.image = self.image.rotate(rotate, expand=True, fillcolor='white')

        self.binary = self.image.point(lambda x: 255 if x < self.bin_threshold else 0)
        self.binary, rows = self.deskew_image(self.binary)
        self.rows = self.split_fat_rows(rows)
        self.page_size = ImageSize(self.binary.size[0], self.binary.size[1])

        self.chars: list[BBox] = self.find_chars(self.rows)

    def __len__(self) -> int:
        """Return the count of characters on the sheet."""
        return len(self.chars)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, tuple]:
        """Get a single character and its class."""
        image, box = self.char_image(idx)
        data = TF.to_tensor(image)
        return data, box

    def char_image(self, idx):
        """Crop the character and its context and put it into an image."""
        image = Image.new('L', CONTEXT_SIZE, color='black')

        box = self.chars[idx]
        before = self.chars[idx - 1] if idx > 0 else None
        after = self.chars[idx + 1] if idx < len(self.chars) - 1 else None

        cropped = self.binary.crop(box)

        # Put the target character into the middle of the image
        left = (CONTEXT_SIZE.width - cropped.size[0]) // 2
        top = (CONTEXT_SIZE.height - cropped.size[1]) // 2

        image.paste(cropped, (left, top))

        if before and (box.left - before.right) < self.char_size.width:
            cropped = self.binary.crop(before)
            new_left = left - (box.left - before.left)
            image.paste(cropped, (new_left, top))

        if after and (after.left - box.right) < self.char_size.width:
            cropped = self.binary.crop(after)
            new_left = left + (after.left - box.left)
            image.paste(cropped, (new_left, top))

        return image, box

    def split_fat_rows(self, rows):
        """Split fat rows at the point where there are the fewest "on" pixels."""
        new_rows = []
        proj = None
        for row in rows:
            count = round((row.bottom - row.top) / self.fat_row)
            if count < 2:
                new_rows.append(row)
            elif count == 2:
                if proj is None:
                    data = np.array(self.binary).copy() / 255
                    proj = data.sum(axis=1)
                mid = row.top + ((row.bottom - row.top) // 2)
                up = mid - self.split_radius
                down = mid + self.split_radius + 1
                split = np.argmin(proj[up:down])
                div = mid - self.split_radius + split
                new_rows.append(Row(row.top, div))
                new_rows.append(Row(div, row.bottom))
            else:
                raise ValueError

        return new_rows

    def deskew_image(self, binary_image):
        """Deskew a binary image."""
        rows = self.find_rows(binary_image)
        fat_rows = sum(1 for r in rows if r.bottom - r.top > self.fat_row)

        if fat_rows == 0:
            return binary_image, rows

        thin_rows = sum(1 for r in rows if r.bottom - r.top < self.fat_row)
        best = (thin_rows, binary_image, rows)

        for angle in np.arange(*self.deskew_range):
            rotated = rotate_image(binary_image, angle)
            rows = self.find_rows(rotated)
            thin_rows = sum(1 for r in rows if r.bottom - r.top < self.fat_row)
            if thin_rows > best[0]:
                best = (thin_rows, rotated, rows)

        return best[1], best[2]

    def find_rows(self, image) -> list[Row]:
        """Find rows in the image."""
        wheres = chop_image(image, threshold=self.row_threshold)
        tops, bottoms = self.pairs(wheres)
        rows = [Row(t, b) for t, b in zip(tops, bottoms)]
        return rows

    def find_chars(self, rows: list[Row]) -> list[BBox]:
        """Find all characters in a line."""
        chars = []
        for row in rows:
            image = self.binary.crop((0, row.top, self.page_size.width, row.bottom))

            wheres = chop_image(image, axis=0, threshold=self.col_threshold)
            lefts, rights = self.pairs(wheres)

            boxes = merge_boxes(lefts, rights, row)
            boxes = [b for b in boxes if b.right - b.left >= self.box_width]
            chars.extend(boxes)

        return chars

    def pairs(self, wheres):
        """Convert where items into pairs of lines (top, bottom) or (left, right)."""
        starts = [w.line - self.padding for w in wheres if w.type == 1]
        ends = [w.line + self.padding for w in wheres if w.type == 0][1:]
        return starts, ends


def rotate_image(image, angle):
    """Rotate the image by a fractional degree."""
    theta = np.deg2rad(angle)
    cos, sin = np.cos(theta), np.sin(theta)
    data = (cos, sin, 0.0, -sin, cos, 0.0)
    rotated = image.transform(image.size, Image.AFFINE, data, fillcolor='black')
    return rotated


def chop_image(bin_image, axis=1, threshold=20) -> list[Where]:
    """Cop the image into rows or columns."""
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
    where = [Where(w, s[0]) for w, s in zip(where, splits)]

    return where


def merge_boxes(lefts, rights, row, inside=4, outside=40) -> list[BBox]:
    """Merge character boxes that are near to each other.

    We are finding boxes around each character. However, there are a lot of missing
    pixels in the characters and the profile projection method relies on finding
    blank columns to delimit the characters. This function merges nearby boxes to
    join broken characters into a single box.
    """
    boxes = [BBox(lefts[0], row.top, rights[0], row.bottom)]

    for left, right in zip(lefts[1:], rights[1:]):
        prev_left, prev_right = boxes[-1].left, boxes[-1].right

        if (left - prev_right) <= inside and (right - prev_left) <= outside:
            boxes.pop()
            boxes.append(BBox(prev_left, row.top, right, row.bottom))
        else:
            boxes.append(BBox(left, row.top, right, row.bottom))

    return boxes
