"""Common functions for dissecting allometry sheets to create a dataset."""

from collections import namedtuple
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from allometry.const import IMAGE_SIZE

Where = namedtuple('Where', 'line type')
BBox = namedtuple('BBox', 'left top right bottom')


class AllometrySheet(Dataset):
    """A dataset for dissecting real allometry sheets."""

    padding = 1  # How much space around each character
    bin_threshold = 230  # Threshold for converting the image to binary
    row_threshold = 20  # Max number of pixels for a row to be empty
    col_threshold = 0  # Max number of pixels for a column to be empty

    def __init__(self, path: Path, rotate: int = 0):
        self.path = path  # Path to the image
        self.rotate = rotate  # Rotate the image before processing

        self.image = Image.open(path).convert('L')

        if rotate:
            self.image = self.image.rotate(rotate, expand=True, fillcolor='white')

        self.binary = self.image.point(lambda x: 255 if x < self.bin_threshold else 0)
        self.width, self.height = self.binary.size

        tops, bottoms = self.find_rows()
        self.chars: list[BBox] = self.find_chars(tops, bottoms)

    def __len__(self) -> int:
        return len(self.chars)

    def __getitem__(self, idx: int) -> torch.uint8:
        image = Image.new('L', IMAGE_SIZE, color='black')

        box = self.chars[idx]

        char = self.binary.crop(box)
        width, height = char.size

        left = (IMAGE_SIZE[0] - width) // 2
        top = (IMAGE_SIZE[1] - height) // 2

        image.paste(char, (left, top))

        data = TF.to_tensor(image)
        return data, box

    def find_rows(self) -> tuple[list[int], list[int]]:
        """Find rows in the image."""
        wheres = profile_projection(self.binary, threshold=self.row_threshold)
        tops, bottoms = self.pairs(wheres)
        return tops, bottoms

    def find_chars(self, tops: list[int], bottoms: list[int]) -> list[BBox]:
        """Find all characters in a line."""
        chars = []
        for top, bottom in zip(tops, bottoms):
            row = self.binary.crop((0, top, self.width, bottom))

            wheres = profile_projection(row, axis=0, threshold=self.col_threshold)
            lefts, rights = self.pairs(wheres)

            boxes = merge_boxes(lefts, rights, top, bottom)
            chars.extend(boxes)

        return chars

    def pairs(self, wheres):
        """Convert where items into pairs of lines (top, bottom) or (left, right)."""
        starts = [w.line - self.padding for w in wheres if w.type == 1]
        ends = [w.line + self.padding for w in wheres if w.type == 0][1:]
        return starts, ends


def profile_projection(bin_section, threshold=20, axis=1) -> list[Where]:
    """Characters in the image via a profile projection.

    Look for blank rows or columns to delimit a line of printed text or a character.
    """
    data = np.array(bin_section).copy() / 255

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


def merge_boxes(lefts, rights, top, bottom, inside=4, outside=30) -> list[BBox]:
    """Merge character boxes that are near to each other.

    We are finding boxes around each character. However, there are a lot of missing
    pixels in the characters and the profile projection method relies on finding
    blank columns to delimit the characters. This function merges nearby boxes to
    join broken characters into a single box.
    """
    boxes = [BBox(lefts[0], top, rights[0], bottom)]

    for left, right in zip(lefts[1:], rights[1:]):
        prev_left, prev_right = boxes[-1].left, boxes[-1].right

        if (left - prev_right) <= inside and (right - prev_left) <= outside:
            boxes.pop()
            boxes.append(BBox(prev_left, top, right, bottom))
        else:
            boxes.append(BBox(left, top, right, bottom))

    return boxes
