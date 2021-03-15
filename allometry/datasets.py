"""Training and  validation datasets either generated on the fly or from files."""

from pathlib import Path

import numpy as np
from PIL import Image
from random import sample

from torch.utils.data import Dataset


class ImageFileDataset(Dataset):
    """Get a dataset from image files stored in 'clean' and 'dirty' directories."""

    def __init__(self, image_pairs, resize=None):
        """Generate a dataset using pairs of images.

        The pairs are in tuples of (dirty_image, clean_image).
        """
        self.resize = resize
        self.images = []
        for dirty, clean in image_pairs:
            self.images.append({
                'dirty': dirty,  # X
                'clean': clean,  # Y
                'name': dirty.name,
                'shape': Image.open(dirty).size,
            })

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        dirty = Image.open(image['dirty']).convert('F')
        clean = Image.open(image['clean']).convert('F')

        if self.resize:
            dirty = dirty.resize(self.resize)
            clean = clean.resize(self.resize)

        dirty = np.array(dirty)
        dirty /= 255.0
        dirty = dirty[np.newaxis, :, :]

        clean = np.array(clean)
        clean /= 255.0
        clean = clean[np.newaxis, :, :]

        return dirty, clean, image['shape'], image['name']

    @staticmethod
    def split_files(dirty_dir, clean_dir, *fractions, glob='*.jpg', count=None):
        """Split contents of a dir into datasets."""
        clean = {p.name: p for x in Path(clean_dir).glob(glob) if (p := Path(x))}
        dirty = {p.name: p for x in Path(dirty_dir).glob(glob) if (p := Path(x))}

        names = set(dirty.keys()) & set(clean.keys())
        count = count if count else len(names)
        names = sample(names, count)

        splits = []
        start, end = 0, 0
        for fract in fractions:
            end = start + round(count * fract)
            splits.append([(dirty[n], clean[n]) for n in names[start:end]])
            start = end

        if end < count:
            splits[0].extend(names[end:count])

        return tuple(splits)
