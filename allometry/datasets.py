"""Training and  validation datasets either generated on the fly or from files."""

import random
from pathlib import Path
from random import random, randrange, sample

import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop


class ImageFileDataset(Dataset):
    """Get data from image files stored in 'clean' and 'dirty' directories."""

    def __init__(self, image_pairs, *, size=None, seed=None):
        """Generate a dataset using pairs of images.

        The pairs are in tuples of (dirty_image, clean_image).
        """
        self.images = []
        self.size = size
        self.seed = seed

        for dirty, clean in image_pairs:
            self.images.append((dirty, clean, dirty.name))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        dirty = Image.open(image[0])
        # dirty = np.asarray(dirty).copy()
        # dirty /= 255.0
        # dirty = Image.fromarray(dirty)

        clean = Image.open(image[1])
        # clean = np.asarray(clean).copy()
        # clean /= 255.0
        # clean = Image.fromarray(clean)

        dirty, clean = self._crop(dirty, clean)
        # dirty, clean = self._rotate(dirty, clean)
        # dirty, clean = self._h_flip(dirty, clean)
        # dirty, clean = self._v_flip(clean, dirty)

        # dirty = self._brightness(dirty)
        # dirty = self._contrast(dirty)
        # dirty = self._equalize(dirty)

        dirty = TF.to_tensor(dirty)
        clean = TF.to_tensor(clean)

        return dirty, clean, image[2]

    def _crop(self, dirty, clean):
        i, j, h, w = RandomCrop.get_params(dirty, output_size=self.size)
        dirty = TF.crop(dirty, i, j, h, w)
        clean = TF.crop(clean, i, j, h, w)
        return clean, dirty

    @staticmethod
    def _equalize(dirty):
        if random() < 0.05:
            dirty = TF.equalize(dirty)
        return dirty

    @staticmethod
    def _contrast(dirty):
        if random() < 0.05:
            dirty = TF.adjust_contrast(dirty, 2.0)
        return dirty

    @staticmethod
    def _brightness(dirty):
        if random() < 0.05:
            dirty = TF.adjust_brightness(dirty, 2.0)
        return dirty

    @staticmethod
    def _v_flip(dirty, clean):
        if random() > 0.05:
            dirty = TF.vflip(dirty)
            clean = TF.vflip(clean)
        return dirty, clean

    @staticmethod
    def _h_flip(dirty, clean):
        if random() > 0.05:
            dirty = TF.hflip(dirty)
            clean = TF.hflip(clean)
        return clean, dirty

    @staticmethod
    def _rotate(dirty, clean):
        if random() > 0.1:
            theta = randrange(0, 2, 1) if random() < 0.5 else randrange(358, 360, 1)
            dirty = TF.rotate(dirty, theta)
            clean = TF.rotate(clean, theta)
        return clean, dirty

    @staticmethod
    def split_files(dirty_dir, clean_dir, *segments, glob='*.jpg', count=None):
        """Split contents of a dir into datasets."""
        clean = {p.name: p for x in Path(clean_dir).glob(glob) if (p := Path(x))}
        dirty = {p.name: p for x in Path(dirty_dir).glob(glob) if (p := Path(x))}

        names = set(dirty.keys()) & set(clean.keys())
        count = count if count else len(names)
        names = sample(names, count)

        splits = [list([]) for _ in segments]

        start, end = 0, 0
        for s, seg in enumerate(segments):
            end = int(start + seg if seg > 1.0 else start + round(count * seg))
            splits[s] = [(dirty[n], clean[n]) for n in names[start:end]]
            start = end

        if end < count:
            splits[0].extend(names[end:count])

        return tuple(splits)
