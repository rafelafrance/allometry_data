"""Training and  validation datasets either generated on the fly or from files."""

import random
from pathlib import Path
from random import random, randrange, sample

import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop


class ImageFileDataset(Dataset):
    """Get data from image files stored in 'x_dir' and 'y_dir' directories."""

    def __init__(self, image_pairs, *, size=None):
        """Generate a dataset using pairs of images.

        The pairs are in tuples of (x_image, y_image).
        """
        self.images = []
        self.size = size

        # Make sure there are some pixels in the randomly cropped image
        self.threshold = (size[0] + size[1]) * 255

        for x, y in image_pairs:
            self.images.append((x, y, x.name))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        x = Image.open(image[0])
        y = Image.open(image[1])

        x, y = self._crop(x, y)
        x, y = self._rotate(x, y)

        x = self._brightness(x)
        x = self._contrast(x)
        x = self._equalize(x)

        x = TF.to_tensor(x)
        y = TF.to_tensor(y)

        return x, y, image[2]

    def _crop(self, x, y):
        i, j, h, w = 0, 0, 0, 0  # Squash linter

        for _ in range(5):
            i, j, h, w = RandomCrop.get_params(y, output_size=self.size)
            y = TF.crop(y, i, j, h, w)

            if np.array(y).flatten().sum() > self.threshold:
                break

        x = TF.crop(x, i, j, h, w)
        return x, y

    @staticmethod
    def _orient(x, y):
        if random() < 0.05:
            x = TF.rotate(x, 90)
            y = TF.rotate(y, 90)
        elif random() < 0.05:
            x = TF.rotate(x, 180)
            y = TF.rotate(y, 180)
        elif random() < 0.05:
            x = TF.rotate(x, 270)
            y = TF.rotate(y, 270)
        return x, y

    @staticmethod
    def _rotate(x, y):
        if random() < 0.1:
            theta = randrange(0, 2, 1) if random() < 0.5 else randrange(358, 360, 1)
            x = TF.rotate(x, theta)
            y = TF.rotate(y, theta)
        return x, y

    @staticmethod
    def _h_flip(x, y):
        if random() < 0.05:
            x = TF.hflip(x)
            y = TF.hflip(y)
        return x, y

    @staticmethod
    def _v_flip(x, y):
        if random() < 0.05:
            x = TF.vflip(x)
            y = TF.vflip(y)
        return x, y

    @staticmethod
    def _brightness(x):
        if random() < 0.05:
            x = TF.adjust_brightness(x, 2.0)
        return x

    @staticmethod
    def _contrast(x):
        if random() < 0.05:
            x = TF.adjust_contrast(x, 2.0)
        return x

    @staticmethod
    def _equalize(x):
        if random() < 0.05:
            x = TF.equalize(x)
        return x

    @staticmethod
    def _normalize(x):
        if random() < 0.05:
            x = TF.normalize(x, [0.5], [0.224])
        return x

    @staticmethod
    def get_files(dir_, glob='*.jpg', count=None):
        """Split contents of a dir into datasets."""
        x_dir = Path(dir_) / 'X'
        y_dir = Path(dir_) / 'Y'

        xs = {p.name: p for x in x_dir.glob(glob) if (p := Path(x))}
        ys = {p.name: p for x in y_dir.glob(glob) if (p := Path(x))}

        name_set = set(xs.keys()) & set(ys.keys())

        count = count if count else len(name_set)
        count = min(count, len(name_set))

        names = sample(list(name_set), count)

        pairs = [(xs[n], ys[n]) for n in names if n in name_set]

        return pairs
