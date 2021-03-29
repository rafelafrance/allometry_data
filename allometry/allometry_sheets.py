"""Run a set if real images through a trained model."""

import math
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import DefaultDict

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

ImagePart = namedtuple('ImagePart', 'part name box')
Size = namedtuple('Size', 'width height')
Sizes = namedtuple('Sizes', 'width height padded_width padded_height')


class AllometrySheets(Dataset):
    """A dataset for handling real images."""

    def __init__(self, input_dir, output_dir, crop_size, rotate=0, glob='*.tif'):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.crop = Size(*crop_size)
        self.rotate = rotate
        self.image_sizes: dict[str, Sizes] = {}

        self.cleaned_parts: DefaultDict[str, list[ImagePart]] = defaultdict(list)
        self.image_parts: list[ImagePart] = []
        for path in self.input_dir.glob(glob):
            self.image_parts.extend(self.chop_image(path))

    def __len__(self):
        return len(self.image_parts)

    def __getitem__(self, idx):
        part = self.image_parts[idx]
        image = TF.to_tensor(part.part)
        return image, part.name, part.box

    def chop_image(self, path):
        """Chop the image into chunks small enough for the model."""
        image = Image.open(path)
        if self.rotate:
            image = image.rotate(self.rotate, fillcolor='white', expand=True)

        width, height = image.size
        padded_width = math.ceil(width / self.crop.width) * self.crop.width
        padded_height = math.ceil(height / self.crop.height) * self.crop.height
        self.image_sizes[path.name] = Sizes(width, height, padded_width, padded_height)

        padded = Image.new('L', (padded_width, padded_height), color='white')
        padded.paste(image, image.getbbox())

        image_parts = []
        for left in range(0, padded.size[0], self.crop.width):
            for top in range(0, padded.size[1], self.crop.height):
                box = [left, top, left + self.crop.width, top + self.crop.height]
                cropped_image = padded.crop(box)
                image_part = ImagePart(cropped_image, path.name, box)
                image_parts.append(image_part)

        return image_parts

    def save_predictions(self, predictions, names, boxes):
        """Save model predictions for later stitching."""
        boxes = [b.tolist() for b in boxes]
        boxes = zip(*boxes)
        for pred, name, box in zip(predictions, names, boxes):
            image = TF.to_pil_image(pred)
            self.cleaned_parts[name].append(ImagePart(image, name, box))

    def stitch_images(self):
        """Put the images back together."""
        for name, parts in self.cleaned_parts.items():
            width, height, padded_width, padded_height = self.image_sizes[name]
            clean = Image.new('L', (padded_width, padded_height), color='white')
            for part in parts:
                clean.paste(part.part, part.box)
            clean = clean.crop((0, 0, width, height))
            path = self.output_dir / name
            clean.save(path, 'TIFF')
