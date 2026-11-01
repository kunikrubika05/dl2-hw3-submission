"""Аугментации Albumentations для пары PIL image/mask."""

import albumentations as A
import numpy as np
from PIL import Image
import torch


class LibraryAffine:
    def __init__(
        self,
        p: float,
    ) -> None:
        self.transform = A.Affine(
            scale=(0.85, 1.15), translate_percent=(-0.1, 0.1),
            rotate=(-15, 15), p=p,
        )

    def __call__(
        self,
        image: Image.Image,
        mask: Image.Image,
    ) -> tuple[Image.Image, Image.Image]:
        return _apply(self.transform, image, mask)


class LibraryColorJitter:
    def __init__(
        self,
        p: float,
    ) -> None:
        self.transform = A.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03, p=p,
        )

    def __call__(
        self,
        image: Image.Image,
        mask: Image.Image,
    ) -> tuple[Image.Image, Image.Image]:
        return _apply(self.transform, image, mask)


def _apply(
    transform,
    image: Image.Image,
    mask: Image.Image,
) -> tuple[Image.Image, Image.Image]:
    # Сохраняем RNG остальных аугментаций и используем seed текущего примера.
    with torch.random.fork_rng(devices=[]):
        seed = torch.randint(2 ** 31, ()).item()
    transform.set_random_seed(seed)
    transformed = transform(image=np.asarray(image), mask=np.asarray(mask))
    return Image.fromarray(transformed["image"]), Image.fromarray(transformed["mask"])
