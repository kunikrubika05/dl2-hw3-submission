"""Путь примера от JPG/PNG и парных преобразований до CPU-тензоров и батчей."""

from pathlib import Path
import random
from typing import Callable, Sequence

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from .files import index_image_mask_pairs
from .split import DataSplits


# Mean/std из официальной ячейки source_cell_012: нормализация ImageNet
# для предобученного VGG13. Применяется к RGB после деления значений на 255.
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
_MEAN = torch.tensor(IMAGE_MEAN, dtype=torch.float32).reshape(3, 1, 1)
_STD = torch.tensor(IMAGE_STD, dtype=torch.float32).reshape(3, 1, 1)


def image_to_tensor(
    image: Image.Image,
) -> torch.Tensor:
    """Преобразовать проверенную PIL RGB-фотографию в нормализованный тензор 3×H×W."""
    # np.array создаёт собственную float32-копию: исходная PIL-фотография не меняется.
    values = torch.from_numpy(np.array(image, dtype=np.float32)).permute(2, 0, 1)
    return values.div_(255).sub_(_MEAN).div_(_STD).contiguous()


def mask_to_tensor(
    mask: Image.Image,
) -> torch.Tensor:
    """Проверить значения PIL L-маски и преобразовать 0/255 в тензор 1×H×W, 0/1."""
    values = np.asarray(mask)
    foreground = values == 255
    if not np.all((values == 0) | foreground):
        raise ValueError("Mask must contain only 0 and 255")
    return torch.from_numpy(foreground.astype(np.float32)).unsqueeze(0)


def denormalize_image(
    image_tensor: torch.Tensor,
) -> torch.Tensor:
    """Вернуть RGB-тензор 3×H×W в диапазоне 0..1 для отображения, не меняя исходник."""
    mean = _MEAN.to(device=image_tensor.device, dtype=image_tensor.dtype)
    std = _STD.to(device=image_tensor.device, dtype=image_tensor.dtype)
    return (image_tensor * std + mean).clamp(0, 1)


PairTransform = Callable[[Image.Image, Image.Image | None], tuple[Image.Image, Image.Image | None]]


def _check_pair(
    image: Image.Image,
    mask: Image.Image | None,
    check_values: bool = False,
) -> None:
    """Проверить типы и размеры пары; при необходимости — исходные метки до transforms."""
    if not isinstance(image, Image.Image) or image.mode != "RGB":
        raise ValueError("Image must be a PIL RGB image")
    if mask is None:
        return
    if not isinstance(mask, Image.Image) or mask.mode != "L":
        raise ValueError("Mask must be a PIL L image containing 0 and 255")
    if image.size != mask.size:
        raise ValueError("Image and mask dimensions differ")
    if check_values:
        values = np.asarray(mask)
        if not np.all((values == 0) | (values == 255)):
            raise ValueError("Mask must contain only 0 and 255")


class PhotosDataset(Dataset):
    """JPG + PNG → (RGB-тензор 3×H×W, бинарная маска 1×H×W).

    transforms принимает и возвращает PIL-пару до нормализации. Если target_dir
    отсутствует, mask=None и Dataset возвращает только тензор фотографии.
    """

    def __init__(
        self,
        images_dir: str | Path,
        target_dir: str | Path | None = None,
        transforms: PairTransform | None = None,
        ids: Sequence[str] | None = None,
    ) -> None:
        """Arguments
        ---------
        images_dir : str
            Path to directory with images
            
        target_dir : str
            Path to directory with masks.
            Each mask corresponds to one image.
            Corresponding mask and image have the same name, but different format.
            
        transforms : some collection
            Sequence of transformations for images and masks. 
        """
        self.images_dir = Path(images_dir)
        self.target_dir = Path(target_dir) if target_dir is not None else None
        self._images, self._masks = index_image_mask_pairs(self.images_dir, self.target_dir)
        self.ids = tuple(sorted(self._images)) if ids is None else tuple(ids)
        if len(set(self.ids)) != len(self.ids) or not set(self.ids).issubset(self._images):
            raise ValueError("IDs must be unique and present in the dataset")
        self.transforms = transforms

    def __len__(
        self,
    ) -> int:
        return len(self.ids)

    def read_pair(
        self,
        idx: int,
    ) -> tuple[Image.Image, Image.Image | None]:
        """Decode the original PIL pair, before augmentation and normalization."""
        sample_id = self.ids[idx]
        with Image.open(self._images[sample_id]) as source:
            image = source.convert("RGB")
        mask = None
        if self._masks is not None:
            with Image.open(self._masks[sample_id]) as source:
                mask = source.copy()
        return image, mask

    def __getitem__(
        self,
        idx: int,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Arguments
        ---------
        idx : int
            Index of image and mask
        
        Returns
        -------
        (image, mask)
        """
        image, mask = self.read_pair(idx)
        _check_pair(image, mask, check_values=self.transforms is not None)
        if self.transforms is not None:
            image, mask = self.transforms(image, mask)
            if (mask is None) != (self._masks is None):
                raise ValueError("Transforms must preserve the presence of a mask")
            _check_pair(image, mask)
        image_tensor = image_to_tensor(image)
        if mask is None:
            return image_tensor
        return image_tensor, mask_to_tensor(mask)


def _seed_worker(
    worker_id: int,
) -> None:
    """Инициализировать Python/NumPy генераторы в процессе загрузчика."""
    # PyTorch задаёт seed worker; остаток приводит его к диапазону legacy NumPy seed.
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def make_loaders(
    splits: DataSplits,
    batch_size: int = 8,
    num_workers: int = 0,
    train_transforms: PairTransform | None = None,
    pin_memory: bool = False,
) -> dict[str, DataLoader]:
    """Создать train/validation/test загрузчики; перемешивание и transforms только для train."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError("num_workers must be a nonnegative integer")
    loaders = {}
    for name, folder, ids in (("train", "train", splits.train_ids),
                              ("validation", "train", splits.validation_ids),
                              ("test", "test", splits.test_ids)):
        dataset = PhotosDataset(
            images_dir=splits.root / folder,
            target_dir=splits.root / f"{folder}_mask",
            transforms=train_transforms if name == "train" else None,
            ids=ids,
        )
        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=name == "train",
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            generator=torch.Generator().manual_seed(splits.seed),
            worker_init_fn=_seed_worker,
        )
    return loaders
