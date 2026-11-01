"""PIL pair transforms. Random draws use Torch's process/worker RNG.

Seed torch.manual_seed for direct calls; DataLoader seeds each worker itself.
All transforms preserve source objects. Probabilities are explicitly configured.
"""

from collections.abc import Sequence
from math import isfinite
from numbers import Real
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps
import torch

from .dataset import PairTransform

Range = float | tuple[float, float]


def _range(
    value: Range,
    name: str,
    maximum: float = float("inf"),
    positive: bool = False,
) -> tuple[float, float]:
    bounds = (value, value) if isinstance(value, Real) else value
    if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
        raise ValueError(f"{name} must be a number or (min, max)")
    if any(isinstance(x, bool) or not isinstance(x, Real) or not isfinite(x) for x in bounds):
        raise ValueError(f"{name} must contain finite numbers")
    low, high = map(float, bounds)
    if not 0 <= low <= high <= maximum or (positive and low == 0):
        raise ValueError(f"invalid {name} bounds: {bounds}")
    return low, high


def _probability(
    p: float,
) -> float:
    if not isinstance(p, Real):
        raise ValueError("p must be a number in [0, 1]")
    return _range(p, "p", maximum=1)[0]


def _apply(
    p: float,
) -> bool:
    return p == 1 or (p > 0 and torch.rand(()).item() < p)


def _sample(
    bounds: tuple[float, float],
) -> float:
    low, high = bounds
    return low if low == high else low + (high - low) * torch.rand(()).item()


class Compose:
    """Apply configured transforms in their supplied order."""

    def __init__(
        self,
        transforms: Sequence[PairTransform],
    ) -> None:
        self.transforms = tuple(transforms)
        if any(not callable(transform) for transform in self.transforms):
            raise TypeError("every transform must be callable")

    def __call__(
        self,
        image: Image.Image,
        mask: Image.Image | None,
    ) -> tuple[Image.Image, Image.Image | None]:
        for transform in self.transforms:
            image, mask = transform(image, mask)
        return image, mask


class HorizontalFlip:
    def __init__(
        self,
        p: float,
    ) -> None:
        self.p = _probability(p)

    def __call__(
        self,
        image: Image.Image,
        mask: Image.Image | None,
    ) -> tuple[Image.Image, Image.Image | None]:
        if not _apply(self.p):
            return image, mask
        return ImageOps.mirror(image), None if mask is None else ImageOps.mirror(mask)


class Cutout:
    """One rectangular hole; height/width are fractions of image H/W.

    A number fixes the fraction, (min, max) samples it uniformly per call.
    Fractions are rounded to pixels and clamped to at least one pixel.
    keep preserves targets; erase marks the hole as background and needs a mask.
    """

    def __init__(
        self,
        *,
        height: Range,
        width: Range,
        mask_mode: str,
        p: float,
        fill: tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        self.height = _range(height, "height", maximum=1, positive=True)
        self.width = _range(width, "width", maximum=1, positive=True)
        if mask_mode not in ("keep", "erase"):
            raise ValueError("mask_mode must be 'keep' or 'erase'")
        if not isinstance(fill, (tuple, list)) or len(fill) != 3 or any(
            isinstance(x, bool) or not isinstance(x, int) or not 0 <= x <= 255 for x in fill
        ):
            raise ValueError("fill must contain three integer RGB values in [0, 255]")
        self.mask_mode, self.fill, self.p = mask_mode, tuple(fill), _probability(p)

    def __call__(
        self,
        image: Image.Image,
        mask: Image.Image | None,
    ) -> tuple[Image.Image, Image.Image | None]:
        if self.mask_mode == "erase" and mask is None:
            raise ValueError("Cutout erase requires a mask")
        if not _apply(self.p):
            return image, mask
        height = max(1, round(image.height * _sample(self.height)))
        width = max(1, round(image.width * _sample(self.width)))
        top = torch.randint(image.height - height + 1, ()).item()
        left = torch.randint(image.width - width + 1, ()).item()
        box = (left, top, left + width, top + height)
        output = image.copy()
        output.paste(self.fill, box)
        if self.mask_mode == "erase":
            mask = mask.copy()
            mask.paste(0, box)
        return output, mask


class Brightness:
    """Multiply RGB brightness by factor or a uniformly sampled factor range."""

    def __init__(
        self,
        *,
        factor: Range,
        p: float,
    ) -> None:
        self.factor, self.p = _range(factor, "factor"), _probability(p)

    def __call__(
        self,
        image: Image.Image,
        mask: Image.Image | None,
    ) -> tuple[Image.Image, Image.Image | None]:
        if not _apply(self.p):
            return image, mask
        return ImageEnhance.Brightness(image).enhance(_sample(self.factor)), mask


class ReplaceBackground:
    """Uniformly choose a supplied background; replace only zero-mask pixels.

    Paths are decoded lazily; PIL inputs are copied once at construction.
    Backgrounds are resized and center-cropped to preserve their aspect ratio.
    Requires a binary L mask (0/255), including when p=0.
    """

    def __init__(
        self,
        backgrounds: Sequence[str | Path | Image.Image],
        *,
        p: float,
    ) -> None:
        if isinstance(backgrounds, (str, Path, Image.Image)) or not backgrounds:
            raise ValueError("backgrounds must be a nonempty sequence of paths or PIL images")
        if any(not isinstance(item, (str, Path, Image.Image)) for item in backgrounds):
            raise TypeError("backgrounds must contain paths or PIL images")
        self.backgrounds = tuple(
            item.convert("RGB") if isinstance(item, Image.Image) else Path(item)
            for item in backgrounds
        )
        self.p = _probability(p)

    def __call__(
        self,
        image: Image.Image,
        mask: Image.Image | None,
    ) -> tuple[Image.Image, Image.Image]:
        if mask is None:
            raise ValueError("ReplaceBackground requires a mask")
        if mask.mode != "L" or mask.size != image.size:
            raise ValueError("mask must be L and match image dimensions")
        histogram = mask.histogram()
        if sum(histogram[1:255]):
            raise ValueError("mask must contain only 0 and 255")
        if not _apply(self.p):
            return image, mask
        source = self.backgrounds[torch.randint(len(self.backgrounds), ()).item()]
        if isinstance(source, Path):
            with Image.open(source) as loaded:
                background = loaded.convert("RGB")
        else:
            background = source
        background = ImageOps.fit(background, image.size, method=Image.Resampling.BILINEAR)
        return Image.composite(image, background, mask), mask
