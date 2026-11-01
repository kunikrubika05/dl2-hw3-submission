"""Воспроизводимое разбиение исходного train и сохранение выбранных ID."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Sequence

from .files import index_image_mask_pairs


@dataclass(frozen=True)
class DataSplits:
    root: Path
    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    seed: int


def validate_splits(
    splits: DataSplits,
    original_train_ids: Sequence[str],
    original_test_ids: Sequence[str],
    validation_count: int | None = None,
) -> None:
    """Проверить уникальность, полноту, отсутствие утечки train/validation и сохранность test."""
    partitions = (splits.train_ids, splits.validation_ids, splits.test_ids)
    if any(not ids or len(ids) != len(set(ids)) for ids in partitions):
        raise ValueError("Each split must contain nonempty unique IDs")
    train, validation, test = map(set, partitions)
    if train & validation:
        raise ValueError("Train and validation IDs overlap")
    if train | validation != set(original_train_ids):
        raise ValueError("Train and validation must cover original train exactly")
    if test != set(original_test_ids):
        raise ValueError("Test IDs must preserve original test exactly")
    if len(splits.train_ids) + len(splits.validation_ids) != len(original_train_ids):
        raise ValueError("Train and validation cardinality differs from original train")
    if len(splits.test_ids) != len(original_test_ids):
        raise ValueError("Test cardinality differs from original test")
    if validation_count is not None and len(splits.validation_ids) != validation_count:
        raise ValueError("Validation cardinality differs from the requested split")
    # Имена могут совпадать в разных каталогах train/test: это не один и тот же файл.


def make_splits(
    data_root: str | Path,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> DataSplits:
    """Выделить validation только из train; проверить пары без декодирования изображений."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    root = Path(data_root)
    train_images, _ = index_image_mask_pairs(root / "train", root / "train_mask")
    test_images, _ = index_image_mask_pairs(root / "test", root / "test_mask")
    shuffled_ids = sorted(train_images)
    count = round(len(shuffled_ids) * validation_fraction)
    if not 0 < count < len(shuffled_ids):
        raise ValueError("Both train and validation must contain at least one image")
    random.Random(seed).shuffle(shuffled_ids)
    splits = DataSplits(
        root=root,
        train_ids=tuple(sorted(shuffled_ids[count:])),
        validation_ids=tuple(sorted(shuffled_ids[:count])),
        test_ids=tuple(sorted(test_images)),
        seed=seed,
    )
    validate_splits(splits, tuple(train_images), tuple(test_images), validation_count=count)
    return splits


def save_splits(
    splits: DataSplits,
    path: str | Path,
) -> Path:
    """Сохранить точные ID трёх выборок, seed и путь к данным в JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(splits)
    payload["root"] = str(splits.root)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
