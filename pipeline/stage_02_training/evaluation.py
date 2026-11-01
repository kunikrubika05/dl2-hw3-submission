"""Загрузка экспортированного эксперимента и оценка без обучения."""

import json
from pathlib import Path

import torch

from ..stage_01_data import DataSplits, make_loaders, prepare_data
from .models import LinkNet, UNet
from .objectives import overlap_scores


def load_saved_experiment(
    directory: str | Path,
    device: str = "cuda",
    data_archive: str | Path = "data/raw/people_data.zip",
    data_directory: str | Path = "data/prepared",
) -> tuple:
    """Загрузить state_dict и загрузчики с исходными ID, без аугментаций."""
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    saved_split = json.loads((directory / "splits.json").read_text())
    data_root = prepare_data(data_archive, data_directory)
    splits = DataSplits(
        root=data_root,
        train_ids=tuple(saved_split["train_ids"]),
        validation_ids=tuple(saved_split["validation_ids"]),
        test_ids=tuple(saved_split["test_ids"]),
        seed=saved_split["seed"],
    )
    model = {"unet": UNet, "linknet": LinkNet}[config["model"]](
        num_blocks=config["num_blocks"], pretrained=False,
    )
    model.load_state_dict(torch.load(
        directory / "best_weights.pt", map_location="cpu", weights_only=True,
    ))
    loaders = make_loaders(splits, batch_size=config["batch_size"])
    return model.to(device).eval(), loaders


@torch.inference_mode()
def evaluate_iou(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
) -> float:
    """Средний IoU по изображениям в FP32; пустые пары исключаются, как в обучении."""
    model.eval()
    device = next(model.parameters()).device
    scores = []
    for images, masks in loader:
        logits = model(images.to(device))
        scores.append(overlap_scores(logits, masks.to(device))["iou"].cpu())
    return torch.cat(scores).nanmean().item()


@torch.inference_mode()
def show_predictions(
    model: torch.nn.Module,
    dataset,
    indices: tuple[int, ...] = (0, 1),
):
    """Показать фиксированные примеры: фото, разметку и маску при p > 0.5."""
    import matplotlib.pyplot as plt
    from ..stage_01_data import denormalize_image
    from ..tools import PLOT_STYLE

    images, masks = zip(*(dataset[index] for index in indices))
    model.eval()
    device = next(model.parameters()).device
    logits = model(torch.stack(images).to(device)).cpu()
    scores = overlap_scores(logits, torch.stack(masks))["iou"]
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(len(indices), 3, squeeze=False, layout="constrained")
        for row, (image, mask, prediction) in enumerate(zip(images, masks, logits > 0)):
            axes[row, 0].imshow(denormalize_image(image).permute(1, 2, 0).numpy())
            axes[row, 1].imshow(mask[0], cmap="gray", vmin=0, vmax=1)
            axes[row, 2].imshow(prediction[0], cmap="gray", vmin=0, vmax=1)
            axes[row, 2].set_xlabel(f"IoU = {scores[row]:.4f}")
            for ax in axes[row]:
                ax.set_xticks([])
                ax.set_yticks([])
        for ax, title in zip(axes[0], ("Фотография", "Разметка", "Предсказание")):
            ax.set_title(title)
    return fig
