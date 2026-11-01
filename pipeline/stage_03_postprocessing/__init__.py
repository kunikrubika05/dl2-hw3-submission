"""Постобработка: выбор на validation, оценка и примеры на test."""

import numpy as np
import torch
from scipy.ndimage import binary_fill_holes, label

from ..stage_02_training.objectives import overlap_scores


def clean_mask(
    mask: np.ndarray,
    largest: bool = False,
    fill_holes: bool = False,
) -> np.ndarray:
    """Оставить крупнейшую 4-связную область, затем заполнить внутренние дыры."""
    if largest and mask.any():
        labels, _ = label(mask)
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        mask = labels == sizes.argmax()
    return binary_fill_holes(mask) if fill_holes else mask


@torch.inference_mode()
def _predict(
    model,
    loader,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    device = next(model.parameters()).device
    masks, targets = [], []
    for images, target in loader:
        masks.append((model(images.to(device)) > 0).cpu().numpy()[:, 0])
        targets.append(target.numpy()[:, 0] > 0)
    return np.concatenate(masks), np.concatenate(targets)


def _scores(
    masks,
    targets,
) -> torch.Tensor:
    return overlap_scores(torch.from_numpy(masks), torch.from_numpy(targets))["iou"]


def run_postprocessing(
    model,
    loaders,
) -> dict:
    """Сравнить 4 варианта на validation; оценить выбранный на test без подбора."""
    masks, targets = _predict(model, loaders["validation"])
    trials = []
    for largest, fill_holes in ((False, False), (True, False), (False, True), (True, True)):
        processed = np.stack([clean_mask(mask, largest, fill_holes) for mask in masks])
        trials.append({
            "largest": largest, "fill_holes": fill_holes,
            "iou": _scores(processed, targets).nanmean().item(),
        })
    best = max(trials, key=lambda row: row["iou"])
    settings = {key: best[key] for key in ("largest", "fill_holes")}
    before, targets = _predict(model, loaders["test"])
    after = np.stack([clean_mask(mask, **settings) for mask in before])
    test_before = _scores(before, targets).nanmean().item()
    test_after = _scores(after, targets).nanmean().item()
    for row in trials:
        print(f"largest={row['largest']}, fill_holes={row['fill_holes']}: val IoU={row['iou']:.6f}")
    print(f"Selected: {settings}")
    print(f"Test IoU: {test_before:.6f} → {test_after:.6f} (Δ {test_after - test_before:+.6f})")
    return {
        "validation": trials, "settings": settings,
        "test_before": test_before, "test_after": test_after,
        "before": before, "after": after, "targets": targets,
    }


def show_postprocessing(
    result,
    dataset,
):
    """Показать первые два изменившихся test-примера, без отбора по улучшению IoU."""
    import matplotlib.pyplot as plt
    from ..stage_01_data import denormalize_image
    from ..tools import PLOT_STYLE

    before, after, targets = (result[key] for key in ("before", "after", "targets"))
    indices = np.flatnonzero((before != after).any(axis=(1, 2)))[:2]
    if not len(indices):
        print("Постобработка не изменила test-маски.")
        return
    scores_before = _scores(before[indices], targets[indices])
    scores_after = _scores(after[indices], targets[indices])
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(len(indices), 4, squeeze=False, layout="constrained")
        for row, index in enumerate(indices):
            image, _ = dataset[int(index)]
            axes[row, 0].imshow(denormalize_image(image).permute(1, 2, 0).numpy())
            for ax, mask in zip(axes[row, 1:], (targets[index], before[index], after[index])):
                ax.imshow(mask, cmap="gray", vmin=0, vmax=1)
            axes[row, 2].set_xlabel(f"IoU {scores_before[row]:.4f}")
            axes[row, 3].set_xlabel(f"IoU {scores_after[row]:.4f}")
            for ax in axes[row]:
                ax.set_xticks([])
                ax.set_yticks([])
        for ax, title in zip(axes[0], ("Фото", "Разметка", "До", "После")):
            ax.set_title(title)
    return fig
