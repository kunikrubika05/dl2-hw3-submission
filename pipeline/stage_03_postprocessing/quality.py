"""Усреднение с отражением и подбор порога поверх фиксированной постобработки."""

from time import perf_counter

import numpy as np
import torch

from . import clean_mask, _scores


def _probabilities(
    model,
    images,
    flip,
):
    probabilities = model(images).sigmoid()
    if flip:
        reflected = model(images.flip(-1)).sigmoid().flip(-1)
        probabilities = (probabilities + reflected) / 2
    return probabilities


@torch.inference_mode()
def predict_probabilities(
    model,
    loader,
    flip=False,
):
    """FP32; время после прогрева включает чтение, переносы и предсказания."""
    model.eval()
    device = next(model.parameters()).device
    images = next(iter(loader))[0].to(device)
    _probabilities(model, images, flip).cpu()  # Прогрев и ожидание завершения GPU.
    probabilities, targets = [], []
    start = perf_counter()
    for images, target in loader:
        probabilities.append(_probabilities(model, images.to(device), flip).cpu().numpy()[:, 0])
        targets.append(target.numpy()[:, 0] > 0)
    probabilities, targets = np.concatenate(probabilities), np.concatenate(targets)
    return probabilities, targets, perf_counter() - start


def _evaluate(
    probabilities,
    targets,
    threshold,
    settings,
):
    start = perf_counter()
    masks = np.stack([clean_mask(p > threshold, **settings) for p in probabilities])
    seconds = perf_counter() - start
    return masks, _scores(masks, targets).nanmean().item(), seconds


def run_quality_comparison(
    model,
    loaders,
    settings,
    thresholds=(0.5, 0.3, 0.4, 0.6, 0.7),
):
    """Выбрать flip/порог по validation; сравнить с порогом 0.5 без flip на test."""
    trials = []
    for flip in (False, True):
        probabilities, targets, inference_seconds = predict_probabilities(model, loaders["validation"], flip)
        for threshold in thresholds:
            _, iou, post_seconds = _evaluate(probabilities, targets, threshold, settings)
            trials.append({
                "flip": flip, "threshold": threshold, "iou": iou,
                "seconds": inference_seconds + post_seconds,
            })
    best = max(trials, key=lambda row: row["iou"])
    selected = {key: best[key] for key in ("flip", "threshold")}
    for row in trials:
        print(f"flip={row['flip']}, threshold={row['threshold']:.1f}: "
              f"val IoU={row['iou']:.4f}, {row['seconds']:.2f} s")
    print(f"Selected on validation: {selected}", flush=True)

    probabilities, targets, base_seconds = predict_probabilities(model, loaders["test"])
    before, baseline, post_seconds = _evaluate(probabilities, targets, 0.5, settings)
    baseline_seconds = base_seconds + post_seconds
    if selected["flip"]:
        probabilities, targets, base_seconds = predict_probabilities(model, loaders["test"], flip=True)
    after, improved, post_seconds = _evaluate(probabilities, targets, selected["threshold"], settings)
    selected_seconds = base_seconds + post_seconds
    print(f"Test IoU: {baseline:.4f} → {improved:.4f} (Δ {improved - baseline:+.4f})")
    print(f"Test time: {baseline_seconds:.2f} → {selected_seconds:.2f} s; "
          f"device={next(model.parameters()).device}, FP32, after warm-up; "
          "includes data loading, transfers, inference and postprocessing")
    return {
        "validation": trials, "selected": selected, "postprocessing": dict(settings),
        "test_before": baseline, "test_after": improved,
        "test_seconds_before": baseline_seconds, "test_seconds_after": selected_seconds,
        "before": before, "after": after, "targets": targets,
    }
