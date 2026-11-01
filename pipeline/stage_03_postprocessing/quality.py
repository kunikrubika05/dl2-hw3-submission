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
