"""Бинарная сегментация: BCE/Dice и метрики по изображениям, а не по батчам."""

from math import isfinite
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

Reduction = Literal["mean", "sum"] | None


def _validate_reduction(
    reduction: Reduction,
) -> None:
    if reduction not in (None, "mean", "sum"):
        raise ValueError("reduction must be None, 'mean' or 'sum'")


def _check_shapes(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    if predictions.ndim < 2 or predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have identical [B, ...] shapes")


class DiceLoss(nn.Module):
    def __init__(
        self,
        eps: float = 1e-7,
        reduction: Reduction = None,
        with_logits: bool = True,
    ) -> None:
        """Arguments
        ---------
        eps : float
            eps in denominator
        reduction : Optional[str] (None, 'mean' or 'sum')
            specifies the reduction to apply to the output:
            
            None: no reduction will be applied
            'mean': the sum of the output will be divided by the number of elements in the batch
            'sum':  the output will be summed. 
        with_logits : bool
            If True, use additional sigmoid for inputs
        """
        super().__init__()
        _validate_reduction(reduction)
        if not isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        self.eps, self.reduction, self.with_logits = eps, reduction, with_logits

    def forward(
        self,
        logits: torch.Tensor,
        true_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Arguments
        ---------
        logits: torch.Tensor
            Unnormalized probability of true class. Shape: [B, ...]
        true_labels: torch.Tensor
            Mask of correct predictions. Shape: [B, ...]
        Returns
        -------
        torch.Tensor
            If reduction is 'mean' or 'sum' returns a tensor with a single element
            Otherwise, returns a tensor of shape [B]
        """
        _check_shapes(logits, true_labels)
        # Float32 reductions avoid FP16 overflow for 320×240 masks.
        probabilities = logits.float().sigmoid() if self.with_logits else logits.float()
        targets = true_labels.float()
        numerator = 2 * (probabilities * targets).flatten(1).sum(1)
        # Official formula adds eps to EACH pixel before summing.
        denominator = (probabilities + targets + self.eps).flatten(1).sum(1)
        losses = 1 - numerator / denominator
        if self.reduction == "mean":
            return losses.mean()
        if self.reduction == "sum":
            return losses.sum()
        return losses


class SegmentationLoss(nn.Module):
    """w×BCE + (1−w)×Dice; endpoints skip the unused loss computation."""

    def __init__(
        self,
        bce_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if not isfinite(bce_weight) or not 0 <= bce_weight <= 1:
            raise ValueError("bce_weight must be finite and in [0, 1]")
        self.bce_weight = bce_weight
        self.dice = DiceLoss(reduction="mean")

    def forward(
        self,
        logits: torch.Tensor,
        true_labels: torch.Tensor,
    ) -> torch.Tensor:
        _check_shapes(logits, true_labels)
        if self.bce_weight == 0:
            return self.dice(logits, true_labels)
        bce = F.binary_cross_entropy_with_logits(logits.float(), true_labels.float())
        if self.bce_weight == 1:
            return bce
        return self.bce_weight * bce + (1 - self.bce_weight) * self.dice(logits, true_labels)


@torch.no_grad()
def overlap_scores(
    logits: torch.Tensor,
    true_labels: torch.Tensor,
    threshold: float = 0.0,
) -> dict[str, torch.Tensor]:
    """IoU/Dice per image; empty prediction AND target yield NaN for explicit skipping."""
    _check_shapes(logits, true_labels)
    predictions, targets = logits > threshold, true_labels > 0
    intersection = (predictions & targets).flatten(1).sum(1).float()
    union = (predictions | targets).flatten(1).sum(1).float()
    return {"iou": intersection / union, "dice": 2 * intersection / (union + intersection)}


class IoUScore(nn.Module):
    def __init__(
        self,
        threshold: float = 0.0,
        reduction: Reduction = None,
    ) -> None:
        """Arguments
        ---------
        threshold : float
            threshold for logits binarization
        reduction : Optional[str] (None, 'mean' or 'sum')
            specifies the reduction to apply to the output:
            
            None: no reduction will be applied
            'mean': the sum of the output will be divided by the number of elements in the batch
            'sum':  the output will be summed.
        
        The original stub mentions with_logits, but its signature has no such parameter.
        This implementation thresholds logits directly.
        """
        super().__init__()
        _validate_reduction(reduction)
        if not isfinite(threshold):
            raise ValueError("threshold must be finite; it applies to logits")
        self.threshold, self.reduction = threshold, reduction

    @torch.no_grad()
    def forward(
        self,
        logits: torch.Tensor,
        true_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Arguments
        ---------
        logits: torch.Tensor
            Unnormalized probability of true class. Shape: [B, ...]
        true_labels: torch.Tensor
            Mask of correct predictions. Shape: [B, ...]
        Returns
        -------
        torch.Tensor
            If reduction is 'mean' or 'sum' returns a tensor with a single element
            Otherwise, returns a tensor of shape [B]
        """
        scores = overlap_scores(logits, true_labels, self.threshold)["iou"]
        if self.reduction == "mean":
            return scores.nanmean()
        if self.reduction == "sum":
            return scores.nansum()
        return scores
