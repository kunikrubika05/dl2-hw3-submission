"""Equal-weight probability ensemble compatible with the logits-based pipeline."""

import torch
from torch import nn

from .quality import predict_probabilities, _evaluate


class ProbabilityEnsemble(nn.Module):
    """Average model probabilities and return finite FP32 logits."""

    def __init__(
        self,
        models,
    ) -> None:
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        probabilities = sum(model(images).float().sigmoid() for model in self.models) / len(self.models)
        return torch.logit(probabilities, eps=torch.finfo(probabilities.dtype).eps)


def compare_ensemble(
    model,
    other_model,
    loader,
    settings,
    flip=True,
    threshold=0.5,
):
    """Compare fixed variants on an unshuffled loader; models use the same device."""

    probabilities, targets, seconds = predict_probabilities(model, loader)
    before, baseline, post_seconds = _evaluate(probabilities, targets, 0.5, settings)
    baseline_seconds = seconds + post_seconds
    ensemble = ProbabilityEnsemble([model, other_model])
    probabilities, targets, seconds = predict_probabilities(ensemble, loader, flip=flip)
    after, improved, post_seconds = _evaluate(probabilities, targets, threshold, settings)
    ensemble_seconds = seconds + post_seconds
    print(f"IoU: {baseline:.4f} → {improved:.4f} (Δ {improved - baseline:+.4f})")
    print(f"Time: {baseline_seconds:.2f} → {ensemble_seconds:.2f} s; "
          f"device={next(model.parameters()).device}, FP32, after warm-up; "
          "includes data loading, transfers, inference and postprocessing")
    return {
        "test_before": baseline, "test_after": improved,
        "test_seconds_before": baseline_seconds, "test_seconds_after": ensemble_seconds,
        "selected": {"flip": flip, "threshold": threshold}, "postprocessing": dict(settings),
        "before": before, "after": after, "targets": targets,
    }
