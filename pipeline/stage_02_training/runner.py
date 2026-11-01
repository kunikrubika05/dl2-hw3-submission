"""Один эксперимент Ignite; воспроизводимое продолжение с границы эпохи."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace
from itertools import product
import hashlib
import json
import shutil
from math import isfinite
import os
from pathlib import Path
import random
import re
import subprocess
import time
from typing import Callable, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, default_collate
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver, EarlyStopping

from ..stage_01_data import DataSplits, PhotosDataset, save_splits
from ..stage_01_data.augmentations import Brightness, Compose, Cutout, HorizontalFlip, ReplaceBackground
from .diagnostics import ExperimentLogger
from .models import LinkNet, UNet
from .objectives import SegmentationLoss, overlap_scores


@dataclass(frozen=True)
class AugmentationConfig:
    flip_p: float = 0.5
    cutout_p: float = 0.25
    cutout_height: tuple[float, float] = (0.1, 0.3)
    cutout_width: tuple[float, float] = (0.1, 0.3)
    mask_mode: Literal["keep", "erase"] = "keep"
    brightness_p: float = 0.5
    brightness_factor: tuple[float, float] = (0.8, 1.2)
    background_p: float = 0.0
    backgrounds: tuple[str, ...] = ()
    affine_p: float = 0.0
    color_jitter_p: float = 0.0

    def build(
        self,
    ) -> Compose:
        transforms = [
            HorizontalFlip(self.flip_p),
            Cutout(height=self.cutout_height, width=self.cutout_width, mask_mode=self.mask_mode, p=self.cutout_p),
            Brightness(factor=self.brightness_factor, p=self.brightness_p),
        ]
        if not isfinite(self.affine_p) or not 0 <= self.affine_p <= 1:
            raise ValueError("affine_p must be in [0, 1]")
        if not isfinite(self.color_jitter_p) or not 0 <= self.color_jitter_p <= 1:
            raise ValueError("color_jitter_p must be in [0, 1]")
        if self.color_jitter_p:
            from ..stage_01_data.library_augmentations import LibraryColorJitter
            transforms[-1] = LibraryColorJitter(self.color_jitter_p)
        if self.affine_p:
            from ..stage_01_data.library_augmentations import LibraryAffine
            transforms.insert(0, LibraryAffine(self.affine_p))
        if not isfinite(self.background_p) or not 0 <= self.background_p <= 1:
            raise ValueError("background_p must be in [0, 1]")
        if self.background_p:
            transforms.append(ReplaceBackground(self.backgrounds, p=self.background_p))
        return Compose(transforms)


@dataclass(frozen=True)
class Experiment:
    name: str
    model: Literal["unet", "linknet"] = "unet"
    bce_weight: float = 1.0
    epochs: int = 20
    patience: int | None = 5
    batch_size: int = 8
    lr: float = 1e-4
    seed: int = 42
    num_blocks: int = 4
    pretrained: bool = True
    precision: Literal["amp", "fp32"] = "amp"
    num_workers: int = 2
    channels_last: bool = True
    compile: bool = False
    augment: AugmentationConfig | None = None

    def __post_init__(
        self,
    ) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.name):
            raise ValueError("name must be a safe, nonempty directory name")
        if self.model not in ("unet", "linknet") or self.precision not in ("amp", "fp32"):
            raise ValueError("Unknown model or precision")
        for name in ("epochs", "batch_size", "num_blocks"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.num_blocks > 4 or isinstance(self.num_workers, bool) or not isinstance(self.num_workers, int) or self.num_workers < 0:
            raise ValueError("num_blocks <= 4 and nonnegative integer num_workers required")
        if self.patience is not None and (isinstance(self.patience, bool) or not isinstance(self.patience, int) or self.patience < 1):
            raise ValueError("patience must be a positive integer or None")
        if not isfinite(self.lr) or self.lr <= 0:
            raise ValueError("lr must be finite and positive")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2 ** 32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        SegmentationLoss(self.bce_weight)
        if self.augment is not None:
            self.augment.build()


def experiment_grid(
    base: Experiment,
    **axes,
) -> list[Experiment]:
    """Combine scalar axes or named choices, e.g. augment={"light": config}.

    Base supplies shared settings; sorted axis names make run names independent
    of keyword order. Complex values require explicit, readable labels.
    """
    allowed = {field.name for field in fields(Experiment)} - {"name"}
    if not axes or set(axes) - allowed:
        raise ValueError("Provide nonempty axes using Experiment fields other than name")
    keys = sorted(axes)
    choices = []
    for key in keys:
        axis = axes[key]
        if isinstance(axis, (str, bytes)):
            raise ValueError(f"{key}: provide a sequence of choices, not a string")
        items = list(axis.items()) if isinstance(axis, Mapping) else [(str(value), value) for value in axis]
        if not items or any(not re.fullmatch(r"[A-Za-z0-9_.-]+", str(label)) for label, _ in items):
            raise ValueError(f"{key}: use nonempty choices with simple labels; name complex choices explicitly")
        choices.append(items)
    experiments = []
    for combination in product(*choices):
        suffix = "__".join(f"{key}-{label}" for key, (label, _) in zip(keys, combination))
        parameters = {key: value for key, (_, value) in zip(keys, combination)}
        experiments.append(replace(base, name=f"{base.name}__{suffix}", **parameters))
    if len({cfg.name for cfg in experiments}) != len(experiments):
        raise ValueError("Duplicate grid combinations or experiment names")
    return experiments


class _EpochDataset(Dataset):
    """Sample RNG depends on epoch/index, independent of workers and restart history."""

    def __init__(
        self,
        dataset: PhotosDataset,
        seed: int,
    ) -> None:
        self.dataset, self.seed = dataset, seed
        self.epoch = torch.zeros((), dtype=torch.int64).share_memory_()

    def __len__(
        self,
    ) -> int:
        return len(self.dataset)

    def __getitem__(
        self,
        index: int,
    ) -> tuple:
        seed = (self.seed + int(self.epoch) * len(self) + index) % (2 ** 63 - 1)
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed)
            image, mask = self.dataset[index]
        return image, mask, index


class _RunState:
    def __init__(
        self,
        device: torch.device,
        signature: dict,
    ) -> None:
        self.device, self.signature = device, signature
        self.history = []
        self.successful_updates = 0
        self.skipped_updates = 0

    def state_dict(
        self,
    ) -> dict:
        return {
            "signature": self.signature, "history": self.history,
            "successful_updates": self.successful_updates, "skipped_updates": self.skipped_updates,
            "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "cuda_rng": torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None,
        }

    def load_state_dict(
        self,
        saved: dict,
    ) -> None:
        if saved["signature"] != self.signature:
            raise ValueError("Checkpoint configuration/data differ from this experiment")
        self.history = saved["history"]
        self.successful_updates, self.skipped_updates = saved["successful_updates"], saved["skipped_updates"]
        torch.set_rng_state(saved["torch_rng"].cpu())
        np.random.set_state(saved["numpy_rng"])
        random.setstate(saved["python_rng"])
        if saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state(saved["cuda_rng"].cpu(), self.device)


def _write_json(
    path: Path,
    payload: dict | list,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _save_result(
    output: Path,
    history: list,
    **details,
) -> dict:
    """Use the same summary for a finished run and a completed checkpoint."""
    path = output / "result.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    result = {
        "exp_name": output.name, "status": "completed", "output_dir": str(output),
        "epochs": len(history), "history": history,
        "successful_updates": history[-1]["successful_updates"],
        "skipped_updates": history[-1]["skipped_updates"],
        "best_validation_iou": max(row["metrics"]["validation"]["iou"] for row in history),
        "run_seconds": previous.get("run_seconds"),
        "peak_allocated_mib": previous.get("peak_allocated_mib"),
        "peak_reserved_mib": previous.get("peak_reserved_mib"),
        **details,
    }
    _write_json(path, result)
    _write_json(output / "metrics.json", history)
    return result


def _restore_timings(
    history: list,
    path: Path,
) -> None:
    """Merge only measured timings from the post-checkpoint sidecar, never model state."""
    if path.exists():
        for saved, measured in zip(history, json.loads(path.read_text())):
            if saved["epoch"] == measured["epoch"] and saved["input_sha256"] == measured["input_sha256"]:
                for key in ("seconds", "checkpoint_seconds"):
                    if key in measured:
                        saved[key] = measured[key]


def _signature(
    experiment: Experiment,
    splits: DataSplits,
) -> dict:
    config = asdict(experiment)
    config.pop("epochs")  # A completed experiment may be explicitly extended.
    config.pop("num_workers")  # Epoch/sample RNG makes this operational choice independent.
    digest = hashlib.sha256()
    for subset, directory in (("train", "train"), ("validation", "train"), ("test", "test")):
        for sample_id in getattr(splits, subset + "_ids"):
            digest.update(f"{subset}/{sample_id}".encode())
            for folder, suffix in ((directory, ".jpg"), (directory + "_mask", ".png")):
                digest.update((splits.root / folder / (sample_id + suffix)).read_bytes())
    if experiment.augment:
        for path in experiment.augment.backgrounds:
            digest.update(Path(path).read_bytes())
    return json.loads(json.dumps({"config": config, "data_sha256": digest.hexdigest()}))


def _model_digest(
    model: torch.nn.Module,
) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _add_metrics(
    totals: torch.Tensor,
    loss: torch.Tensor,
    scores: dict,
    count: int,
) -> None:
    iou, dice = scores["iou"], scores["dice"]
    totals += torch.stack((loss.detach() * count, loss.new_tensor(count), iou.nansum(),
                           iou.isfinite().sum(), dice.nansum(), dice.isfinite().sum())).to(totals)


def _metrics(
    totals: torch.Tensor,
) -> dict:
    loss_sum, count, iou_sum, iou_count, dice_sum, dice_count = totals.tolist()
    return {"loss": loss_sum / count, "iou": iou_sum / iou_count if iou_count else float("nan"),
            "dice": dice_sum / dice_count if dice_count else float("nan"),
            "images": int(count), "valid_iou_images": int(iou_count)}


def train_experiment(
    experiment: Experiment,
    splits: DataSplits,
    output_root: str | Path,
    device: str = "cuda:0",
    resume: bool = False,
    progress_callback: Callable[[dict], None] | None = None,
) -> dict:
    """Train or resume at completed epoch boundary; never silently reuse another config."""
    started = time.monotonic()
    cfg = experiment
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if cfg.precision == "amp" and device.type != "cuda":
        raise ValueError("FP16 AMP requires CUDA; choose precision='fp32' for CPU")
    torch.set_num_threads(2)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.backends.cudnn.benchmark = False
    output = Path(output_root) / cfg.name
    if output.is_symlink():
        raise ValueError("Experiment output must not be a symlink")
    signature = _signature(cfg, splits)
    if not resume and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.json"
    previous_manifest = json.loads(manifest.read_text()) if manifest.exists() else None
    if previous_manifest is not None:
        if previous_manifest["signature"] != signature:
            raise ValueError("Experiment name already belongs to another configuration/data")
    elif any(output.iterdir()):
        raise FileExistsError("Experiment directory contains unrecognized files")
    saved = torch.load(output / "last.pt", map_location="cpu", weights_only=False) if resume and (output / "last.pt").exists() else None
    if saved and saved["run"]["signature"] != signature:
        raise ValueError("Checkpoint signature mismatch")
    stopped = bool(saved and cfg.patience is not None and
                   saved.get("early_stopping", {}).get("counter", 0) >= cfg.patience)
    if saved and (len(saved["run"]["history"]) >= cfg.epochs or stopped):
        history = saved["run"]["history"]
        _restore_timings(history, output / "metrics.json")
        if not (output / "best.pt").exists():
            raise FileNotFoundError("Completed experiment is missing best.pt")
        result = _save_result(output, history, skipped=True,
                              stop_reason="early_stopping" if stopped else "max_epochs")
        completed = {"exp_name": cfg.name, "status": "completed", "phase": "completed",
                     "epochs_done": len(history), "epochs_total": max(cfg.epochs, len(history)),
                     "last_val_iou": history[-1]["metrics"]["validation"]["iou"],
                     "eta_seconds": 0.0, "device": str(device), "stop_reason": result["stop_reason"]}
        _write_json(output / "status.json", completed)
        if progress_callback:
            progress_callback(completed)
        return result
    model = (UNet if cfg.model == "unet" else LinkNet)(num_blocks=cfg.num_blocks, pretrained=cfg.pretrained and saved is None)
    initial_digest = _model_digest(model) if saved is None else previous_manifest["initial_model_sha256"]
    model = model.to(device)
    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last)
    executable = torch.compile(model) if cfg.compile else model
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.precision == "amp")
    criterion = SegmentationLoss(cfg.bce_weight)
    train_base = PhotosDataset(splits.root / "train", splits.root / "train_mask", ids=splits.train_ids,
                               transforms=cfg.augment.build() if cfg.augment else None)
    train_set = _EpochDataset(train_base, cfg.seed)
    generator = torch.Generator()
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size,
                              sampler=RandomSampler(train_set, generator=generator),
                              generator=torch.Generator().manual_seed(cfg.seed),
                              num_workers=cfg.num_workers, persistent_workers=cfg.num_workers > 0,
                              pin_memory=device.type == "cuda")
    eval_sets = {
        name: PhotosDataset(splits.root / folder, splits.root / (folder + "_mask"), ids=ids)
        for name, folder, ids in (("train", "train", splits.train_ids),
                                  ("validation", "train", splits.validation_ids), ("test", "test", splits.test_ids))
    }
    eval_loaders = {name: DataLoader(ds, batch_size=cfg.batch_size, num_workers=0,
                                   pin_memory=device.type == "cuda")
                    for name, ds in eval_sets.items() if name != "train"}
    fixed = {name: default_collate([ds[i] for i in range(min(2, len(ds)))])
             for name, ds in eval_sets.items() if name in ("train", "test")}
    state = _RunState(device, signature)
    logger = ExperimentLogger(output, model, asdict(cfg), fixed, device,
                              resume_step=saved["trainer"]["iteration"] if saved else (0 if previous_manifest is not None else None))
    totals = torch.zeros(6, device=device, dtype=torch.float64)
    status = {"exp_name": cfg.name, "status": "running", "epochs_done": 0, "epochs_total": cfg.epochs,
              "epoch": 1, "batches_done": 0, "batches_total": len(train_loader), "device": str(device),
              "loss": None, "last_val_iou": None, "eta_seconds": None, "phase": "initializing"}
    last_publish = 0.0
    epoch_start = 0.0
    batch_digest = hashlib.sha256()

    def publish(
        force: bool = False,
    ) -> None:
        nonlocal last_publish
        now = time.monotonic()
        if force or now - last_publish >= 1:
            if state.history and status["status"] == "running":
                remaining = (cfg.epochs - len(state.history)) * np.mean([r["seconds"] for r in state.history])
                active_elapsed = now - epoch_start if status["epoch"] > len(state.history) else 0.0
                status["eta_seconds"] = max(0.0, remaining - active_elapsed)
            _write_json(output / "status.json", status)
            if progress_callback:
                progress_callback(dict(status))
            last_publish = now

    def transfer(
        batch: tuple,
    ) -> tuple:
        x, y = (item.to(device, non_blocking=device.type == "cuda") for item in batch[:2])
        if cfg.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        return x, y

    def update(
        engine: Engine,
        batch: tuple,
    ) -> dict:
        batch_digest.update(batch[2].numpy().tobytes())
        if (engine.state.iteration - 1) % len(train_loader) == 0:
            for tensor in batch[:2]:
                batch_digest.update(tensor.numpy().tobytes())
        x, y = transfer(batch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
            logits = executable(x)
            loss = criterion(logits, y)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        weight_norm = torch.stack([p.detach().float().square().sum() for p in model.parameters()]).sum().sqrt()
        grad_norm = torch.stack([p.grad.detach().float().square().sum() for p in model.parameters() if p.grad is not None]).sum().sqrt()
        if not scaler.is_enabled() and not torch.isfinite(grad_norm):
            raise FloatingPointError("Non-finite FP32 gradients")
        scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        skipped = int(scaler.get_scale() < scale)
        state.skipped_updates += skipped
        state.successful_updates += 1 - skipped
        scores = overlap_scores(logits.detach(), y)
        _add_metrics(totals, loss, scores, len(x))
        metrics = {"loss": loss.item(), "iou": scores["iou"].nanmean().item(),
                   "dice": scores["dice"].nanmean().item(), "weight_norm": weight_norm.item(),
                   "gradient_norm": grad_norm.item(), "lr": optimizer.param_groups[0]["lr"],
                   "amp_scale": scaler.get_scale(), "skipped": skipped}
        logger.log_step(metrics, engine.state.iteration)
        status.update(loss=metrics["loss"], batches_done=(engine.state.iteration - 1) % len(train_loader) + 1)
        publish()
        return metrics

    trainer = Engine(update)

    @trainer.on(Events.EPOCH_STARTED)
    def begin_epoch(
        engine: Engine,
    ) -> None:
        nonlocal epoch_start, batch_digest
        epoch_start = time.monotonic()
        batch_digest = hashlib.sha256()
        totals.zero_()
        train_set.epoch.fill_(engine.state.epoch)
        generator.manual_seed(cfg.seed + engine.state.epoch)
        status.update(phase="train", epoch=engine.state.epoch, epochs_done=engine.state.epoch - 1,
                      batches_done=0, batches_total=len(train_loader))
        publish(True)

    def evaluate(
        loader: DataLoader,
    ) -> dict:
        aggregate = torch.zeros(6, device=device, dtype=torch.float64)
        model.eval()
        with torch.no_grad():
            for batch_index, batch in enumerate(loader, start=1):
                x, y = transfer(batch)
                with torch.autocast(device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
                    logits = executable(x)
                    loss = criterion(logits, y)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite evaluation loss")
                _add_metrics(aggregate, loss, overlap_scores(logits, y), len(x))
                status.update(batches_done=batch_index)
                publish()
        return _metrics(aggregate)

    objects = {"model": model, "optimizer": optimizer, "scaler": scaler, "trainer": trainer, "run": state}
    disk = DiskSaver(output, require_empty=False, atomic=True)

    def save_checkpoint(
        payload: dict,
        filename: str,
        metadata: dict | None = None,
    ) -> None:
        # Callable adapter preserves the old fixed filename until DiskSaver's atomic replace.
        disk(payload, filename)

    best = Checkpoint(objects, save_checkpoint, score_function=lambda engine: status["last_val_iou"],
                      filename_pattern="best.pt", n_saved=1, include_self=True)
    early_stopping = EarlyStopping(cfg.patience, lambda engine: status["last_val_iou"], trainer) if cfg.patience is not None else None
    last_objects = {**objects, "best_handler": best}
    if early_stopping is not None:
        last_objects["early_stopping"] = early_stopping
    last = Checkpoint(last_objects, save_checkpoint, filename_pattern="last.pt", n_saved=1)

    @trainer.on(Events.EPOCH_COMPLETED)
    def finish_epoch(
        engine: Engine,
    ) -> None:
        metrics = {"train": _metrics(totals)}
        for subset, loader in eval_loaders.items():
            status.update(phase=subset, batches_done=0, batches_total=len(loader))
            publish(True)
            metrics[subset] = evaluate(loader)
        if not isfinite(metrics["validation"]["iou"]):
            raise ValueError("Validation IoU is undefined for every image")
        status.update(phase="logging", last_val_iou=metrics["validation"]["iou"])
        publish(True)
        logger.log_epoch(metrics, engine.state.epoch, criterion)
        record = {"epoch": engine.state.epoch, "seconds": time.monotonic() - epoch_start,
                  "metrics": metrics, "input_sha256": batch_digest.hexdigest(),
                  "successful_updates": state.successful_updates, "skipped_updates": state.skipped_updates}
        state.history.append(record)
        checkpoint_start = time.monotonic()
        best(engine)
        if early_stopping is not None:
            early_stopping(engine)
        last(engine)
        record["checkpoint_seconds"] = time.monotonic() - checkpoint_start
        record["seconds"] = time.monotonic() - epoch_start
        _write_json(output / "metrics.json", state.history)
        status.update(epochs_done=engine.state.epoch, batches_done=len(train_loader),
                      batches_total=len(train_loader), phase="epoch_completed")
        publish(True)

    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)
    code_version = os.environ.get("HW3_CODE_VERSION") or (git.stdout.strip() if git.returncode == 0 else "unknown")
    _write_json(manifest, {"signature": signature, "initial_model_sha256": initial_digest,
                           "code_version": code_version, "torch": torch.__version__, "device": str(device)})
    save_splits(splits, output / "splits.json")
    try:
        if saved:
            Checkpoint.load_objects(last_objects, saved)
            status.update(epochs_done=len(state.history), last_val_iou=state.history[-1]["metrics"]["validation"]["iou"])
            _restore_timings(state.history, output / "metrics.json")
            _write_json(output / "metrics.json", state.history)
        logger.start()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        trainer.run(train_loader, max_epochs=cfg.epochs)
        result = _save_result(
            output, state.history,
            stop_reason="early_stopping" if early_stopping is not None and early_stopping.counter >= cfg.patience else "max_epochs",
            run_seconds=time.monotonic() - started,
            peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2 ** 20 if device.type == "cuda" else None,
            peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 2 ** 20 if device.type == "cuda" else None,
        )
        status.update(status="completed", phase="completed", eta_seconds=0.0, stop_reason=result["stop_reason"])
        publish(True)
        return result
    except BaseException as error:
        status.update(status="failed", phase="failed", error=f"{type(error).__name__}: {error}")
        publish(True)
        raise
    finally:
        try:
            logger.close()
        finally:
            # DataLoader has no public close(); explicitly release persistent processes.
            iterator = getattr(train_loader, "_iterator", None)
            if iterator is not None:
                iterator._shutdown_workers()
                train_loader._iterator = None
