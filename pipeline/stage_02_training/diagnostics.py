"""TensorBoard diagnostics and timestamped host/GPU resource measurements."""

from contextlib import contextmanager
import json
from pathlib import Path
import random
import shutil
import subprocess
import threading
import time

from ignite.engine import Engine, Events
from ignite.handlers.tensorboard_logger import TensorboardLogger, WeightsHistHandler, GradsHistHandler
import numpy as np
import psutil
import torch
from torch import nn

from ..stage_01_data.dataset import denormalize_image


@contextmanager
def _preserve_state(
    model: nn.Module,
    device: torch.device,
):
    states = [(module, module.training) for module in model.modules()]
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            model.eval()
            yield
    finally:
        for module, training in states:
            module.training = training
        random.setstate(python_state)
        np.random.set_state(numpy_state)


class ExperimentLogger:
    """Fixed, unaugmented train/test diagnostics; FP32 probe never updates weights.

    Activation histograms use at most 65,536 evenly spaced values per layer;
    images show the first four channels of the first fixed train example.
    """

    def __init__(
        self,
        output_dir: Path,
        model: nn.Module,
        config: dict,
        fixed: dict[str, tuple[torch.Tensor, torch.Tensor]],
        device: torch.device,
        resume_step: int | None = None,
    ) -> None:
        if not {"train", "test"} <= fixed.keys():
            raise ValueError("fixed requires unaugmented train and test batches")
        self.output_dir, self.model = Path(output_dir), model
        self.config, self.device = dict(config), torch.device(device)
        self.fixed = {key: tuple(value.detach().cpu().clone() for value in pair) for key, pair in fixed.items()}
        self.resume_step = resume_step
        self.last_step = resume_step or 0
        self.logger = TensorboardLogger(log_dir=str(self.output_dir / "tensorboard"),
                                        purge_step=resume_step + 1 if resume_step is not None else None)
        self.writer = self.logger.writer
        self.engine = Engine(lambda engine, batch: batch)
        self.weights = WeightsHistHandler(model)
        self.gradients = GradsHistHandler(model)

    def start(
        self,
    ) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(self.config, indent=2, ensure_ascii=False, default=str)
        (self.output_dir / "config.json").write_text(serialized, encoding="utf-8")
        hparams = {key: value if isinstance(value, (str, int, float, bool)) else json.dumps(value, default=str)
                   for key, value in self.config.items()}
        self.writer.add_text("config", serialized, self.last_step)
        if self.resume_step in (None, 0):
            self.writer.add_hparams(hparams, {}, run_name="hparams")
            with _preserve_state(self.model, self.device), torch.no_grad():
                self.writer.add_graph(self.model, self.fixed["train"][0][:1].to(self.device).float())
        self.writer.flush()

    def log_step(
        self,
        metrics: dict,
        step: int,
    ) -> None:
        self.last_step = step
        for name, value in metrics.items():
            self.writer.add_scalar(f"train/{name}", value, step)

    def _activation_hook(
        self,
        name: str,
        epoch: int,
    ):
        def capture(
            module,
            inputs,
            output,
        ):
            self._log_activation("activations", name, output, epoch)
            if output.requires_grad:
                output.register_hook(lambda grad: self._log_activation("activation_gradients", name, grad, epoch))
        return capture

    def _log_activation(
        self,
        category: str,
        name: str,
        value: torch.Tensor,
        epoch: int,
    ) -> None:
        values = value.detach().float()
        stride = max(1, (values.numel() + 65535) // 65536)
        self.writer.add_histogram(f"{category}/{name}", values.reshape(-1)[::stride].cpu(), epoch)
        if values.ndim == 4:
            channels = values[0, :4].cpu()
            low, high = channels.amin((-2, -1), keepdim=True), channels.amax((-2, -1), keepdim=True)
            channels = (channels - low) / (high - low).clamp_min(1e-8)
            self.writer.add_images(f"{category}_images/{name}", channels[:, None], epoch)

    def log_epoch(
        self,
        metrics: dict[str, dict],
        epoch: int,
        criterion: nn.Module,
    ) -> None:
        self.writer.add_scalar("epoch_number", epoch, self.last_step)
        for subset, values in metrics.items():
            for name, value in values.items():
                self.writer.add_scalar(f"epoch/{subset}/{name}", value, self.last_step)
        handles = []
        try:
            with _preserve_state(self.model, self.device), torch.enable_grad(), torch.autocast(self.device.type, enabled=False):
                self.model.zero_grad(set_to_none=True)
                for name, module in self.model.named_modules():
                    if isinstance(module, nn.Conv2d):
                        handles.append(module.register_forward_hook(self._activation_hook(name or "model", self.last_step)))
                x, y = (value.to(self.device).float() for value in self.fixed["train"])
                criterion(self.model(x), y).backward()
                for handle in handles:
                    handle.remove()
                handles.clear()
                self.engine.state.epoch = self.last_step
                self.weights(self.engine, self.logger, Events.EPOCH_COMPLETED)
                self.gradients(self.engine, self.logger, Events.EPOCH_COMPLETED)
                with torch.no_grad():
                    for subset in ("train", "test"):
                        x, y = (value.to(self.device).float() for value in self.fixed[subset])
                        probability = self.model(x).sigmoid().cpu()
                        images, targets = denormalize_image(x.cpu()), y.cpu()
                        for index in range(len(images)):
                            panel = torch.cat((images[index], targets[index].expand(3, -1, -1),
                                               probability[index].expand(3, -1, -1),
                                               (probability[index] > .5).float().expand(3, -1, -1)), dim=2)
                            self.writer.add_image(f"predictions/{subset}/{index}", panel, self.last_step)
                            from PIL import Image
                            preview_dir = self.output_dir / "previews" / f"epoch_{epoch:04d}"
                            preview_dir.mkdir(parents=True, exist_ok=True)
                            pixels = panel.clamp(0, 1).mul(255).byte().permute(1, 2, 0).numpy()
                            Image.fromarray(pixels).save(preview_dir / f"{subset}_{index}.png")
        finally:
            for handle in handles:
                handle.remove()
            self.model.zero_grad(set_to_none=True)
            self.writer.flush()

    def close(
        self,
    ) -> None:
        self.logger.close()


class ResourceMonitor:
    """Append one timestamped JSON record per sample; GPU indices remain explicit."""

    def __init__(
        self,
        output_path: Path,
        interval: float = 1,
    ) -> None:
        if not np.isfinite(interval) or interval <= 0:
            raise ValueError("interval must be positive and finite")
        self.output_path, self.interval = Path(output_path), interval
        self._stop = threading.Event()
        self._thread = None
        self._error = None
        self._nvidia = shutil.which("nvidia-smi")

    def _sample(
        self,
    ) -> dict:
        memory = psutil.virtual_memory()
        row = {"timestamp": time.time(), "cpu_percent": psutil.cpu_percent(),
               "ram_used_bytes": memory.used, "ram_available_bytes": memory.available,
               "process_rss_bytes": psutil.Process().memory_info().rss, "gpus": []}
        if self._nvidia is None:
            row["gpu_status"] = "nvidia-smi unavailable"
            return row
        fields = ("index", "utilization.gpu", "utilization.memory", "memory.used", "memory.total")
        try:
            result = subprocess.run([self._nvidia, "--query-gpu=" + ",".join(fields),
                                     "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=True)
            for line in result.stdout.strip().splitlines():
                values = line.split(",")
                if len(values) != len(fields):
                    raise ValueError(f"Unexpected nvidia-smi row: {line}")
                row["gpus"].append({key: float(value.strip()) if value.strip().replace(".", "", 1).isdigit() else value.strip()
                                    for key, value in zip(fields, values)})
            row["gpu_memory_unit"] = "MiB"
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            row["gpu_error"] = str(error)
            if isinstance(error, subprocess.CalledProcessError):
                row["gpu_error_stderr"] = error.stderr
        return row

    def _run(
        self,
    ) -> None:
        try:
            psutil.cpu_percent()
            with self.output_path.open("a", encoding="utf-8") as stream:
                while not self._stop.is_set():
                    try:
                        row = self._sample()
                    except Exception as error:
                        row = {"timestamp": time.time(), "monitor_error": repr(error)}
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                    self._stop.wait(self.interval)
        except Exception as error:
            self._error = error

    def start(
        self,
    ):
        if self._thread is not None and self._thread.is_alive():
            return self
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._stop.clear()
        self._error = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(
        self,
    ) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=6)
            if self._thread.is_alive():
                raise RuntimeError("Resource monitor did not stop")
        if self._error is not None:
            raise RuntimeError("Resource monitor could not write telemetry") from self._error

    def __enter__(
        self,
    ):
        return self.start()

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        self.stop()
