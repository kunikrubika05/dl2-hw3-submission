"""Fresh-process experiment queue; rendering and resource monitoring stay in the parent."""

from collections import deque
import html
import multiprocessing as mp
from pathlib import Path
import queue
import sys
import time
import traceback


def _worker(
    experiment,
    splits,
    output_root,
    device,
    resume,
    messages,
):
    try:
        from .runner import train_experiment

        result = train_experiment(
            experiment, splits, output_root, device=device, resume=resume,
            progress_callback=lambda state: messages.put((experiment.name, "progress", state)),
        )
        messages.put((experiment.name, "result", result))
    except Exception:
        messages.put((experiment.name, "result", {
            "exp_name": experiment.name, "status": "failed", "error": traceback.format_exc(),
        }))


def _number(
    value,
):
    return "—" if value is None else f"{value:.4f}"


def _render_text(
    states,
):
    ready = sum(state["status"] in ("completed", "skipped") for state in states.values())
    lines = [f"[exps: {ready}/{len(states)} ready!]"]
    for name, state in states.items():
        eta = state.get("eta_seconds")
        eta_text = "—" if eta is None else f"{max(0, round(eta))}s"
        lines.extend([
            "", f"[{name}]:",
            f"  epochs: {state.get('epochs_done', 0)}/{state['epochs_total']}, "
            f"batches epoch {state.get('epoch', 1)}: "
            f"{state.get('batches_done', 0)}/{state.get('batches_total', '—')} "
            f"· {state.get('phase', state['status'])}",
            f"  loss {_number(state.get('loss'))} · last val IoU {_number(state.get('last_val_iou'))} "
            f"· ETA ~{eta_text}",
        ])
        if state.get("stop_reason") == "early_stopping":
            lines.append("  early stopping: validation IoU did not improve")
        if state.get("error"):
            lines.append(f"  ERROR: {state['error'].strip().splitlines()[-1]}")
    return "\n".join(lines)


class _Display:
    def __init__(
        self,
        enabled,
        show_examples=False,
    ):
        self.enabled = enabled
        self.show_examples = show_examples
        self.last_update = -float("inf")
        self.handle = None
        self.live = None
        self.panels = {}
        self.output_root = None
        self.notebook = False
        if enabled:
            try:
                from IPython import get_ipython
                self.notebook = getattr(get_ipython(), "kernel", None) is not None
            except ImportError:
                pass

    def update(
        self,
        states,
        *,
        force=False,
    ):
        if not self.enabled:
            return
        now = time.monotonic()
        tty = sys.stdout.isatty()
        interval = .25 if self.notebook or tty else 5
        if not force and now - self.last_update < interval:
            return
        self.last_update = now
        text = _render_text(states)
        if self.notebook:
            from IPython.display import HTML, display
            from .reporting import epoch_panel
            if self.output_root is not None:
                for name, state in states.items():
                    epoch = state.get("epochs_done", 0)
                    previous_epoch, handle = self.panels.get(name, (0, None))
                    if epoch > previous_epoch:
                        try:
                            content = epoch_panel(self.output_root / name, show_examples=self.show_examples)
                        except Exception as error:
                            content = f"<p>{html.escape(name)}: ошибка отчёта: {html.escape(str(error))}</p>"
                        if handle is None:
                            handle = display(HTML(content), display_id=True)
                        else:
                            handle.update(HTML(content))
                        self.panels[name] = (epoch, handle)
            content = HTML(f"<pre>{html.escape(text)}</pre>")
            if self.handle is None:
                self.handle = display(content, display_id=True)
            else:
                self.handle.update(content)
        elif tty:
            from rich.console import Console
            from rich.live import Live
            from rich.text import Text
            if self.live is None:
                self.live = Live(Text(text), console=Console(file=sys.stdout), auto_refresh=False,
                                 redirect_stdout=False, redirect_stderr=False)
                self.live.start(refresh=True)
            else:
                self.live.update(Text(text), refresh=True)
        else:
            print(text, flush=True)

    def close(
        self,
    ):
        if self.live is not None:
            self.live.stop()
            self.live = None


def run_experiments(
    experiments,
    splits,
    output_root,
    devices=("cuda:0", "cuda:1"),
    resume=False,
    display=True,
    show_examples=False,
) -> list[dict]:
    """Run at most one experiment per device; failures do not cancel other jobs.

    show_examples controls live preview images only; epoch images stay on disk.
    Return results in input order, including explicit failed/error entries. On an
    interrupt or parent failure, terminate and join only this queue's children.
    Call from an importable module/notebook; scripts need a __main__ guard.
    """
    experiments, devices = list(experiments), list(devices)
    names = [experiment.name for experiment in experiments]
    if len(set(names)) != len(names):
        raise ValueError("experiment names must be unique")
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("devices must be nonempty and unique")
    if not experiments:
        return []
    from .diagnostics import ResourceMonitor

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    messages = context.Queue()
    states = {
        experiment.name: {"status": "queued", "epochs_total": experiment.epochs}
        for experiment in experiments
    }
    results, active = {}, {}
    pending = deque(experiments)
    renderer = _Display(display, show_examples=show_examples)
    renderer.output_root = output_root

    def receive(
        message,
    ):
        name, kind, payload = message
        if kind == "progress":
            states[name].update(payload)
        else:
            result = dict(payload)
            result.setdefault("exp_name", name)
            result.setdefault("status", "completed")
            results[name] = result
            states[name].update(result)
            if result["status"] in ("completed", "skipped"):
                states[name]["epochs_done"] = result.get("epochs", states[name].get("epochs_done", 0))
                states[name].update(phase="completed", eta_seconds=0.0)

    def drain_messages(
    ):
        try:
            while True:
                receive(messages.get_nowait())
        except queue.Empty:
            pass

    try:
        with ResourceMonitor(output_root / "resources.jsonl", interval=1):
            while active or pending:
                for device in devices:
                    if device in active or not pending:
                        continue
                    experiment = pending.popleft()
                    process = context.Process(
                        target=_worker,
                        args=(experiment, splits, output_root, device, resume, messages),
                    )
                    states[experiment.name].update(status="running", device=device)
                    try:
                        process.start()
                    except Exception as error:
                        process.close()
                        receive((experiment.name, "result", {
                            "status": "failed", "error": f"Process start failed: {error}",
                        }))
                    else:
                        active[device] = (experiment.name, process)
                try:
                    receive(messages.get(timeout=.05))
                except queue.Empty:
                    pass
                drain_messages()
                for device, (name, process) in list(active.items()):
                    if process.is_alive():
                        continue
                    process.join()
                    # A clean child flushes its queue feeder before exiting.
                    drain_messages()
                    if name not in results or process.exitcode:
                        receive((name, "result", {
                            "status": "failed", "error": f"Worker exited with code {process.exitcode}"
                            if process.exitcode else "Worker exited without returning a result",
                        }))
                    process.close()
                    del active[device]
                renderer.update(states)
    finally:
        for name, process in active.values():
            if process.is_alive():
                process.terminate()
            process.join()
            process.close()
        messages.close()
        messages.join_thread()
        try:
            renderer.update(states, force=True)
        finally:
            renderer.close()
    failed = [name for name in names if results[name]["status"] == "failed"]
    if failed:
        print("Failed experiments: " + ", ".join(failed), file=sys.stderr, flush=True)
    from .reporting import ExperimentResults
    report = ExperimentResults(results[name] for name in names)
    summary = report._repr_html_()
    (output_root / "summary.html").write_text(summary, encoding="utf-8")
    if display and renderer.notebook:
        from IPython.display import HTML, display as show
        show(HTML(summary))
    return report


def _saved_setup(
    directory,
):
    """Настройки и split исходного опыта; обученные веса не загружаются."""
    import json
    from ..stage_01_data import DataSplits, PhotosDataset, prepare_data
    from ..tools import collect_backgrounds

    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    saved_split = json.loads((directory / "splits.json").read_text())
    saved_split["root"] = prepare_data("data/raw/people_data.zip", "data/prepared")
    for name in ("train_ids", "validation_ids", "test_ids"):
        saved_split[name] = tuple(saved_split[name])
    splits = DataSplits(**saved_split)
    train = PhotosDataset(splits.root / "train", splits.root / "train_mask", ids=splits.train_ids)
    backgrounds = collect_backgrounds(train, Path("artifacts/backgrounds"))
    config["augment"]["backgrounds"] = tuple(map(str, backgrounds))
    return config, splits


def run_library_augmentation(
    directory,
    epochs=30,
    device="cuda:0",
    kind="affine",
):
    """Обучить новую LinkNet с Albumentations, сохранив настройки и split базового опыта."""
    from .runner import AugmentationConfig, Experiment

    if kind not in ("affine", "color_jitter"):
        raise ValueError("kind must be affine or color_jitter")
    config, splits = _saved_setup(directory)
    config["augment"] = AugmentationConfig(**{
        **config["augment"], "affine_p": 0.75 if kind == "affine" else 0.0,
        "color_jitter_p": 0.5 if kind == "color_jitter" else 0.0,
    })
    config.update(name=f"library_{kind}", epochs=epochs)
    return run_experiments(
        [Experiment(**config)], splits, "artifacts/library_augmentation",
        devices=(device,), resume=False,
    )


def run_compile_comparison(
    directory,
    epochs=20,
    device="cuda:1",
):
    """Последовательно обучить eager/compile с одинаковыми настройками и patience=5."""
    from dataclasses import replace
    import os
    from tempfile import TemporaryDirectory
    from .runner import AugmentationConfig, Experiment

    config, splits = _saved_setup(directory)
    config["augment"] = AugmentationConfig(**config["augment"])
    config.update(name="compile_eager", epochs=epochs, patience=5, compile=False)
    base = Experiment(**config)
    experiments = [base, replace(base, name="compile_enabled", compile=True)]
    # Не используем дисковый кэш от предыдущего короткого замера.
    with TemporaryDirectory(prefix="hw3-compile-") as cache:
        overrides = {"TORCHINDUCTOR_CACHE_DIR": cache, "TRITON_CACHE_DIR": str(Path(cache) / "triton")}
        previous = {key: os.environ.get(key) for key in overrides}
        os.environ.update(overrides)
        try:
            return run_experiments(
                experiments, splits, "artifacts/compile_comparison",
                devices=(device,), resume=False,
            )
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
