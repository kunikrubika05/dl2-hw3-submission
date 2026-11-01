"""Общие помощники: запуск среды и показ результатов."""

from pathlib import Path
import subprocess
import sys


def enable_notebook_updates(
) -> None:
    """Enable standard autoreload, backporting its import on old hosted images.

    Python 3.12 removed imp; IPython 7.34 still imports reload from it.
    Apply the upstream importlib equivalent without upgrading IPython.
    """
    from importlib import util
    try:
        from IPython import get_ipython
    except ImportError:
        return

    shell = get_ipython()
    if shell is None:
        return
    path = Path(util.find_spec("IPython.extensions.autoreload").origin)
    source = path.read_text(encoding="utf-8")
    if "from imp import reload" in source:
        path.write_text(source.replace("from imp import reload", "from importlib import reload"), encoding="utf-8")
    if shell.find_line_magic("autoreload") is None:
        shell.run_line_magic("load_ext", "IPython.extensions.autoreload")
    # Setup may run after git pull: refresh already imported project modules now.
    # Only our package is tracked; hosted notebook dependencies are left alone.
    reloader = shell.find_line_magic("autoreload").__self__._reloader
    mode = "1 --full" if hasattr(reloader, "deduper_reloader") else "1"
    shell.run_line_magic("autoreload", mode)
    for name in tuple(sys.modules):
        if name == "pipeline" or name.startswith("pipeline."):
            reloader.mark_module_reloadable(name)
            reloader.modules_mtimes[name] = 0
    reloader.check()


def setup_runtime(
    root: Path,
) -> None:
    """Update the project while retaining the notebook namespace."""
    root = root.resolve()
    if not (3, 10) <= sys.version_info[:2] < (3, 14):
        raise RuntimeError("Use Python 3.10–3.13 for this project.")
    enable_notebook_updates()
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "-e", str(root)], check=True)
    version = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    import os
    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    print("Code version:", version)


PLOT_STYLE = {
    "figure.figsize": (11, 7),
    "font.family": "DejaVu Serif",
    "lines.linewidth": 2,
    "lines.markersize": 8,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "axes.titlesize": 24,
    "axes.labelsize": 16,
}


def show_idx_image(
    dataset,
    idx: int = 0,
):
    """Показать фотографию и маску выбранного примера после преобразований."""
    import matplotlib.pyplot as plt
    from .stage_01_data.dataset import denormalize_image

    image, mask = dataset[idx]
    rgb = denormalize_image(image).permute(1, 2, 0).cpu().numpy()
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(1, 2, layout="constrained")
        axes[0].imshow(rgb)
        axes[0].set_title("Фотография")
        axes[1].imshow(mask[0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Маска: человек = 1")
        for ax in axes:
            ax.set_axis_off()
    return fig


def split_summary(
    splits,
) -> dict[str, int]:
    """Вернуть количество примеров в каждой выборке."""
    return {
        "train": len(splits.train_ids),
        "validation": len(splits.validation_ids),
        "test": len(splits.test_ids),
    }


def batch_summary(
    batch,
) -> dict:
    """Показать размерности, типы и метки уже полученного батча без чтения загрузчика."""
    images, masks = batch
    return {
        "images": {"shape": list(images.shape), "dtype": str(images.dtype)},
        "masks": {
            "shape": list(masks.shape),
            "dtype": str(masks.dtype),
            "values": masks.unique().tolist(),
        },
    }


def collect_backgrounds(
    dataset,
    output_dir: Path,
    count: int = 3,
    size: tuple[int, int] = (96, 96),
) -> list[Path]:
    """Сохранить чистые фоновые участки только из переданного train Dataset; size=(W,H)."""
    import numpy as np
    from PIL import Image

    if dataset.target_dir is None or dataset.transforms is not None:
        raise ValueError("Use an untransformed training dataset with masks")
    width, height = size
    if count < 1 or min(width, height) < 2:
        raise ValueError("Positive count and patch dimensions >= 2 are required")
    output_dir.mkdir(parents=True, exist_ok=True)
    backgrounds = []
    for sample_id in dataset.ids:
        with Image.open(dataset.target_dir / f"{sample_id}.png") as source:
            mask = np.asarray(source)
        for top in range(0, mask.shape[0] - height + 1, height // 2):
            for left in range(0, mask.shape[1] - width + 1, width // 2):
                if np.any(mask[top:top + height, left:left + width]):
                    continue
                path = output_dir / f"{sample_id}_x{left}_y{top}_{width}x{height}.png"
                with Image.open(dataset.images_dir / f"{sample_id}.jpg") as image:
                    image.crop((left, top, left + width, top + height)).save(path)
                backgrounds.append(path)
                break
            else:
                continue
            break
        if len(backgrounds) == count:
            return backgrounds
    raise ValueError("Not enough clean background patches; reduce size or count")


def show_augmentations(
    dataset,
    transforms: dict,
    index: int = 0,
    seed: int = 42,
) -> None:
    """Показать каждое преобразование отдельно на одной паре; не менять RNG обучения."""
    import matplotlib.pyplot as plt
    import torch

    if dataset.transforms is not None:
        raise ValueError("Use an untransformed dataset for isolated demonstrations")
    image, mask = dataset.read_pair(index)
    for name, transform in transforms.items():
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed)
            changed_image, changed_mask = transform(image.copy(), mask.copy())
        with plt.rc_context(PLOT_STYLE):
            fig, axes = plt.subplots(2, 2, layout="constrained")
            fig.suptitle(name, fontsize=20)
            for row, (photo, target) in enumerate(((image, mask), (changed_image, changed_mask))):
                axes[row][0].imshow(photo)
                axes[row][1].imshow(target, cmap="gray", vmin=0, vmax=255)
                axes[row][0].set_title("До" if row == 0 else "После")
                axes[row][1].set_title("Маска")
                axes[row][0].set_axis_off()
                axes[row][1].set_axis_off()
            plt.show()
            plt.close(fig)


if __name__ == "__main__":
    setup_runtime(Path(__file__).resolve().parents[1])
