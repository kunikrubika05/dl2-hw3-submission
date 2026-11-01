"""Notebook reports from completed epoch files; no model inference in the UI."""

import base64
import html
import io
import json
from pathlib import Path

from ..tools import PLOT_STYLE


def _image(
    data,
):
    encoded = base64.b64encode(data).decode("ascii")
    return f'<img style="max-width:100%;height:auto" src="data:image/png;base64,{encoded}">'


def epoch_panel(
    directory,
    show_examples=False,
):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    directory = Path(directory)
    path = directory / "metrics.json"
    if not path.exists():
        return ""
    history = json.loads(path.read_text())
    if not history:
        return ""
    epochs = [row["epoch"] for row in history]
    parts = [f"<h3>{html.escape(directory.name)} · эпоха {history[-1]['epoch']}</h3>"]
    with plt.rc_context(PLOT_STYLE):
        for metric, title in (("loss", "Loss"), ("iou", "IoU")):
            fig, ax = plt.subplots(layout="constrained")
            try:
                for subset in ("train", "validation", "test"):
                    ax.plot(epochs,
                            [r["metrics"][subset][metric] for r in history], label=subset,
                            marker="o" if len(history) == 1 else None)
                ax.set(xlabel="Эпоха", ylabel=title, title=f"{directory.name}: {title}")
                ax.set_xlim(.5, max(epochs) + .5)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.legend()
                ax.grid(alpha=.25)
                buffer = io.BytesIO()
                fig.savefig(buffer, format="png", dpi=100)
                parts.append(_image(buffer.getvalue()))
            finally:
                plt.close(fig)
    if show_examples:
        parts.append("<p>Фиксированные примеры: фото → истинная маска → вероятность → бинарная маска</p>")
        for subset in ("train", "test"):
            for path in sorted((directory / "previews" / f"epoch_{history[-1]['epoch']:04d}").glob(f"{subset}_*.png")):
                parts.extend([f"<p>{html.escape(path.stem)}</p>", _image(path.read_bytes())])
    return "".join(parts)


class ExperimentResults(list):
    """Keep programmatic access to history without dumping it in notebooks."""

    def __repr__(
        self,
    ):
        return "\n".join(f"{r['exp_name']}: {r['status']}" for r in self)

    def _repr_html_(
        self,
    ):
        rows = []
        for result in self:
            history = result.get("history", [])
            best = max(history, key=lambda r: r["metrics"]["validation"]["iou"]) if history else None
            epoch_seconds = sum(row["seconds"] for row in history)
            values = [result["exp_name"], result["status"],
                      str(len(history)), result.get("stop_reason", "—"),
                      str(best["epoch"]) if best else "—",
                      f"{best['metrics']['validation']['iou']:.4f}" if best else "—",
                      f"{result['run_seconds']:.1f}" if result.get("run_seconds") is not None else "—",
                      f"{epoch_seconds:.1f}" if history else "—",
                      f"{epoch_seconds / len(history):.1f}" if history else "—"]
            rows.append("<tr>" + "".join(f"<td>{html.escape(v)}</td>" for v in values) + "</tr>")
        headers = ("Exp", "Статус", "Эпох выполнено", "Причина остановки", "Лучшая эпоха по val IoU", "Validation IoU", "Весь запуск, с", "Сумма эпох, с", "Средняя эпоха, с")
        return ("<table><thead><tr>" + "".join(f"<th>{h}</th>" for h in headers)
                + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
                + "<p>Время эпох включает обучение, оценку, логирование и checkpoints; без первоначальной подготовки.</p>")
