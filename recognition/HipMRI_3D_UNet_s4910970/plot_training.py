"""Render auditable training curves from saved experiment histories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


History = list[dict[str, object]]


def load_history(path: Path | str) -> History:
    """Load a JSON history list or a training summary containing ``history``."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    history = payload.get("history") if isinstance(payload, dict) else payload
    if not isinstance(history, list) or not history:
        raise ValueError(f"{source} does not contain a non-empty training history")
    if not all(isinstance(row, dict) for row in history):
        raise ValueError(f"{source} history rows must be JSON objects")
    return history


def _prostate_key(row: dict[str, object]) -> tuple[float, float, float, int]:
    required = (
        "validation_patient_prostate_dice",
        "validation_worst_patient_prostate_dice",
        "validation_prostate_false_positive_voxels",
        "validation_prostate_false_negative_voxels",
        "epoch",
    )
    if any(key not in row for key in required):
        raise ValueError("prostate_dice selection requires complete prostate validation metrics")
    return (
        float(row["validation_patient_prostate_dice"]),
        float(row["validation_worst_patient_prostate_dice"]),
        -float(row["validation_prostate_false_positive_voxels"]) - float(row["validation_prostate_false_negative_voxels"]),
        -int(float(row["epoch"])),
    )


def selected_epoch(history: History, selection_metric: str) -> int:
    """Return the validation-only selected epoch using the training selection rule."""

    if not history:
        raise ValueError("cannot select an epoch from an empty history")
    if selection_metric == "mean_dice":
        if any("validation_mean_dice" not in row or "epoch" not in row for row in history):
            raise ValueError("mean_dice selection requires validation_mean_dice and epoch")
        return int(float(max(history, key=lambda row: float(row["validation_mean_dice"]))["epoch"]))
    if selection_metric == "prostate_dice":
        return int(float(max(history, key=_prostate_key)["epoch"]))
    raise ValueError(f"unsupported selection metric: {selection_metric}")


def _selection_field(selection_metric: str) -> tuple[str, str]:
    if selection_metric == "mean_dice":
        return "validation_mean_dice", "Validation mean Dice"
    if selection_metric == "prostate_dice":
        return "validation_patient_prostate_dice", "Validation patient prostate Dice"
    raise ValueError(f"unsupported selection metric: {selection_metric}")


def render_training_curves(histories: Mapping[str, History], output_path: Path | str, selection_metric: str = "mean_dice") -> None:
    """Plot train loss and the declared validation selection metric for saved histories."""

    if not histories:
        raise ValueError("at least one labelled history is required")
    metric_field, metric_title = _selection_field(selection_metric)
    figure, (loss_axis, metric_axis) = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for label, history in histories.items():
        if not str(label).strip():
            raise ValueError("history labels must be non-empty")
        if any("epoch" not in row or "train_loss" not in row or metric_field not in row for row in history):
            raise ValueError(f"{label} history lacks fields required for {selection_metric} plotting")
        epochs = [float(row["epoch"]) for row in history]
        losses = [float(row["train_loss"]) for row in history]
        metric_values = [float(row[metric_field]) for row in history]
        epoch = selected_epoch(history, selection_metric)
        selected_index = epochs.index(float(epoch))
        line = loss_axis.plot(epochs, losses, marker="o", linewidth=1.6, label=label)[0]
        metric_axis.plot(epochs, metric_values, marker="o", linewidth=1.6, label=label, color=line.get_color())
        metric_axis.scatter([epoch], [metric_values[selected_index]], color=line.get_color(), s=70, zorder=3)
        metric_axis.annotate(f"selected e{epoch}", (epoch, metric_values[selected_index]), textcoords="offset points", xytext=(4, 7), fontsize=8, color=line.get_color())
    loss_axis.set(title="Training loss", xlabel="Epoch", ylabel="Composite loss")
    metric_axis.set(title=metric_title, xlabel="Epoch", ylabel="Dice")
    metric_axis.margins(x=0.08, y=0.12)
    for axis in (loss_axis, metric_axis):
        axis.grid(alpha=0.25)
        axis.legend()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def render_learning_rate_curve(histories: Mapping[str, History], output_path: Path | str) -> None:
    """Render only recorded rates; historical missing rates are never invented."""

    for label, history in histories.items():
        if not history or any("learning_rate" not in row or "epoch" not in row for row in history):
            raise ValueError(f"{label} lacks recorded learning_rate")
    figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for label, history in histories.items():
        axis.plot([row["epoch"] for row in history], [row["learning_rate"] for row in history], marker="o", label=label)
    axis.set(xlabel="Epoch", ylabel="Learning rate used", title="Recorded optimizer learning rate")
    axis.grid(alpha=0.25)
    axis.legend()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", action="append", required=True, type=Path, help="JSON history or training summary; repeat for each series")
    parser.add_argument("--label", action="append", required=True, help="Series label; repeat in the same order as --history")
    parser.add_argument("--selection-metric", choices=("mean_dice", "prostate_dice"), default="mean_dice")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--lr-output", type=Path, help="Optional plot of recorded rates; missing rates cause an error")
    return parser


def main() -> None:
    args = create_argument_parser().parse_args()
    if len(args.history) != len(args.label):
        raise ValueError("--history and --label must be supplied equally often")
    histories = {label: load_history(path) for label, path in zip(args.label, args.history, strict=True)}
    render_training_curves(histories, args.output, args.selection_metric)
    if args.lr_output is not None:
        render_learning_rate_curve(histories, args.lr_output)


if __name__ == "__main__":
    main()
