"""Metrics and reproducibility helpers for multiclass segmentation."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Configure deterministic pseudo-random generators for an experiment."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def hard_segmentation_metrics(
    logits: torch.Tensor, target: torch.Tensor, num_classes: int
) -> dict[str, Any]:
    """Compute hard-prediction Dice and IoU, excluding absent classes."""
    prediction = logits.argmax(dim=1)
    per_class: dict[str, dict[str, float | int]] = {}
    present: list[int] = []
    dice_values: list[float] = []
    iou_values: list[float] = []
    for class_id in range(num_classes):
        predicted = prediction == class_id
        expected = target == class_id
        intersection = int((predicted & expected).sum().item())
        predicted_count = int(predicted.sum().item())
        expected_count = int(expected.sum().item())
        union = predicted_count + expected_count - intersection
        if predicted_count + expected_count == 0:
            per_class[str(class_id)] = {"dice": float("nan"), "iou": float("nan"), "support": 0}
            continue
        dice = 2.0 * intersection / (predicted_count + expected_count)
        iou = intersection / union
        per_class[str(class_id)] = {"dice": dice, "iou": iou, "support": expected_count}
        present.append(class_id)
        dice_values.append(dice)
        iou_values.append(iou)
    return {
        "per_class": per_class,
        "classes_present": present,
        "mean_dice": float(np.mean(dice_values)) if dice_values else float("nan"),
        "mean_iou": float(np.mean(iou_values)) if iou_values else float("nan"),
    }


def write_json(path: str | Path, value: Any) -> None:
    """Write structured evidence with non-finite floats represented as null."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
