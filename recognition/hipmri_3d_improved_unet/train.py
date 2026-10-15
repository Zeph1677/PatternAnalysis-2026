"""Training utilities for fair HipMRI 3D baseline-versus-improved experiments."""

from __future__ import annotations

import time
from typing import Iterable

import torch
from torch.nn import functional as F


def _soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    probabilities = logits.softmax(dim=1)
    one_hot = F.one_hot(target, num_classes).movedim(-1, 1).float()
    overlap = (probabilities * one_hot).sum(dim=(0, 2, 3, 4))
    total = probabilities.sum(dim=(0, 2, 3, 4)) + one_hot.sum(dim=(0, 2, 3, 4))
    return 1 - ((2 * overlap + 1e-6) / (total + 1e-6)).mean()


def composite_3d_loss(output: torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]], target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Cross entropy plus soft Dice, with explicitly downsampled auxiliary targets."""

    final, auxiliaries = output if isinstance(output, tuple) else (output, ())
    loss = F.cross_entropy(final, target) + _soft_dice_loss(final, target, num_classes)
    for auxiliary in auxiliaries:
        scaled_target = F.interpolate(target[:, None].float(), size=auxiliary.shape[-3:], mode="nearest")[:, 0].long()
        loss = loss + 0.25 * (F.cross_entropy(auxiliary, scaled_target) + _soft_dice_loss(auxiliary, scaled_target, num_classes))
    return loss


def evaluate_3d_batches(model: torch.nn.Module, batches: Iterable[tuple[torch.Tensor, torch.Tensor]], device: torch.device, num_classes: int) -> dict[str, object]:
    """Compute hard held-out Dice/IoU and resource fields without model selection."""

    model.eval()
    prediction_parts, target_parts = [], []
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        for images, masks in batches:
            output = model(images.to(device))
            logits = output[0] if isinstance(output, tuple) else output
            prediction_parts.append(logits.argmax(1).cpu()); target_parts.append(masks.cpu())
    if device.type == "cuda": torch.cuda.synchronize(device)
    prediction, target = torch.cat(prediction_parts), torch.cat(target_parts)
    per_class, dice_values, iou_values = {}, [], []
    for label in range(num_classes):
        predicted, actual = prediction == label, target == label
        intersection = int((predicted & actual).sum())
        union = int((predicted | actual).sum())
        denominator = int(predicted.sum() + actual.sum())
        dice = (2 * intersection / denominator) if denominator else None
        iou = (intersection / union) if union else None
        per_class[str(label)] = {"dice": dice, "iou": iou, "voxels": int(actual.sum())}
        if dice is not None: dice_values.append(dice)
        if iou is not None: iou_values.append(iou)
    return {"per_class": per_class, "mean_dice": sum(dice_values) / len(dice_values), "mean_iou": sum(iou_values) / len(iou_values), "inference_seconds": time.perf_counter() - started, "peak_vram_mb": torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0}


def select_validation_checkpoint(history: list[dict[str, float]]) -> dict[str, float]:
    """Choose solely by saved validation Dice; test values cannot influence selection."""

    if not history: raise ValueError("cannot select checkpoint from empty history")
    return max(history, key=lambda row: row["validation_mean_dice"])
