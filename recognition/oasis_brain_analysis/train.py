"""Train and fairly evaluate OASIS 2D segmentation models."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dataset import SegmentationDataset, build_subject_split_manifest, discover_pairs, validate_subject_split_manifest
from modules import BaselineUNet, ImprovedUNet
from utils import hard_segmentation_metrics, seed_everything, write_json


def composite_segmentation_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    ce = F.cross_entropy(logits, target)
    probabilities = logits.softmax(1)
    one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    numerator = 2 * (probabilities * one_hot).sum((0, 2, 3)) + 1e-6
    denominator = probabilities.sum((0, 2, 3)) + one_hot.sum((0, 2, 3)) + 1e-6
    return ce + (1 - (numerator / denominator).mean())


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the exact number of trainable scalar parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def limit_records(records: list[dict[str, str]], maximum: int) -> list[dict[str, str]]:
    """Limit a smoke test deterministically; zero retains the complete split."""
    return records if maximum == 0 else records[:maximum]


def history_row(metrics: dict) -> dict:
    """Keep scalar epoch evidence separate from nested class metrics."""
    return {key: value for key, value in metrics.items() if key not in {"per_class", "classes_present"}}


def collect_environment(device: torch.device) -> dict[str, object]:
    """Collect machine-readable runtime details without assuming CUDA exists."""
    cuda_available = torch.cuda.is_available()
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "device": str(device),
        "cuda_available": cuda_available,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }


def load_or_create_manifest(
    data_root: str | Path, seed: int, manifest_path: str | Path | None
) -> dict[str, list[dict[str, str]]]:
    """Load a reviewed split manifest or deterministically create a new one."""
    if manifest_path is None:
        manifest = build_subject_split_manifest(discover_pairs(data_root), seed)
    else:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return validate_subject_split_manifest(manifest, data_root)


def write_learning_curves(history: list[dict], path: str | Path) -> None:
    """Write compact epoch curves for loss and validation metrics."""
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    epochs = [row["epoch"] for row in history]
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train loss")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Training loss")
    axes[0].legend()
    axes[1].plot(epochs, [row["mean_dice"] for row in history], label="Validation Dice")
    axes[1].plot(epochs, [row["mean_iou"] for row in history], label="Validation IoU")
    axes[1].set(xlabel="Epoch", ylabel="Score", title="Validation metrics")
    axes[1].legend()
    figure.savefig(path, dpi=160)
    plt.close(figure)


@torch.no_grad()
def evaluate_batches(model: nn.Module, batches, device: torch.device, num_classes: int) -> dict:
    model.eval(); logits_all = []; targets_all = []; started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
    for images, masks in batches:
        logits_all.append(model(images.to(device)).cpu()); targets_all.append(masks.cpu())
    if device.type == "cuda": torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    metrics = hard_segmentation_metrics(torch.cat(logits_all), torch.cat(targets_all), num_classes)
    metrics.update({"inference_seconds": elapsed, "peak_vram_mb": (torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0)})
    return metrics


def _loader(records, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(SegmentationDataset(records), batch_size=batch_size, shuffle=shuffle, num_workers=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train OASIS baseline or Improved U-Net")
    parser.add_argument("--data-root", required=True); parser.add_argument("--model", choices=("baseline", "improved"), required=True)
    parser.add_argument("--output-dir", default="artifacts"); parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8); parser.add_argument("--seed", type=int, default=42); parser.add_argument("--num-classes", type=int, default=4); parser.add_argument("--max-cases-per-split", type=int, default=0)
    parser.add_argument("--split-manifest", help="Reuse a validated subject-disjoint split manifest")
    parser.add_argument("--learning-curves", default="learning_curves.png", help="Filename for epoch curves inside the model output directory")
    args = parser.parse_args(); seed_everything(args.seed); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_manifest = load_or_create_manifest(args.data_root, args.seed, args.split_manifest)
    manifest = {name: limit_records(records, args.max_cases_per_split) for name, records in full_manifest.items()}
    output = Path(args.output_dir) / args.model; output.mkdir(parents=True, exist_ok=True)
    write_json(output / "split_manifest.json", full_manifest)
    write_json(output / "config.json", {**vars(args), "device": str(device)})
    write_json(output / "environment.json", collect_environment(device))
    loaders = {name: _loader(rows, args.batch_size, name == "train") for name, rows in manifest.items()}
    model = (BaselineUNet if args.model == "baseline" else ImprovedUNet)(num_classes=args.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3); best = float("-inf"); history = []; training_started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0
        for images, masks, _ in loaders["train"]:
            optimizer.zero_grad(); loss = composite_segmentation_loss(model(images.to(device)), masks.to(device), args.num_classes); loss.backward(); optimizer.step(); total += loss.item()
        validation = evaluate_batches(model, ((x, y) for x, y, _ in loaders["validation"]), device, args.num_classes)
        history.append({"epoch": epoch, "train_loss": total / max(len(loaders["train"]), 1), **validation})
        if validation["mean_dice"] > best:
            best = validation["mean_dice"]; torch.save({"model": args.model, "state_dict": model.state_dict(), "num_classes": args.num_classes}, output / "best.pt")
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=True); model.load_state_dict(checkpoint["state_dict"])
    results = evaluate_batches(model, ((x, y) for x, y, _ in loaders["test"]), device, args.num_classes); results.update({"model": args.model, "seed": args.seed, "test_cases": len(manifest["test"]), "trainable_parameters": count_trainable_parameters(model), "device": str(device), "torch_version": torch.__version__})
    write_json(output / "test_metrics.json", results)
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "mean_dice", "mean_iou", "inference_seconds", "peak_vram_mb"]); writer.writeheader(); writer.writerows(history_row(row) for row in history)
    write_learning_curves(history, output / args.learning_curves)
    write_json(output / "resource_profile.json", {
        "train_seconds": time.perf_counter() - training_started,
        "trainable_parameters": count_trainable_parameters(model),
        "test_inference_seconds": results["inference_seconds"],
        "test_cases": len(manifest["test"]),
        "seconds_per_case": results["inference_seconds"] / max(len(manifest["test"]), 1),
        "peak_vram_mb": results["peak_vram_mb"],
    })
    print(results)


if __name__ == "__main__": main()
