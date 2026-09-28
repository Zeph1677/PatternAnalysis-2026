"""Train and fairly evaluate OASIS 2D segmentation models."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dataset import SegmentationDataset, build_subject_split_manifest, discover_pairs
from modules import BaselineUNet, ImprovedUNet
from utils import hard_segmentation_metrics, seed_everything, write_json


def composite_segmentation_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    ce = F.cross_entropy(logits, target)
    probabilities = logits.softmax(1)
    one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    numerator = 2 * (probabilities * one_hot).sum((0, 2, 3)) + 1e-6
    denominator = probabilities.sum((0, 2, 3)) + one_hot.sum((0, 2, 3)) + 1e-6
    return ce + (1 - (numerator / denominator).mean())


@torch.no_grad()
def evaluate_batches(model: nn.Module, batches, device: torch.device, num_classes: int) -> dict:
    model.eval(); logits_all = []; targets_all = []; started = time.perf_counter()
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    for images, masks in batches:
        logits_all.append(model(images.to(device)).cpu()); targets_all.append(masks.cpu())
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
    parser.add_argument("--batch-size", type=int, default=8); parser.add_argument("--seed", type=int, default=42); parser.add_argument("--num-classes", type=int, default=4)
    args = parser.parse_args(); seed_everything(args.seed); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = build_subject_split_manifest(discover_pairs(args.data_root), args.seed)
    output = Path(args.output_dir) / args.model; output.mkdir(parents=True, exist_ok=True); write_json(output / "split_manifest.json", manifest)
    loaders = {name: _loader(rows, args.batch_size, name == "train") for name, rows in manifest.items()}
    model = (BaselineUNet if args.model == "baseline" else ImprovedUNet)(num_classes=args.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3); best = float("-inf"); history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0
        for images, masks, _ in loaders["train"]:
            optimizer.zero_grad(); loss = composite_segmentation_loss(model(images.to(device)), masks.to(device), args.num_classes); loss.backward(); optimizer.step(); total += loss.item()
        validation = evaluate_batches(model, ((x, y) for x, y, _ in loaders["validation"]), device, args.num_classes)
        history.append({"epoch": epoch, "train_loss": total / max(len(loaders["train"]), 1), **validation})
        if validation["mean_dice"] > best:
            best = validation["mean_dice"]; torch.save({"model": args.model, "state_dict": model.state_dict(), "num_classes": args.num_classes}, output / "best.pt")
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=True); model.load_state_dict(checkpoint["state_dict"])
    results = evaluate_batches(model, ((x, y) for x, y, _ in loaders["test"]), device, args.num_classes); results.update({"model": args.model, "seed": args.seed, "test_cases": len(manifest["test"])})
    write_json(output / "test_metrics.json", results)
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "mean_dice", "mean_iou", "inference_seconds", "peak_vram_mb"]); writer.writeheader(); writer.writerows(history)
    print(results)


if __name__ == "__main__": main()
