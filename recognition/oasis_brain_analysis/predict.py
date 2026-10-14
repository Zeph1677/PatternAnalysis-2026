"""Create qualitative OASIS segmentation evidence from a trained checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import SegmentationDataset, build_subject_split_manifest, discover_pairs
from modules import BaselineUNet, ImprovedUNet


def error_overlay(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Colour false positives red, false negatives blue, and wrong tissue classes yellow."""
    overlay = np.zeros((*target.shape, 3), dtype=np.uint8)
    overlay[(prediction != 0) & (target == 0)] = (255, 0, 0)
    overlay[(prediction == 0) & (target != 0)] = (0, 0, 255)
    overlay[(prediction != 0) & (target != 0) & (prediction != target)] = (255, 255, 0)
    return overlay


def limit_prediction_records(records: list[dict[str, str]], maximum: int) -> list[dict[str, str]]:
    """Limit smoke-test panels deterministically; zero evaluates every held-out case."""
    return records if maximum == 0 else records[:maximum]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create OASIS inference/error panels")
    parser.add_argument("--data-root", required=True); parser.add_argument("--checkpoint", required=True); parser.add_argument("--output-dir", default="artifacts/predictions")
    parser.add_argument("--seed", type=int, default=42); parser.add_argument("--max-cases", type=int, default=0); args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); payload = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = (BaselineUNet if payload["model"] == "baseline" else ImprovedUNet)(num_classes=payload["num_classes"]).to(device); model.load_state_dict(payload["state_dict"]); model.eval()
    records = limit_prediction_records(build_subject_split_manifest(discover_pairs(args.data_root), args.seed)["test"], args.max_cases); output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    failures = []
    for image, target, record in DataLoader(SegmentationDataset(records), batch_size=1, shuffle=False):
        with torch.no_grad(): prediction = model(image.to(device)).argmax(1).cpu().numpy()[0]
        truth, source = target.numpy()[0], image.numpy()[0, 0]; error = error_overlay(prediction, truth); score = float((prediction != truth).mean()); failures.append((score, record["case_id"][0]))
        figure, axes = plt.subplots(1, 4, figsize=(14, 4)); panels = [(source, "Input", "gray"), (truth, "Ground truth", "tab20"), (prediction, "Prediction", "tab20"), (error, "FP red / FN blue", None)]
        for axis, (pixels, title, cmap) in zip(axes, panels): axis.imshow(pixels, cmap=cmap); axis.set_title(title); axis.axis("off")
        figure.tight_layout(); figure.savefig(output / f"{record['case_id'][0]}.png", dpi=160); plt.close(figure)
    (output / "failure_ranking.txt").write_text("\n".join(f"{score:.6f}\t{case}" for score, case in sorted(failures, reverse=True)), encoding="utf-8")


if __name__ == "__main__": main()
