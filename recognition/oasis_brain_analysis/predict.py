"""Create qualitative OASIS segmentation evidence from a trained checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import SegmentationDataset, build_subject_split_manifest, discover_pairs, validate_subject_split_manifest
from modules import BaselineUNet, ImprovedUNet
from utils import write_json


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


def load_prediction_records(data_root: str | Path, manifest_path: str | Path, maximum: int) -> list[dict[str, str]]:
    """Load the saved held-out records rather than deriving a new split."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return limit_prediction_records(validate_subject_split_manifest(manifest, data_root)["test"], maximum)


def _load_model(checkpoint_path: str | Path, device: torch.device):
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = (BaselineUNet if payload["model"] == "baseline" else ImprovedUNet)(num_classes=payload["num_classes"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload["model"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create OASIS inference/error panels")
    parser.add_argument("--data-root", required=True); parser.add_argument("--checkpoint", required=True); parser.add_argument("--output-dir", default="artifacts/predictions")
    parser.add_argument("--seed", type=int, default=42); parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--split-manifest", help="Saved split manifest required for formal held-out panels")
    parser.add_argument("--comparison-checkpoint", help="Optional second model checkpoint for six-panel comparison")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_name = _load_model(args.checkpoint, device)
    comparison_model = None
    comparison_name = None
    if args.comparison_checkpoint:
        comparison_model, comparison_name = _load_model(args.comparison_checkpoint, device)
    if args.split_manifest:
        records = load_prediction_records(args.data_root, args.split_manifest, args.max_cases)
    else:
        records = limit_prediction_records(build_subject_split_manifest(discover_pairs(args.data_root), args.seed)["test"], args.max_cases)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    failures = []
    for image, target, record in DataLoader(SegmentationDataset(records), batch_size=1, shuffle=False):
        with torch.no_grad(): prediction = model(image.to(device)).argmax(1).cpu().numpy()[0]
        truth, source = target.numpy()[0], image.numpy()[0, 0]; error = error_overlay(prediction, truth); score = float((prediction != truth).mean())
        failure = {"case_id": record["case_id"][0], f"{model_name}_pixel_error": score}
        panels = [(source, "Input", "gray"), (truth, "Ground truth", "tab20"), (prediction, f"{model_name} prediction", "tab20"), (error, f"{model_name} FP/FN", None)]
        if comparison_model is not None:
            with torch.no_grad(): comparison = comparison_model(image.to(device)).argmax(1).cpu().numpy()[0]
            comparison_error = error_overlay(comparison, truth)
            failure[f"{comparison_name}_pixel_error"] = float((comparison != truth).mean())
            panels.extend([(comparison, f"{comparison_name} prediction", "tab20"), (comparison_error, f"{comparison_name} FP/FN", None)])
        failures.append(failure)
        figure, axes = plt.subplots(1, len(panels), figsize=(3.5 * len(panels), 4))
        for axis, (pixels, title, cmap) in zip(axes, panels): axis.imshow(pixels, cmap=cmap); axis.set_title(title); axis.axis("off")
        figure.tight_layout(); figure.savefig(output / f"{record['case_id'][0]}.png", dpi=160); plt.close(figure)
    failures.sort(key=lambda item: item[f"{model_name}_pixel_error"], reverse=True)
    write_json(output / "failure_ranking.json", {"primary_model": model_name, "comparison_model": comparison_name, "cases": failures})


if __name__ == "__main__": main()
