"""Create a reproducible, read-only audit of an OASIS PNG image/mask dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from dataset import discover_pairs


DISPLAY_TO_CLASS = {0: 0, 85: 1, 170: 2, 255: 3}


def _remap_labels(mask: np.ndarray, case_id: str) -> np.ndarray:
    observed = set(np.unique(mask).tolist())
    if observed.issubset(DISPLAY_TO_CLASS):
        return np.vectorize(DISPLAY_TO_CLASS.__getitem__)(mask).astype(np.uint8)
    if observed.issubset({0, 1, 2, 3}):
        return mask.astype(np.uint8)
    raise ValueError(f"unsupported mask labels {sorted(observed)} for {case_id}")


def _relative_pair(pair: dict[str, str], root: Path) -> dict[str, str]:
    return {
        "case_id": pair["case_id"],
        "subject_id": pair["subject_id"],
        "image": Path(pair["image"]).relative_to(root).as_posix(),
        "mask": Path(pair["mask"]).relative_to(root).as_posix(),
    }


def _write_sample(image: np.ndarray, mask: np.ndarray, case_id: str, output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(9, 3), constrained_layout=True)
    axes[0].imshow(image, cmap="gray"); axes[0].set_title("MRI slice")
    axes[1].imshow(mask, cmap="tab10", vmin=0, vmax=3); axes[1].set_title("Tissue labels")
    axes[2].imshow(image, cmap="gray")
    axes[2].imshow(np.ma.masked_where(mask == 0, mask), cmap="tab10", vmin=0, vmax=3, alpha=0.55)
    axes[2].set_title("Label overlay")
    for axis in axes:
        axis.set_axis_off()
    figure.suptitle(case_id, fontsize=10)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def audit_dataset(data_root: str | Path, output_dir: str | Path, sample_count: int = 4) -> dict:
    """Audit all paired PNGs without modifying the source dataset and write evidence outputs."""
    root = Path(data_root).resolve()
    output = Path(output_dir)
    samples = output / "samples"
    output.mkdir(parents=True, exist_ok=True)
    samples.mkdir(parents=True, exist_ok=True)
    pairs = discover_pairs(root)
    raw_labels: Counter[int] = Counter()
    remapped_labels: Counter[int] = Counter()
    shapes: Counter[str] = Counter()
    selected_indices = {round(index * (len(pairs) - 1) / max(sample_count - 1, 1)) for index in range(min(sample_count, len(pairs)))}
    sample_cases: list[str] = []
    for index, pair in enumerate(pairs):
        image = np.asarray(Image.open(pair["image"]).convert("L"), dtype=np.uint8)
        mask = np.asarray(Image.open(pair["mask"]), dtype=np.uint8)
        if image.shape != mask.shape:
            raise ValueError(f"unaligned image/mask shapes for {pair['case_id']}: {image.shape} != {mask.shape}")
        shapes[f"{image.shape[0]}x{image.shape[1]}"] += 1
        values, counts = np.unique(mask, return_counts=True)
        raw_labels.update({int(value): int(count) for value, count in zip(values, counts)})
        remapped = _remap_labels(mask, pair["case_id"])
        values, counts = np.unique(remapped, return_counts=True)
        remapped_labels.update({int(value): int(count) for value, count in zip(values, counts)})
        if index in selected_indices:
            _write_sample(image, remapped, pair["case_id"], samples / f"{pair['case_id']}.png")
            sample_cases.append(pair["case_id"])
    report = {
        "audit_version": 1,
        "source_root": str(root),
        "pair_count": len(pairs),
        "subject_count": len({pair["subject_id"] for pair in pairs}),
        "shapes": dict(sorted(shapes.items())),
        "raw_mask_labels": {str(key): value for key, value in sorted(raw_labels.items())},
        "remapped_mask_labels": {str(key): value for key, value in sorted(remapped_labels.items())},
        "sample_cases": sample_cases,
        "pairs": [_relative_pair(pair, root) for pair in pairs],
    }
    (output / "dataset_manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit paired OASIS PNG data and write a manifest plus sample panels")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", default="evidence/data_audit")
    parser.add_argument("--sample-count", type=int, default=4)
    args = parser.parse_args()
    report = audit_dataset(args.data_root, args.output_dir, args.sample_count)
    print(json.dumps({key: report[key] for key in ("pair_count", "subject_count", "shapes", "raw_mask_labels", "remapped_mask_labels", "sample_cases")}, indent=2))


if __name__ == "__main__":
    main()
