"""Create a machine-readable, patient-disjoint audit manifest for HipMRI volumes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dataset import VolumePair, build_patient_split_manifest, discover_volume_pairs, validate_patient_split_manifest


def build_dataset_manifest(pairs: list[VolumePair], seed: int = 42) -> dict[str, Any]:
    """Record validated cases, metadata and a deterministic patient-level split."""

    split_manifest = build_patient_split_manifest(pairs, seed=seed)
    validate_patient_split_manifest(split_manifest)
    split_by_patient = {
        patient_id: split
        for split in ("train", "validation", "test")
        for patient_id in split_manifest[split]
    }
    if len(split_by_patient) != len({pair.patient_id for pair in pairs}):
        raise ValueError("every patient must occur in exactly one split")

    cases = [
        {
            "case_id": pair.case_id,
            "patient_id": pair.patient_id,
            "image_path": pair.image_path.as_posix(),
            "mask_path": pair.mask_path.as_posix(),
            "shape": list(pair.shape),
            "spacing": list(pair.spacing),
            "orientation": list(pair.orientation),
            "labels": list(pair.labels),
            "split": split_by_patient[pair.patient_id],
        }
        for pair in sorted(pairs, key=lambda item: item.case_id)
    ]
    split_case_counts = {split: sum(case["split"] == split for case in cases) for split in ("train", "validation", "test")}
    split_patient_counts = {split: len(split_manifest[split]) for split in ("train", "validation", "test")}
    labels = sorted({label for pair in pairs for label in pair.labels})
    return {
        "schema_version": 1,
        "seed": seed,
        "splits": split_manifest,
        "summary": {
            "case_count": len(cases),
            "patient_count": len(split_by_patient),
            "labels": labels,
            "split_case_counts": split_case_counts,
            "split_patient_counts": split_patient_counts,
        },
        "cases": cases,
    }


def write_dataset_manifest(pairs: list[VolumePair], output_path: Path | str, seed: int = 42) -> dict[str, Any]:
    """Build and atomically save the audit manifest without copying source data."""

    manifest = build_dataset_manifest(pairs, seed=seed)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit HipMRI NIfTI pairs and write a patient-level manifest.")
    parser.add_argument("data_root", type=Path, help="directory containing semantic_MRs and semantic_labels_only")
    parser.add_argument("--output", type=Path, default=Path("evidence/dataset_manifest.json"))
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    manifest = write_dataset_manifest(discover_volume_pairs(arguments.data_root), arguments.output, seed=arguments.seed)
    print(json.dumps(manifest["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
