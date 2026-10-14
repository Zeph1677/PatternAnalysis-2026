"""OASIS PNG pairing and patient-safe segmentation data loading."""

from __future__ import annotations

import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _key_and_subject(path: Path, image_prefix: str, mask_prefix: str) -> tuple[str, str]:
    prefix = image_prefix if path.name.startswith(image_prefix) else mask_prefix
    stem = path.stem[len(prefix) :]
    match = re.match(r"(?P<subject>[^_]+)_(?P<slice>.+)", stem)
    if match is None:
        raise ValueError(f"cannot extract subject and slice from {path.name}")
    return stem, match.group("subject")


def discover_pairs(root: str | Path, image_prefix: str = "case_", mask_prefix: str = "seg_") -> list[dict[str, str]]:
    """Return aligned PNG image/mask pairs and their subject identifiers."""
    directory = Path(root)
    images = {path.stem[len(image_prefix) :]: path for path in directory.rglob(f"{image_prefix}*.png")}
    masks = {path.stem[len(mask_prefix) :]: path for path in directory.rglob(f"{mask_prefix}*.png")}
    missing = sorted(set(images) - set(masks))
    if missing:
        raise ValueError(f"missing mask for image keys: {missing[:3]}")
    if not images:
        raise ValueError(f"no {image_prefix}*.png files found in {directory}")
    pairs = []
    for key in sorted(images):
        _, subject = _key_and_subject(images[key], image_prefix, mask_prefix)
        pairs.append({"image": str(images[key]), "mask": str(masks[key]), "subject_id": subject, "case_id": key})
    return pairs


def build_subject_split_manifest(
    pairs: list[dict[str, str]], seed: int, train_fraction: float = 0.7, val_fraction: float = 0.15
) -> dict[str, list[dict[str, str]]]:
    """Split complete subjects once, then expand them back into slice records."""
    if not 0 < train_fraction < 1 or not 0 <= val_fraction < 1 or train_fraction + val_fraction >= 1:
        raise ValueError("fractions must be positive and leave a test partition")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for pair in pairs:
        grouped[pair["subject_id"]].append(pair)
    subjects = sorted(grouped)
    if len(subjects) < 3:
        raise ValueError("at least three subjects are required for train/validation/test")
    random.Random(seed).shuffle(subjects)
    train_end = max(1, int(len(subjects) * train_fraction))
    validation_end = max(train_end + 1, int(len(subjects) * (train_fraction + val_fraction)))
    validation_end = min(validation_end, len(subjects) - 1)
    allocations = {"train": subjects[:train_end], "validation": subjects[train_end:validation_end], "test": subjects[validation_end:]}
    return {name: [item for subject in member_ids for item in grouped[subject]] for name, member_ids in allocations.items()}


def validate_subject_split_manifest(
    manifest: dict[str, list[dict[str, str]]], data_root: str | Path
) -> dict[str, list[dict[str, str]]]:
    """Validate that a saved manifest remains complete, local, and subject-disjoint."""
    root = Path(data_root).resolve()
    seen_subjects: dict[str, str] = {}
    required = {"image", "mask", "subject_id", "case_id"}
    for partition in ("train", "validation", "test"):
        records = manifest.get(partition)
        if not records:
            raise ValueError(f"manifest partition {partition!r} is missing or empty")
        for record in records:
            missing = required - record.keys()
            if missing:
                raise ValueError(f"manifest record is missing fields: {sorted(missing)}")
            subject = record["subject_id"]
            existing = seen_subjects.get(subject)
            if existing is not None and existing != partition:
                raise ValueError(f"subject {subject!r} appears in multiple partitions")
            seen_subjects[subject] = partition
            for key in ("image", "mask"):
                path = Path(record[key]).resolve()
                if not path.is_file() or root not in path.parents:
                    raise ValueError(f"manifest {key} is outside data root or missing: {path}")
    return manifest


class SegmentationDataset(Dataset[tuple[torch.Tensor, torch.Tensor, dict[str, str]]]):
    def __init__(self, records: list[dict[str, str]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, str]]:
        record = self.records[index]
        image = np.asarray(Image.open(record["image"]).convert("L"), dtype=np.float32)
        mask = np.asarray(Image.open(record["mask"]), dtype=np.int64)
        # The course OASIS PNG export may store four tissue labels as display intensities.
        oasis_display_codes = {0: 0, 85: 1, 170: 2, 255: 3}
        observed = set(np.unique(mask).tolist())
        if observed.issubset(oasis_display_codes):
            mask = np.vectorize(oasis_display_codes.__getitem__)(mask).astype(np.int64)
        elif not observed.issubset({0, 1, 2, 3}):
            raise ValueError(f"unsupported mask labels {sorted(observed)} for {record['case_id']}")
        if image.shape != mask.shape:
            raise ValueError(f"unaligned image/mask shapes for {record['case_id']}")
        scale = max(float(image.std()), 1e-6)
        image = (image - float(image.mean())) / scale
        return torch.from_numpy(image).unsqueeze(0), torch.from_numpy(mask), record
