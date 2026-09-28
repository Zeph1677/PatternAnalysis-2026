from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dataset import build_subject_split_manifest, discover_pairs


def _write_pair(root: Path, subject: str, slice_number: int) -> None:
    image = np.full((8, 8), slice_number, dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    Image.fromarray(image).save(root / f"case_{subject}_slice_{slice_number}.png")
    Image.fromarray(mask).save(root / f"seg_{subject}_slice_{slice_number}.png")


def test_pair_discovery_and_subject_split_are_deterministic(tmp_path: Path):
    for subject in ("001", "002", "003", "004", "005"):
        _write_pair(tmp_path, subject, 0)
        _write_pair(tmp_path, subject, 1)
    pairs = discover_pairs(tmp_path)
    first = build_subject_split_manifest(pairs, seed=7, train_fraction=0.6, val_fraction=0.2)
    second = build_subject_split_manifest(pairs, seed=7, train_fraction=0.6, val_fraction=0.2)
    assert first == second
    subjects = {name: {item["subject_id"] for item in rows} for name, rows in first.items()}
    assert not (subjects["train"] & subjects["validation"])
    assert not (subjects["train"] & subjects["test"])


def test_pair_discovery_rejects_missing_mask(tmp_path: Path):
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(tmp_path / "case_001_slice_0.png")
    with pytest.raises(ValueError, match="missing mask"):
        discover_pairs(tmp_path)
