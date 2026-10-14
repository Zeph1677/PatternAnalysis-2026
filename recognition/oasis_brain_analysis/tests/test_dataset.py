from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dataset import SegmentationDataset, build_subject_split_manifest, discover_pairs, validate_subject_split_manifest


def _write_pair(root: Path, subject: str, slice_number: int) -> None:
    image = np.full((8, 8), slice_number, dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    Image.fromarray(image).save(root / f"case_{subject}_slice_{slice_number}.png")
    Image.fromarray(mask).save(root / f"seg_{subject}_slice_{slice_number}.png")


def _record(root: Path, subject: str) -> dict[str, str]:
    _write_pair(root, subject, 0)
    return {
        "image": str(root / f"case_{subject}_slice_0.png"),
        "mask": str(root / f"seg_{subject}_slice_0.png"),
        "subject_id": subject,
        "case_id": f"{subject}_slice_0",
    }


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


def test_subject_split_keeps_all_three_partitions_nonempty_at_minimum_size(tmp_path: Path):
    for subject in ("001", "002", "003"):
        _write_pair(tmp_path, subject, 0)
    manifest = build_subject_split_manifest(discover_pairs(tmp_path), seed=7)
    assert all(manifest[partition] for partition in ("train", "validation", "test"))


def test_pair_discovery_rejects_missing_mask(tmp_path: Path):
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(tmp_path / "case_001_slice_0.png")
    with pytest.raises(ValueError, match="missing mask"):
        discover_pairs(tmp_path)


def test_pair_discovery_searches_nested_image_and_mask_directories(tmp_path: Path):
    images = tmp_path / "images"; masks = tmp_path / "masks"; images.mkdir(); masks.mkdir()
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(images / "case_101_slice_0.nii.png")
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(masks / "seg_101_slice_0.nii.png")
    assert discover_pairs(tmp_path)[0]["subject_id"] == "101"


def test_dataset_remaps_hypermri_mask_codes_to_class_indices(tmp_path: Path):
    image = np.zeros((2, 2), dtype=np.uint8); mask = np.array([[0, 85], [170, 255]], dtype=np.uint8)
    Image.fromarray(image).save(tmp_path / "case_001_slice_0.png"); Image.fromarray(mask).save(tmp_path / "seg_001_slice_0.png")
    _, target, _ = SegmentationDataset(discover_pairs(tmp_path))[0]
    assert target.tolist() == [[0, 1], [2, 3]]


def test_manifest_validator_rejects_subject_in_two_partitions(tmp_path: Path):
    first = _record(tmp_path, "001")
    manifest = {"train": [first], "validation": [first], "test": [_record(tmp_path, "002")]}
    with pytest.raises(ValueError, match="multiple partitions"):
        validate_subject_split_manifest(manifest, tmp_path)


def test_manifest_validator_returns_valid_records(tmp_path: Path):
    manifest = {"train": [_record(tmp_path, "001")], "validation": [_record(tmp_path, "002")], "test": [_record(tmp_path, "003")]}
    assert validate_subject_split_manifest(manifest, tmp_path) == manifest
