from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from dataset import build_patient_split_manifest, discover_volume_pairs


def _write_volume(path: Path, array: np.ndarray, affine: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(array, np.eye(4) if affine is None else affine), path)


def _make_pair(root: Path, patient: str, shape: tuple[int, int, int] = (8, 10, 12)) -> None:
    image = np.linspace(0, 1, np.prod(shape), dtype=np.float32).reshape(shape)
    label = np.zeros(shape, dtype=np.uint8)
    label[2:5, 3:7, 4:8] = 1
    _write_volume(root / "semantic_MRs" / patient / "scan.nii.gz", image)
    _write_volume(root / "semantic_labels_only" / patient / "scan.nii.gz", label)


def test_discovers_recursive_nifti_pairs_with_patient_ids(tmp_path: Path):
    for patient in ("patient_01", "patient_02", "patient_03"):
        _make_pair(tmp_path, patient)

    pairs = discover_volume_pairs(tmp_path)

    assert [pair.patient_id for pair in pairs] == ["patient_01", "patient_02", "patient_03"]
    assert all(not pair.image_path.is_absolute() for pair in pairs)
    assert all(not pair.mask_path.is_absolute() for pair in pairs)


def test_discovery_rejects_missing_mask_and_shape_mismatch(tmp_path: Path):
    _make_pair(tmp_path, "patient_01")
    _write_volume(tmp_path / "semantic_MRs" / "patient_02" / "scan.nii.gz", np.ones((8, 10, 12), dtype=np.float32))
    _make_pair(tmp_path, "patient_03")
    _write_volume(tmp_path / "semantic_labels_only" / "patient_03" / "scan.nii.gz", np.zeros((8, 10, 11), dtype=np.uint8))

    with pytest.raises(ValueError, match="missing mask|shape mismatch"):
        discover_volume_pairs(tmp_path)


def test_patient_split_is_deterministic_and_has_zero_overlap(tmp_path: Path):
    for number in range(9):
        _make_pair(tmp_path, f"patient_{number:02d}")

    manifest = build_patient_split_manifest(discover_volume_pairs(tmp_path), seed=42)

    groups = [set(manifest[name]) for name in ("train", "validation", "test")]
    assert all(groups)
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert sorted(set.union(*groups)) == [f"patient_{number:02d}" for number in range(9)]
