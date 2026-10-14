from pathlib import Path

import numpy as np
from PIL import Image

from audit import audit_dataset


def _write_pair(root: Path, subject: str, slice_number: int, labels: list[int]) -> None:
    image_dir = root / "images"
    mask_dir = root / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    name = f"{subject}_slice_{slice_number}.nii.png"
    Image.fromarray(np.full((4, 5), 100, dtype=np.uint8)).save(image_dir / f"case_{name}")
    Image.fromarray(np.asarray(labels, dtype=np.uint8).reshape(4, 5)).save(mask_dir / f"seg_{name}")


def test_audit_dataset_writes_relative_manifest_and_label_summary(tmp_path: Path) -> None:
    _write_pair(tmp_path, "001", 0, [0, 85, 170, 255] * 5)
    _write_pair(tmp_path, "002", 1, [0, 1, 2, 3] * 5)

    report = audit_dataset(tmp_path, tmp_path / "output", sample_count=2)

    assert report["pair_count"] == 2
    assert report["subject_count"] == 2
    assert report["shapes"] == {"4x5": 2}
    assert report["raw_mask_labels"] == {"0": 10, "1": 5, "2": 5, "3": 5, "85": 5, "170": 5, "255": 5}
    assert report["remapped_mask_labels"] == {"0": 10, "1": 10, "2": 10, "3": 10}
    assert all(not Path(item["image"]).is_absolute() for item in report["pairs"])
    assert len(report["sample_cases"]) == 2
    assert (tmp_path / "output" / "dataset_manifest.json").is_file()
    assert (tmp_path / "output" / "samples" / "001_slice_0.nii.png").is_file()
