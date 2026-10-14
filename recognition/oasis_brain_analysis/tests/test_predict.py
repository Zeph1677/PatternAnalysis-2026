from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

from predict import error_overlay, limit_prediction_records, load_prediction_records
from utils import write_json


def test_error_overlay_marks_false_positive_and_false_negative_pixels():
    prediction = np.array([[0, 1], [1, 0]])
    target = np.array([[0, 0], [1, 1]])
    overlay = error_overlay(prediction, target)
    assert overlay[0, 1].tolist() == [255, 0, 0]
    assert overlay[1, 1].tolist() == [0, 0, 255]


def test_error_overlay_marks_wrong_foreground_class_separately():
    prediction = np.array([[2]])
    target = np.array([[1]])
    assert error_overlay(prediction, target)[0, 0].tolist() == [255, 255, 0]


def test_prediction_record_limit_keeps_prefix():
    records = [{"case_id": str(index)} for index in range(3)]
    assert limit_prediction_records(records, 2) == records[:2]


def _manifest(root: Path) -> dict[str, list[dict[str, str]]]:
    result = {"train": [], "validation": [], "test": []}
    for partition, subject in zip(result, ("001", "002", "003")):
        image_path = root / f"case_{subject}_slice_0.png"
        mask_path = root / f"seg_{subject}_slice_0.png"
        Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(image_path)
        Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(mask_path)
        result[partition].append({"image": str(image_path), "mask": str(mask_path), "subject_id": subject, "case_id": f"{subject}_slice_0"})
    return result


def test_prediction_records_use_test_partition_from_saved_manifest(tmp_path: Path):
    path = tmp_path / "split.json"
    write_json(path, _manifest(tmp_path))
    assert [record["case_id"] for record in load_prediction_records(tmp_path, path, 0)] == ["003_slice_0"]


def test_predict_cli_advertises_formal_manifest_option():
    result = subprocess.run([sys.executable, "predict.py", "--help"], check=True, capture_output=True, text=True)
    assert "--split-manifest" in result.stdout
