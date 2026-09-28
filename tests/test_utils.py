import torch
import json

from utils import hard_segmentation_metrics, write_json


def test_perfect_prediction_has_unit_dice_and_iou():
    target = torch.tensor([[[0, 1], [2, 3]]])
    logits = torch.full((1, 4, 2, 2), -10.0)
    logits.scatter_(1, target.unsqueeze(1), 10.0)

    metrics = hard_segmentation_metrics(logits, target, num_classes=4)

    assert metrics["mean_dice"] == 1.0
    assert metrics["mean_iou"] == 1.0
    assert metrics["per_class"]["3"]["dice"] == 1.0


def test_absent_class_is_excluded_from_macro_average():
    target = torch.zeros((1, 2, 2), dtype=torch.long)
    logits = torch.zeros((1, 3, 2, 2))
    logits[:, 0] = 2.0

    metrics = hard_segmentation_metrics(logits, target, num_classes=3)

    assert metrics["classes_present"] == [0]
    assert metrics["mean_dice"] == 1.0


def test_json_writer_replaces_non_finite_metrics_with_null(tmp_path):
    write_json(tmp_path / "metrics.json", {"absent_dice": float("nan")})
    assert json.loads((tmp_path / "metrics.json").read_text()) == {"absent_dice": None}
