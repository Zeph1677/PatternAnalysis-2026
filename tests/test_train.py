import torch

from train import composite_segmentation_loss, count_trainable_parameters, evaluate_batches, history_row, limit_records


def test_composite_loss_is_finite_for_multiclass_logits():
    logits = torch.randn(2, 4, 8, 8, requires_grad=True)
    target = torch.randint(0, 4, (2, 8, 8))
    loss = composite_segmentation_loss(logits, target, num_classes=4)
    assert torch.isfinite(loss)
    loss.backward()


def test_evaluator_returns_metrics_and_resource_fields():
    model = torch.nn.Conv2d(1, 4, 1)
    batches = [(torch.randn(2, 1, 8, 8), torch.randint(0, 4, (2, 8, 8)))]
    result = evaluate_batches(model, batches, torch.device("cpu"), num_classes=4)
    assert {"mean_dice", "mean_iou", "inference_seconds", "peak_vram_mb"} <= set(result)


def test_parameter_counter_counts_only_trainable_parameters():
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 1))
    model[1].weight.requires_grad_(False)
    model[1].bias.requires_grad_(False)
    assert count_trainable_parameters(model) == 8


def test_limit_records_keeps_a_deterministic_prefix():
    records = [{"case_id": str(index)} for index in range(4)]
    assert limit_records(records, 2) == records[:2]
    assert limit_records(records, 0) == records


def test_history_row_excludes_nested_metric_fields():
    row = history_row({"epoch": 1, "mean_dice": 0.5, "per_class": {}, "classes_present": [0]})
    assert row == {"epoch": 1, "mean_dice": 0.5}
