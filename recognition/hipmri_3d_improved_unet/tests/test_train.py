import torch

from train import composite_3d_loss, evaluate_3d_batches, select_validation_checkpoint


def test_composite_3d_loss_accepts_final_and_auxiliary_logits():
    target = torch.randint(0, 2, (1, 8, 8, 8))
    final = torch.randn(1, 2, 8, 8, 8, requires_grad=True)
    auxiliary = torch.randn(1, 2, 4, 4, 4, requires_grad=True)
    loss = composite_3d_loss((final, (auxiliary,)), target, num_classes=2)
    loss.backward()
    assert torch.isfinite(loss)
    assert final.grad is not None and auxiliary.grad is not None


def test_evaluator_returns_per_class_metrics_and_resource_fields():
    model = torch.nn.Conv3d(1, 2, 1)
    batches = [(torch.randn(1, 1, 8, 8, 8), torch.randint(0, 2, (1, 8, 8, 8)))]
    result = evaluate_3d_batches(model, batches, torch.device("cpu"), num_classes=2)
    assert {"mean_dice", "mean_iou", "per_class", "inference_seconds", "peak_vram_mb"} <= set(result)


def test_checkpoint_selection_uses_validation_metric_only():
    history = [{"epoch": 1, "validation_mean_dice": 0.5, "test_mean_dice": 0.9}, {"epoch": 2, "validation_mean_dice": 0.7, "test_mean_dice": 0.1}]
    assert select_validation_checkpoint(history)["epoch"] == 2
