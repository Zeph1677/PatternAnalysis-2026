import torch

from train import composite_segmentation_loss, evaluate_batches


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
