import torch

from modules import BaselineUNet, ImprovedUNet


def test_baseline_unet_preserves_spatial_shape_and_emits_class_logits():
    output = BaselineUNet(in_channels=1, num_classes=4, base_channels=8)(torch.randn(2, 1, 64, 64))
    assert output.shape == (2, 4, 64, 64)


def test_improved_unet_preserves_spatial_shape_and_emits_class_logits():
    output = ImprovedUNet(in_channels=1, num_classes=4, base_channels=8)(torch.randn(2, 1, 64, 64))
    assert output.shape == (2, 4, 64, 64)
