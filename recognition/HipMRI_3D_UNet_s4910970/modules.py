"""Plain and residual deep-supervised 3D U-Net architectures for HipMRI."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1), nn.InstanceNorm3d(out_channels), nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1), nn.InstanceNorm3d(out_channels), nn.LeakyReLU(inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.body = ConvBlock3D(in_channels, out_channels)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv3d(in_channels, out_channels, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(self.body(value) + self.skip(value), inplace=True)


class _UNet3D(nn.Module):
    def __init__(self, block: type[nn.Module], in_channels: int, num_classes: int, base_channels: int, deep_supervision: bool) -> None:
        super().__init__()
        b = base_channels
        self.enc1, self.enc2, self.enc3 = block(in_channels, b), block(b, 2 * b), block(2 * b, 4 * b)
        self.pool = nn.MaxPool3d(2)
        self.bottleneck = block(4 * b, 8 * b)
        self.up3, self.up2, self.up1 = nn.ConvTranspose3d(8 * b, 4 * b, 2, 2), nn.ConvTranspose3d(4 * b, 2 * b, 2, 2), nn.ConvTranspose3d(2 * b, b, 2, 2)
        self.dec3, self.dec2, self.dec1 = block(8 * b, 4 * b), block(4 * b, 2 * b), block(2 * b, b)
        self.head = nn.Conv3d(b, num_classes, 1)
        self.deep_supervision = deep_supervision
        self.aux3 = nn.Conv3d(4 * b, num_classes, 1) if deep_supervision else None
        self.aux2 = nn.Conv3d(2 * b, num_classes, 1) if deep_supervision else None

    def forward(self, value: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if any(size < 16 or size % 8 for size in value.shape[-3:]):
            raise ValueError(f"volume dimensions must be at least 16 and divisible by 8, got {tuple(value.shape[-3:])}")
        e1 = self.enc1(value)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat((self.up3(b), e3), dim=1))
        d2 = self.dec2(torch.cat((self.up2(d3), e2), dim=1))
        d1 = self.dec1(torch.cat((self.up1(d2), e1), dim=1))
        final = self.head(d1)
        if not self.training or not self.deep_supervision:
            return final
        assert self.aux3 is not None and self.aux2 is not None
        return final, (self.aux3(d3), self.aux2(d2))


class BaselineUNet3D(_UNet3D):
    """Matched standard volumetric U-Net baseline."""

    def __init__(self, in_channels: int = 1, num_classes: int = 2, base_channels: int = 16) -> None:
        super().__init__(ConvBlock3D, in_channels, num_classes, base_channels, deep_supervision=False)


class ResidualContextBlock3D(nn.Module):
    """Pre-activation residual context block with spatial-channel dropout."""

    def __init__(self, in_channels: int, out_channels: int, dropout_probability: float) -> None:
        super().__init__()
        self.context = nn.Sequential(
            nn.InstanceNorm3d(in_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.Dropout3d(p=dropout_probability),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
        )
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv3d(in_channels, out_channels, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(self.context(value) + self.skip(value), negative_slope=0.01, inplace=True)


class UpsampleConv3D(nn.Module):
    """Interpolation followed by convolution, avoiding transposed-convolution artifacts."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.convolution = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.convolution(F.interpolate(value, scale_factor=2, mode="trilinear", align_corners=False))


class ContextImprovedUNet3D(nn.Module):
    """Residual, regularised 3D U-Net with decoder-logit fusion and deep supervision.

    The public contract intentionally matches the previous Improved model: training
    returns final logits plus two auxiliary logits; evaluation returns final logits.
    """

    architecture_version = "improved_context_v2"

    def __init__(self, in_channels: int = 1, num_classes: int = 2, base_channels: int = 16, dropout_probability: float = 0.3) -> None:
        super().__init__()
        if not 0.0 <= dropout_probability < 1.0:
            raise ValueError("dropout_probability must be in [0, 1)")
        b = base_channels
        block = lambda source, target: ResidualContextBlock3D(source, target, dropout_probability)
        self.enc1, self.enc2, self.enc3 = block(in_channels, b), block(b, 2 * b), block(2 * b, 4 * b)
        self.bottleneck = block(4 * b, 8 * b)
        self.pool = nn.MaxPool3d(2)
        self.up3, self.up2, self.up1 = UpsampleConv3D(8 * b, 4 * b), UpsampleConv3D(4 * b, 2 * b), UpsampleConv3D(2 * b, b)
        self.dec3, self.dec2, self.dec1 = block(8 * b, 4 * b), block(4 * b, 2 * b), block(2 * b, b)
        self.head = nn.Conv3d(b, num_classes, 1)
        self.aux3, self.aux2 = nn.Conv3d(4 * b, num_classes, 1), nn.Conv3d(2 * b, num_classes, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if any(size < 16 or size % 8 for size in value.shape[-3:]):
            raise ValueError(f"volume dimensions must be at least 16 and divisible by 8, got {tuple(value.shape[-3:])}")
        e1 = self.enc1(value)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        bottleneck = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat((self.up3(bottleneck), e3), dim=1))
        d2 = self.dec2(torch.cat((self.up2(d3), e2), dim=1))
        d1 = self.dec1(torch.cat((self.up1(d2), e1), dim=1))
        auxiliary_3, auxiliary_2 = self.aux3(d3), self.aux2(d2)
        final = self.head(d1)
        final = final + F.interpolate(auxiliary_2, size=final.shape[-3:], mode="trilinear", align_corners=False)
        final = final + F.interpolate(auxiliary_3, size=final.shape[-3:], mode="trilinear", align_corners=False)
        if self.training:
            return final, (auxiliary_3, auxiliary_2)
        return final


# Parameterised layers and forward computations preserved from commit 4232e0d;
# current input-size validation is retained.
class ImprovedUNet3D(_UNet3D):
    """Report Improved model with residual blocks and deep supervision."""

    architecture_version = "improved_v1"

    def __init__(self, in_channels: int = 1, num_classes: int = 2, base_channels: int = 16) -> None:
        super().__init__(ResidualBlock3D, in_channels, num_classes, base_channels, deep_supervision=True)


def build_model(config: dict[str, object]) -> nn.Module:
    """Construct the saved architecture; missing Improved version means historical v1.

    Current training writes an explicit version. This compatibility default is for
    historical configs, never an instruction to reinterpret incompatible weights.
    Checkpoint callers must still use strict state-dict loading.
    """
    model_name = config.get("model")
    dimensions = {}
    for field, default in (("in_channels", 1), ("num_classes", None), ("base_channels", None)):
        value = config.get(field, default)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field} must be a positive integer")
        dimensions[field] = value
    version = config.get("architecture_version")
    if model_name == "baseline":
        if version not in (None, "baseline_v1"):
            raise ValueError(f"unsupported baseline architecture version: {version}")
        return BaselineUNet3D(**dimensions)
    if model_name != "improved":
        raise ValueError(f"unsupported model: {model_name}")
    if version in (None, ImprovedUNet3D.architecture_version, "legacy_improved_v1"):
        return ImprovedUNet3D(**dimensions)
    if version == ContextImprovedUNet3D.architecture_version:
        return ContextImprovedUNet3D(
            **dimensions, dropout_probability=config.get("dropout_probability", 0.3)
        )
    raise ValueError(f"unsupported improved architecture version: {version}")
