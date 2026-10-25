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
        if any(size % 8 for size in value.shape[-3:]):
            raise ValueError(f"volume dimensions must be divisible by 8, got {tuple(value.shape[-3:])}")
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


class ImprovedUNet3D(_UNet3D):
    """Hard-tier 3D U-Net with residual blocks and decoder deep supervision."""

    def __init__(self, in_channels: int = 1, num_classes: int = 2, base_channels: int = 16) -> None:
        super().__init__(ResidualBlock3D, in_channels, num_classes, base_channels, deep_supervision=True)
