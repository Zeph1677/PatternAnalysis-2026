"""Baseline and improved 2D U-Net architectures."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.body = ConvBlock(in_channels, out_channels)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.body(x) + self.skip(x), inplace=True)


class AttentionGate(nn.Module):
    def __init__(self, skip_channels: int, gate_channels: int) -> None:
        super().__init__()
        self.skip_projection = nn.Conv2d(skip_channels, skip_channels, 1)
        self.gate_projection = nn.Conv2d(gate_channels, skip_channels, 1)
        self.score = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(skip_channels, 1, 1), nn.Sigmoid())

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        gate = F.interpolate(gate, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return skip * self.score(self.skip_projection(skip) + self.gate_projection(gate))


class SkipThrough(nn.Module):
    """Keep a skip connection unchanged while matching AttentionGate's interface."""
    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return skip


class _UNet(nn.Module):
    def __init__(self, block: type[nn.Module], in_channels: int, num_classes: int, base_channels: int, attention: bool) -> None:
        super().__init__()
        b = base_channels
        self.enc1, self.enc2, self.enc3 = block(in_channels, b), block(b, 2*b), block(2*b, 4*b)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = block(4*b, 8*b)
        self.dropout = nn.Dropout2d(0.15) if attention else nn.Identity()
        self.up3, self.up2, self.up1 = nn.ConvTranspose2d(8*b, 4*b, 2, 2), nn.ConvTranspose2d(4*b, 2*b, 2, 2), nn.ConvTranspose2d(2*b, b, 2, 2)
        self.gate3 = AttentionGate(4*b, 4*b) if attention else SkipThrough()
        self.gate2 = AttentionGate(2*b, 2*b) if attention else SkipThrough()
        self.gate1 = AttentionGate(b, b) if attention else SkipThrough()
        self.dec3, self.dec2, self.dec1 = block(8*b, 4*b), block(4*b, 2*b), block(2*b, b)
        self.head = nn.Conv2d(b, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x); e2 = self.enc2(self.pool(e1)); e3 = self.enc3(self.pool(e2))
        center = self.dropout(self.bottleneck(self.pool(e3)))
        d3 = self.up3(center); d3 = self.dec3(torch.cat((self.gate3(e3, d3), d3), 1))
        d2 = self.up2(d3); d2 = self.dec2(torch.cat((self.gate2(e2, d2), d2), 1))
        d1 = self.up1(d2); d1 = self.dec1(torch.cat((self.gate1(e1, d1), d1), 1))
        return self.head(d1)


class BaselineUNet(_UNet):
    def __init__(self, in_channels: int = 1, num_classes: int = 4, base_channels: int = 32) -> None:
        super().__init__(ConvBlock, in_channels, num_classes, base_channels, attention=False)


class ImprovedUNet(_UNet):
    def __init__(self, in_channels: int = 1, num_classes: int = 4, base_channels: int = 32) -> None:
        super().__init__(ResidualBlock, in_channels, num_classes, base_channels, attention=True)
