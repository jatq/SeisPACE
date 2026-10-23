from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock1D(nn.Module):
    def __init__(
        self,
        channels: int,
        dilation: int = 1,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.block = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            nn.GroupNorm(min(8, channels), channels),
            nn.SiLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.GroupNorm(min(8, channels), channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.block(inputs))


class DownBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 5, stride=2, padding=2),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
            ResidualBlock1D(out_channels, kernel_size=3),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class UpBlock1D(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels, out_channels, 4, stride=2, padding=1)
        self.reduce = nn.Sequential(
            nn.Conv1d(out_channels + skip_channels, out_channels, 3, padding=1),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
            ResidualBlock1D(out_channels, kernel_size=3),
        )

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(inputs)
        if x.shape[-1] != skip.shape[-1]:
            x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
        return self.reduce(torch.cat([x, skip], dim=1))


class SeisPACECalibrator1D(nn.Module):
    """Predicts a calibrated trace from the noisy input and a frozen prior."""

    def __init__(
        self,
        channels: tuple[int, ...] = (32, 64, 128, 256),
        in_channels: int = 2,
        residual: bool = True,
        zero_head: bool = True,
    ) -> None:
        super().__init__()
        self.residual = residual
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, channels[0], 7, padding=3),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.SiLU(inplace=True),
            ResidualBlock1D(channels[0]),
        )
        self.down = nn.ModuleList()
        for index in range(1, len(channels)):
            self.down.append(DownBlock1D(channels[index - 1], channels[index]))
        self.bottleneck = nn.Sequential(
            ResidualBlock1D(channels[-1], dilation=2),
            ResidualBlock1D(channels[-1], dilation=4),
            ResidualBlock1D(channels[-1], dilation=8),
        )
        self.up = nn.ModuleList()
        reversed_channels = list(reversed(channels))
        for index in range(1, len(reversed_channels)):
            self.up.append(
                UpBlock1D(
                    reversed_channels[index - 1],
                    reversed_channels[index],
                    reversed_channels[index],
                )
            )
        self.head = nn.Conv1d(channels[0], 1, 1)
        if zero_head:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        prior = inputs[:, 1:2]
        x = self.stem(inputs)
        skips = [x]
        for index, block in enumerate(self.down):
            x = block(x)
            if index < len(self.down) - 1:
                skips.append(x)
        x = self.bottleneck(x)
        for block in self.up:
            x = block(x, skips.pop())
        output = self.head(x)
        if self.residual:
            return prior + output, output
        return output, output
