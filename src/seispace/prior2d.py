from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class DeconvBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class ContextDAE2D(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        channels: tuple[int, ...] = (128, 64, 32, 16),
    ) -> None:
        super().__init__()
        self.encoder = nn.ModuleList()
        current = embed_dim
        for channel in channels:
            self.encoder.append(ConvBlock2D(current, channel))
            current = channel

        reversed_channels = list(reversed(channels))
        self.decoders = nn.ModuleList()
        self.reductions = nn.ModuleList()
        for index in range(1, len(reversed_channels)):
            in_channels = reversed_channels[index - 1]
            skip_channels = reversed_channels[index]
            out_channels = skip_channels
            self.decoders.append(DeconvBlock2D(in_channels, skip_channels))
            self.reductions.append(
                nn.Sequential(
                    nn.Conv2d(2 * skip_channels, out_channels, 3, padding=1),
                    nn.GroupNorm(min(8, out_channels), out_channels),
                    nn.ReLU(inplace=True),
                )
            )
        self.head = nn.Conv2d(reversed_channels[-1], embed_dim, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        skips = []
        x = inputs
        for index, block in enumerate(self.encoder):
            x = block(x)
            if index < len(self.encoder) - 1:
                skips.append(x)
        for decoder, reduction in zip(self.decoders, self.reductions):
            x = decoder(x)
            skip = skips.pop()
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
            x = reduction(torch.cat([x, skip], dim=1))
        if x.shape[-2:] != inputs.shape[-2:]:
            x = F.interpolate(
                x, size=inputs.shape[-2:], mode="bilinear", align_corners=False
            )
        x = self.head(x)
        return x


class DualSparsePrior2D(nn.Module):
    """Two-dimensional dual-sparse seismic prior."""

    def __init__(
        self,
        patch_height: int = 16,
        patch_width: int = 16,
        stride_height: int = 8,
        stride_width: int = 8,
        embed_dim: int = 128,
        dae_channels: tuple[int, ...] = (64, 32, 16, 8),
        init_embedding: bool = True,
        embed_loss_weight: float = 0.1,
    ) -> None:
        super().__init__()
        coefficient_dim = patch_height * patch_width
        if embed_dim > coefficient_dim:
            raise ValueError(
                "embed_dim cannot exceed the number of 2D DCT coefficients"
            )
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.stride_height = stride_height
        self.stride_width = stride_width
        self.embed_dim = embed_dim
        self.embed_loss_weight = embed_loss_weight
        self.register_buffer("dct_weight", self._build_dct_weight())
        self.coefficient_dim = coefficient_dim
        projection = torch.zeros(coefficient_dim, coefficient_dim)
        projection[:embed_dim, :embed_dim] = torch.eye(embed_dim)
        self.register_buffer("embedding_projection", projection)
        self.embed = nn.Conv2d(coefficient_dim, embed_dim, 1, bias=False)
        self.inverse_embed = nn.Conv2d(embed_dim, coefficient_dim, 1, bias=False)
        if init_embedding:
            self._init_embedding()
        self.dae = ContextDAE2D(embed_dim, channels=dae_channels)

    def _build_dct_weight(self) -> torch.Tensor:
        height = self._dct_matrix(self.patch_height)
        width = self._dct_matrix(self.patch_width)
        basis = height[:, None, :, None] * width[None, :, None, :]
        return basis.reshape(
            self.patch_height * self.patch_width,
            1,
            self.patch_height,
            self.patch_width,
        )

    @staticmethod
    def _dct_matrix(size: int) -> torch.Tensor:
        frequency = torch.arange(size, dtype=torch.float32)[:, None]
        sample = torch.arange(size, dtype=torch.float32)[None, :]
        matrix = torch.cos(
            math.pi * frequency * (2.0 * sample + 1.0) / (2.0 * size)
        )
        matrix[0] *= math.sqrt(1.0 / size)
        matrix[1:] *= math.sqrt(2.0 / size)
        return matrix

    def _init_embedding(self) -> None:
        projection = torch.eye(self.coefficient_dim)[: self.embed_dim]
        self.embed.weight.data.copy_(projection[..., None, None])
        self.inverse_embed.weight.data.copy_(projection.T[..., None, None])

    def _pad_input(self, inputs: torch.Tensor) -> torch.Tensor:
        height, width = inputs.shape[-2:]
        target_height = (
            math.ceil((height - self.patch_height) / self.stride_height)
            * self.stride_height
            + self.patch_height
        )
        target_width = (
            math.ceil((width - self.patch_width) / self.stride_width)
            * self.stride_width
            + self.patch_width
        )
        return F.pad(
            inputs,
            (0, target_width - width, 0, target_height - height),
            mode="constant",
            value=0.0,
        )

    def dct(self, inputs: torch.Tensor) -> torch.Tensor:
        padded = self._pad_input(inputs)
        return F.conv2d(
            padded,
            self.dct_weight,
            stride=(self.stride_height, self.stride_width),
        )

    def idct(self, coefficients: torch.Tensor) -> torch.Tensor:
        patches = F.conv_transpose2d(
            coefficients,
            self.dct_weight,
            stride=(self.stride_height, self.stride_width),
        )
        count = F.conv_transpose2d(
            torch.ones(
                1,
                1,
                coefficients.shape[-2],
                coefficients.shape[-1],
                device=coefficients.device,
                dtype=coefficients.dtype,
            ),
            torch.ones(
                1,
                1,
                self.patch_height,
                self.patch_width,
                device=coefficients.device,
                dtype=coefficients.dtype,
            ),
            stride=(self.stride_height, self.stride_width),
        )
        return patches / count.clamp_min(1.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        coefficients = self.dct(inputs)
        embedded = self.embed(coefficients)
        denoised = self.dae(embedded)
        reconstructed = self.inverse_embed(denoised)
        return self.idct(reconstructed)

    def compute_loss(
        self,
        noisy: torch.Tensor,
        clean: torch.Tensor,
        embed_loss_weight: float | None = None,
    ) -> torch.Tensor:
        mean = noisy.mean(dim=(-2, -1), keepdim=True)
        std = noisy.std(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        normalized_noisy = (noisy - mean) / std
        normalized_clean = (clean - mean) / std
        coefficients = self.dct(normalized_clean)
        embedded = self.embed(self.dct(normalized_noisy))
        reconstructed = self.inverse_embed(self.dae(embedded))
        loss = F.l1_loss(reconstructed, coefficients)
        weight = (
            self.embed_loss_weight
            if embed_loss_weight is None
            else embed_loss_weight
        )
        if weight > 0:
            product = (
                self.inverse_embed.weight[:, :, 0, 0]
                @ self.embed.weight[:, :, 0, 0]
            )
            loss = loss + weight * F.mse_loss(product, self.embedding_projection)
        return loss

    @torch.no_grad()
    def denoise(self, sections: torch.Tensor) -> torch.Tensor:
        if sections.ndim == 2:
            sections = sections[None, None]
        elif sections.ndim == 3:
            sections = sections[:, None]
        self.eval()
        mean = sections.mean(dim=(-2, -1), keepdim=True)
        std = sections.std(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        normalized = (sections - mean) / std
        restored = self.forward(normalized)
        return restored * std + mean
