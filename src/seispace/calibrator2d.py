from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _trunc_normal_(tensor: torch.Tensor, std: float = 0.02) -> None:
    with torch.no_grad():
        tensor.normal_(0.0, std)
        tensor.clamp_(-2.0 * std, 2.0 * std)


class WindowAttention2D(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        head_dim: int,
        window_size: int,
        shifted: bool,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.head_dim = int(head_dim)
        self.num_heads = self.input_dim // self.head_dim
        self.window_size = int(window_size)
        self.shifted = bool(shifted)
        self.scale = self.head_dim**-0.5

        self.embedding_layer = nn.Linear(self.input_dim, 3 * self.input_dim)
        self.relative_position_params = nn.Parameter(
            torch.zeros(
                (2 * self.window_size - 1) ** 2,
                self.num_heads,
            )
        )
        _trunc_normal_(self.relative_position_params)
        self.relative_position_params = nn.Parameter(
            self.relative_position_params.view(
                2 * self.window_size - 1,
                2 * self.window_size - 1,
                self.num_heads,
            )
            .permute(2, 0, 1)
            .contiguous()
        )
        self.linear = nn.Linear(self.input_dim, self.output_dim)

    def _relative_embedding(self) -> torch.Tensor:
        coordinates = torch.arange(
            self.window_size, device=self.relative_position_params.device
        )
        relative = (
            coordinates[:, None] - coordinates[None, :] + self.window_size - 1
        )
        row = relative[:, None, :, None].expand(
            self.window_size,
            self.window_size,
            self.window_size,
            self.window_size,
        ).reshape(self.window_size**2, self.window_size**2)
        column = relative[None, :, None, :].expand(
            self.window_size,
            self.window_size,
            self.window_size,
            self.window_size,
        ).reshape(self.window_size**2, self.window_size**2)
        return self.relative_position_params[:, row, column]

    def _mask(
        self,
        window_h: int,
        window_w: int,
        device: torch.device,
    ) -> torch.Tensor:
        size = self.window_size
        shift = size // 2
        coordinates = torch.arange(size, device=device)
        same_side = (coordinates[:, None] < shift) == (
            coordinates[None, :] < shift
        )
        row_valid = same_side[:, None, :, None]
        column_valid = same_side[None, :, None, :]
        valid = torch.ones(
            window_h,
            window_w,
            size * size,
            size * size,
            dtype=torch.bool,
            device=device,
        )
        valid[-1, :] &= row_valid
        valid[:, -1] &= column_valid
        return valid.reshape(window_h * window_w, size * size, size * size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, height, width, channels = inputs.shape
        size = self.window_size
        if height % size or width % size:
            raise ValueError("Window attention input must be divisible by window size")
        window_h = height // size
        window_w = width // size
        if self.shifted:
            inputs = torch.roll(
                inputs,
                shifts=(-(size // 2), -(size // 2)),
                dims=(1, 2),
            )

        windows = (
            inputs.reshape(batch, window_h, size, window_w, size, channels)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(batch, window_h * window_w, size * size, channels)
        )
        qkv = self.embedding_layer(windows).reshape(
            batch,
            window_h * window_w,
            size * size,
            3,
            self.num_heads,
            self.head_dim,
        )
        query, key, value = qkv.unbind(dim=3)
        attention = torch.einsum(
            "bnphd,bnqhd->bnhpq", query, key
        ) * self.scale
        attention = attention + self._relative_embedding()[None, None]
        if self.shifted:
            attention = attention.masked_fill(
                ~self._mask(window_h, window_w, inputs.device)[None, None],
                float("-inf"),
            )
        probabilities = attention.softmax(dim=-1)
        output = torch.einsum(
            "bnhpq,bnqhd->bnphd", probabilities, value
        )
        output = output.reshape(
            batch,
            window_h * window_w,
            size * size,
            self.input_dim,
        )
        output = self.linear(output)
        output = (
            output.reshape(batch, window_h, window_w, size, size, self.output_dim)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(batch, height, width, self.output_dim)
        )
        if self.shifted:
            output = torch.roll(
                output,
                shifts=(size // 2, size // 2),
                dims=(1, 2),
            )
        return output


class TransformerBlock2D(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        head_dim: int,
        window_size: int,
        shifted: bool,
        input_resolution: int,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WindowAttention2D(
            input_dim,
            input_dim,
            head_dim,
            window_size,
            shifted=shifted and input_resolution > window_size,
        )
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.ReLU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs + self.msa(self.ln1(inputs))
        return inputs + self.mlp(self.ln2(inputs))


class ConvTransformerBlock2D(nn.Module):
    def __init__(
        self,
        conv_dim: int,
        transformer_dim: int,
        head_dim: int,
        window_size: int,
        shifted: bool,
        input_resolution: int,
    ) -> None:
        super().__init__()
        self.conv_dim = int(conv_dim)
        self.transformer_dim = int(transformer_dim)
        self.trans_block = TransformerBlock2D(
            self.transformer_dim,
            self.transformer_dim,
            head_dim,
            window_size,
            shifted=shifted,
            input_resolution=input_resolution,
        )
        total = self.conv_dim + self.transformer_dim
        self.conv1_1 = nn.Conv2d(total, total, 1)
        self.conv1_2 = nn.Conv2d(total, total, 1)
        self.conv_block = nn.Sequential(
            nn.Conv2d(self.conv_dim, self.conv_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.conv_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.conv_dim, self.conv_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.conv_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.conv_dim, self.conv_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.conv_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        conv, transformer = torch.split(
            self.conv1_1(inputs),
            (self.conv_dim, self.transformer_dim),
            dim=1,
        )
        conv = self.conv_block(conv) + conv
        transformer = transformer.permute(0, 2, 3, 1)
        transformer = self.trans_block(transformer).permute(0, 3, 1, 2)
        residual = self.conv1_2(torch.cat((conv, transformer), dim=1))
        return inputs + residual


class SeisPACECalibrator2D(nn.Module):
    """Two-dimensional baseline-anchored calibrator."""

    def __init__(
        self,
        in_channels: int = 2,
        config: tuple[int, ...] = (1, 1, 1, 1, 1),
        dim: int = 64,
        head_dim: int = 32,
        window_size: int = 8,
        input_resolution: int = 256,
    ) -> None:
        super().__init__()
        self.config = config
        self.dim = int(dim)
        self.window_size = int(window_size)
        self.m_head = nn.Sequential(
            nn.Conv2d(in_channels, self.dim, 3, padding=1, bias=False)
        )

        def stage(count: int, index: int) -> nn.Sequential:
            layers: list[nn.Module] = []
            for block_index in range(count):
                layers.append(
                    ConvTransformerBlock2D(
                        self.dim // 2,
                        self.dim // 2,
                        head_dim,
                        self.window_size,
                        shifted=bool(block_index % 2),
                        input_resolution=input_resolution + index,
                    )
                )
            layers.append(nn.Conv2d(self.dim, self.dim, 3, padding=1, bias=False))
            return nn.Sequential(*layers)

        self.m1 = stage(config[0], 0)
        self.m2 = stage(config[1], 1)
        self.m3 = stage(config[2], 2)
        self.m4 = stage(config[2], 3)
        self.m5 = stage(config[2], 4)
        self.m_tail = nn.Sequential(
            nn.Conv2d(self.dim, 1, 3, padding=1, bias=False)
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            _trunc_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        height, width = inputs.shape[-2:]
        multiple = self.window_size
        pad_height = math.ceil(height / multiple) * multiple - height
        pad_width = math.ceil(width / multiple) * multiple - width
        if pad_height or pad_width:
            inputs = F.pad(inputs, (0, pad_width, 0, pad_height))
        x1 = self.m_head(inputs)
        x2 = self.m1(x1)
        x3 = self.m2(x2)
        x4 = self.m3(x3)
        x5 = self.m4(x4 + x3)
        x6 = self.m5(x5 + x2)
        output = self.m_tail(x6 + x1)
        return output[..., :height, :width]
