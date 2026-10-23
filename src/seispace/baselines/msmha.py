from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class KerasStyleMultiHeadAttention(nn.Module):
    """Self-attention matching the dimensions used by the MSMHA Keras code."""

    def __init__(self, embed_dim: int, num_heads: int, key_dim: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        inner_dim = num_heads * key_dim
        self.query = nn.Linear(embed_dim, inner_dim)
        self.key = nn.Linear(embed_dim, inner_dim)
        self.value = nn.Linear(embed_dim, inner_dim)
        self.output = nn.Linear(inner_dim, embed_dim)
        self.scale = key_dim**-0.5

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = inputs.shape
        query = self.query(inputs).view(
            batch, tokens, self.num_heads, self.key_dim
        ).transpose(1, 2)
        key = self.key(inputs).view(
            batch, tokens, self.num_heads, self.key_dim
        ).transpose(1, 2)
        value = self.value(inputs).view(
            batch, tokens, self.num_heads, self.key_dim
        ).transpose(1, 2)
        attention = torch.softmax(query @ key.transpose(-2, -1) * self.scale, dim=-1)
        output = (attention @ value).transpose(1, 2).reshape(batch, tokens, -1)
        return self.output(output)


class MultiStageAttention(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int = 8) -> None:
        super().__init__()
        side = math.isqrt(hidden_dim)
        if side * side != hidden_dim:
            raise ValueError("MSMHA hidden_dim must be a perfect square")
        self.hidden_dim = hidden_dim
        self.side = side
        self.dense1 = nn.Linear(input_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.attention = KerasStyleMultiHeadAttention(
            side, num_heads=num_heads, key_dim=side
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.dense1(inputs.flatten(1)), negative_slope=0.1)
        x = F.leaky_relu(self.dense2(x), negative_slope=0.1)
        attention_input = x.reshape(-1, self.side, self.side)
        attended = self.attention(attention_input).flatten(1)
        return attended + x


class MSMHA(nn.Module):
    """PyTorch implementation of the two-stage self-supervised MSMHA network."""

    def __init__(
        self,
        patch_height: int = 48,
        patch_width: int = 48,
        first_dim: int = 225,
        second_dim: int = 4225,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        input_dim = patch_height * patch_width
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.first = MultiStageAttention(input_dim, first_dim, num_heads)
        self.first_output = nn.Linear(first_dim, input_dim)
        self.second = MultiStageAttention(input_dim, second_dim, num_heads)
        self.second_output = nn.Linear(second_dim, input_dim)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        first = self.first_output(self.first(inputs)).reshape(
            -1, self.patch_height, self.patch_width
        )
        second = self.second_output(self.second(first)).reshape(
            -1, self.patch_height, self.patch_width
        )
        return first, second

    def denoise(self, section: torch.Tensor) -> torch.Tensor:
        _, output = self(section.unsqueeze(0))
        return output[0]
