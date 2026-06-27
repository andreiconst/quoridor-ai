"""Small residual CNN with policy and value heads, AlphaZero-style."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..engine.game import ACTION_SIZE

IN_PLANES = 6


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class QuoridorNet(nn.Module):
    def __init__(self, channels: int = 64, num_blocks: int = 6, board_size: int = 9):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(IN_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])

        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, 1),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, ACTION_SIZE),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(board_size * board_size, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.blocks(x)
        policy_logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return policy_logits, value

    @torch.no_grad()
    def predict(self, planes: torch.Tensor):
        """planes: (C,H,W) or (B,C,H,W) tensor. Returns (policy_probs, value)."""
        single = planes.dim() == 3
        if single:
            planes = planes.unsqueeze(0)
        logits, value = self.forward(planes)
        probs = F.softmax(logits, dim=-1)
        if single:
            return probs[0], value[0].item()
        return probs, value
