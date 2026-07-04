"""Q-network architectures for DQN agents."""

from __future__ import annotations

import torch
from torch import nn


class QNetwork(nn.Module):
    """Fully connected Q-network with two hidden layers.

    ``state_dim`` should match ``env.reset().shape[0]`` and ``action_dim`` is
    eight for the simplified 3U target hunting environment. The optional
    dueling heads use shared feature layers, a value stream ``V(s)``, an
    advantage stream ``A(s, a)``, and combine them as
    ``Q(s, a) = V(s) + A(s, a) - mean_a A(s, a)``.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 8,
        hidden_dim: int = 128,
        second_hidden_dim: int | None = None,
        dueling: bool = False,
    ) -> None:
        super().__init__()
        self.dueling = bool(dueling)
        second_hidden_dim = int(second_hidden_dim if second_hidden_dim is not None else hidden_dim)
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, second_hidden_dim),
            nn.ReLU(),
        )

        if self.dueling:
            head_hidden = max(second_hidden_dim // 2, 1)
            self.value = nn.Sequential(
                nn.Linear(second_hidden_dim, head_hidden),
                nn.ReLU(),
                nn.Linear(head_hidden, 1),
            )
            self.advantage = nn.Sequential(
                nn.Linear(second_hidden_dim, head_hidden),
                nn.ReLU(),
                nn.Linear(head_hidden, action_dim),
            )
        else:
            self.head = nn.Linear(second_hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature(x)
        if not self.dueling:
            return self.head(features)

        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)
