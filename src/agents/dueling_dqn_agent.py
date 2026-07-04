"""Dueling DQN variant."""

from __future__ import annotations

from typing import Dict

from agents.dqn_agent import DQNAgent


class DuelingDQNAgent(DQNAgent):
    """DQN with a dueling value/advantage architecture."""

    def __init__(self, state_dim: int, action_dim: int = 8, config: Dict | None = None) -> None:
        super().__init__(state_dim, action_dim, config, dueling=True)
