"""Experience replay buffer for DQN agents."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Deque, Tuple

import numpy as np


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Fixed-size replay memory for off-policy DQN updates."""

    def __init__(self, capacity: int = 10_000) -> None:
        self.capacity = int(capacity)
        self.buffer: Deque[Transition] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append(
            Transition(
                state=np.asarray(state, dtype=np.float32),
                action=int(action),
                reward=float(reward),
                next_state=np.asarray(next_state, dtype=np.float32),
                done=bool(done),
            )
        )

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, int(batch_size))
        states = np.stack([item.state for item in batch])
        actions = np.asarray([item.action for item in batch], dtype=np.int64)
        rewards = np.asarray([item.reward for item in batch], dtype=np.float32)
        next_states = np.stack([item.next_state for item in batch])
        dones = np.asarray([item.done for item in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def can_sample(self, batch_size: int) -> bool:
        """Return whether at least ``batch_size`` transitions are available."""

        return len(self.buffer) >= int(batch_size)
