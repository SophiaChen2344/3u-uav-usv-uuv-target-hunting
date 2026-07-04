"""Deep Q-learning agent for centralized UUV trajectory decisions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from models.q_network import QNetwork
from utils.replay_buffer import ReplayBuffer


@dataclass
class DQNUpdateStats:
    loss: float
    q_mean: float


class DQNAgent:
    """Vanilla DQN with target network and epsilon-greedy exploration."""

    def __init__(self, state_dim: int, action_dim: int = 8, config: Dict | None = None, dueling: bool = False) -> None:
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.config = dict(config or {})
        self.gamma = float(self.config.get("gamma", 0.95))
        self.batch_size = int(self.config.get("batch_size", 128))
        self.target_update_interval = int(self.config.get("target_update_interval", 100))
        self.gradient_clip = float(self.config.get("gradient_clip", 5.0))
        self.epsilon = float(self.config.get("epsilon", self.config.get("epsilon_start", 0.9)))
        self.epsilon_min = float(self.config.get("epsilon_min", self.config.get("epsilon_end", 0.05)))
        self.epsilon_decay = float(self.config.get("epsilon_decay", 1.0))

        requested_device = str(self.config.get("device", "auto")).lower()
        if requested_device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(requested_device)

        hidden_dim = int(self.config.get("hidden_dim", 128))
        second_hidden_dim = int(self.config.get("second_hidden_dim", hidden_dim))
        self.policy_net = QNetwork(
            state_dim,
            action_dim,
            hidden_dim=hidden_dim,
            second_hidden_dim=second_hidden_dim,
            dueling=dueling,
        ).to(self.device)
        self.target_net = QNetwork(
            state_dim,
            action_dim,
            hidden_dim=hidden_dim,
            second_hidden_dim=second_hidden_dim,
            dueling=dueling,
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = Adam(self.policy_net.parameters(), lr=float(self.config.get("learning_rate", 1e-3)))
        self.loss_fn = nn.SmoothL1Loss()
        self.replay_buffer = ReplayBuffer(int(self.config.get("replay_capacity", 10_000)))
        self.update_steps = 0
        self.dueling = bool(dueling)

    def select_action(self, state: np.ndarray, epsilon: float | None = None) -> int:
        """Choose an action with epsilon-greedy exploration."""

        epsilon = self.epsilon if epsilon is None else float(epsilon)
        if random.random() < epsilon:
            return random.randrange(self.action_dim)

        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def optimize_model(self) -> Optional[DQNUpdateStats]:
        """Run one DQN optimization step from replay memory."""

        if not self.replay_buffer.can_sample(self.batch_size):
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.policy_net(states_t).gather(1, actions_t)
        target_q_values = self._target_q_values(next_states_t)
        targets = rewards_t + self.gamma * (1.0 - dones_t) * target_q_values

        loss = self.loss_fn(q_values, targets.detach())
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.gradient_clip)
        self.optimizer.step()

        self.update_steps += 1
        if self.target_update_interval > 0 and self.update_steps % self.target_update_interval == 0:
            self.update_target_network()

        return DQNUpdateStats(loss=float(loss.item()), q_mean=float(q_values.mean().item()))

    def update(self) -> Optional[DQNUpdateStats]:
        """Backward-compatible alias for :meth:`optimize_model`."""

        return self.optimize_model()

    def _target_q_values(self, next_states_t: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.target_net(next_states_t).max(dim=1, keepdim=True).values

    def update_target_network(self) -> None:
        """Copy online Q-network weights into the target network."""

        self.target_net.load_state_dict(self.policy_net.state_dict())

    def sync_target_network(self) -> None:
        """Backward-compatible alias for :meth:`update_target_network`."""

        self.update_target_network()

    def decay_epsilon(self) -> float:
        """Apply multiplicative epsilon decay and return the new value."""

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return self.epsilon

    def save(self, path: str | Path) -> None:
        """Save model, optimizer, and agent metadata to ``path``."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "config": self.config,
                "dueling": self.dueling,
                "policy_state_dict": self.policy_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "update_steps": self.update_steps,
            },
            path,
        )

    def load(self, path: str | Path, map_location: str | torch.device | None = None) -> None:
        """Load model, target network, optimizer, and epsilon from ``path``."""

        checkpoint = torch.load(path, map_location=map_location or self.device)
        self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        self.target_net.load_state_dict(checkpoint.get("target_state_dict", checkpoint["policy_state_dict"]))
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = float(checkpoint.get("epsilon", self.epsilon))
        self.update_steps = int(checkpoint.get("update_steps", self.update_steps))
