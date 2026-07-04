"""Double DQN variant."""

from __future__ import annotations

import torch

from agents.dqn_agent import DQNAgent


class DoubleDQNAgent(DQNAgent):
    """Double DQN target calculation.

    The online network selects the greedy next action, while the target network
    evaluates that selected action. This reduces the overestimation bias of
    vanilla DQN's max-over-target-network update.
    """

    def _target_q_values(self, next_states_t: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            next_actions = self.policy_net(next_states_t).argmax(dim=1, keepdim=True)
            return self.target_net(next_states_t).gather(1, next_actions)
