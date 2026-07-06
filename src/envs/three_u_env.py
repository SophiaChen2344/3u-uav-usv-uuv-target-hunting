"""NumPy-only UAV-USV-UUV cooperative target hunting environment.

This environment follows a compact abstraction of the 3U hunting problem:

- one UAV monitors the 400 m x 400 m area from altitude ``h``;
- one USV performs a random surface relay walk at ``z = 0``;
- three UUVs are represented by a single underwater group center ``G``;
- the target stays at the hunting depth and can either move away from ``G`` or
  use a one-step Stackelberg best response.

The API is intentionally Gym-like but does not depend on Gym/Gymnasium.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from sensing.fisher_information import (
    fim_metrics,
    fisher_information_matrix,
    range_bearing_measurement,
)


@dataclass
class ThreeUState:
    """Continuous state of the simplified 3U hunting system."""

    uav_position: np.ndarray
    usv_position: np.ndarray
    uuv_center: np.ndarray
    target_position: np.ndarray
    direction_gw: np.ndarray
    total_voyage_distance: float
    step_count: int
    previous_target_distance: float
    total_energy_used: float
    energy_budget: float
    observed_target_position: np.ndarray | None = None
    belief_target_position: np.ndarray | None = None
    target_escaped: bool = False
    constraint_violation: bool = False

    @property
    def target_estimate(self) -> np.ndarray:
        """Compatibility alias for algorithms that expect an estimated target."""

        if self.belief_target_position is not None:
            return self.belief_target_position
        return self.target_position

    @property
    def uuv_positions(self) -> np.ndarray:
        """Compatibility view: three UUVs colocated at the group center."""

        return np.repeat(self.uuv_center[None, :], 3, axis=0)

    @property
    def uuv_energy(self) -> np.ndarray:
        """Compatibility view for scripts that inspect remaining UUV energy."""

        budget_per_uuv = max(0.0, self.energy_budget - self.total_energy_used / 3.0)
        return np.full(3, budget_per_uuv, dtype=float)


class ThreeUEnv:
    """Simplified 3U target hunting simulator.

    State vector:
        ``[U(t), S(t), G(t), W(t), direction_GW, total_voyage_distance]``

    Action ids use eight compass directions for the UUV group center:
        0 east, 1 north-east, 2 north, 3 north-west,
        4 west, 5 south-west, 6 south, 7 south-east.
    """

    primitive_action_count = 8
    action_space_n = 8
    group_center_mode = True

    ACTION_DIRECTIONS = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [-1.0, 1.0],
            [-1.0, 0.0],
            [-1.0, -1.0],
            [0.0, -1.0],
            [1.0, -1.0],
        ],
        dtype=float,
    )
    ACTION_DIRECTIONS = ACTION_DIRECTIONS / np.linalg.norm(ACTION_DIRECTIONS, axis=1, keepdims=True)
    ACTION_NAMES = ("E", "NE", "N", "NW", "W", "SW", "S", "SE")

    def __init__(self, config: Dict | None = None, seed: int | None = None) -> None:
        self.config = deepcopy(config or {})
        self.env_config = self.config.get("environment", self.config)
        self.rng = np.random.default_rng(seed)
        self._seed = seed

        self.area_size = float(self.env_config.get("area_size", 400.0))
        self.half_area = self.area_size / 2.0
        self.max_depth = abs(float(self.env_config.get("max_depth", 120.0)))
        self.max_steps = int(self.env_config.get("max_steps", 100))
        self.dt = float(self.env_config.get("dt", 1.0))
        self.num_uuvs = int(self.env_config.get("num_uuvs", 3))
        self.M = self.num_uuvs

        self.uav_height = float(self.env_config.get("uav_height", self.env_config.get("h", 120.0)))
        self.hunting_depth = -abs(float(self.env_config.get("target_depth", 120.0)))
        self.uuv_initial_depth = -abs(float(self.env_config.get("initial_uuv_depth", 120.0)))

        self.uuv_speed = float(self.env_config.get("uuv_speed", 8.0))
        self.usv_speed = float(self.env_config.get("usv_speed", 4.0))
        self.target_speed = float(self.env_config.get("target_speed", 3.0))
        self.search_radius = float(
            self.env_config.get("uuv_search_radius", self.env_config.get("capture_radius", 18.0))
        )
        self.escape_distance = float(self.env_config.get("escape_distance", self.area_size * 0.72))
        self.uav_usv_range = float(self.env_config.get("uav_usv_range", 700.0))
        self.usv_uuv_range = float(
            self.env_config.get("usv_uuv_range", self.env_config.get("uuv_acoustic_range", 700.0))
        )

        self.energy_base = float(self.env_config.get("energy_base", 2.0))
        self.energy_linear = float(self.env_config.get("energy_linear", 0.8))
        self.energy_quadratic = float(self.env_config.get("energy_quadratic", 0.04))
        self.energy_budget = float(self.env_config.get("uuv_energy_budget", 65_000.0))
        self.safety_config = dict(self.config.get("safety", {}))
        self.use_lyapunov = bool(self.safety_config.get("use_lyapunov", False))
        self.sensing_config = dict(self.config.get("sensing", {}))
        self.use_fim = bool(self.sensing_config.get("use_fim", False))
        self.use_belief_state = bool(self.sensing_config.get("use_belief_state", False))
        self.observation_noise_uav = float(self.sensing_config.get("observation_noise_uav", 20.0))
        self.observation_noise_usv = float(self.sensing_config.get("observation_noise_usv", 10.0))
        self.observation_noise_uuv = float(self.sensing_config.get("observation_noise_uuv", 5.0))
        self.fim_regularization = float(self.sensing_config.get("fim_regularization", 1e-6))
        self.info_reward_weight = float(self.sensing_config.get("info_reward_weight", 0.01))
        self.belief_update_alpha = float(self.sensing_config.get("belief_update_alpha", 0.35))
        self.uuv_observation_range = float(
            self.sensing_config.get("uuv_observation_range", self.env_config.get("sonar_radius", 130.0))
        )
        self.game_config = dict(self.config.get("game", {}))
        self.use_stackelberg = bool(self.game_config.get("use_stackelberg", False))
        self.use_intelligent_target = bool(
            self.game_config.get(
                "use_intelligent_target",
                self.use_stackelberg or str(self.game_config.get("target_model", "")).lower() == "intelligent",
            )
        )

        self.state: ThreeUState | None = None
        self.last_info: Dict[str, float] = {}
        self.history: Dict[str, list[float]] = {}
        self.action_tuples = [(action,) for action in range(self.action_space_n)]
        self._last_boundary_violation = False
        self.current_fim = np.zeros((3, 3), dtype=float)
        self.current_fim_metrics = self._empty_fim_metrics()
        self.current_belief_error = 0.0
        self.current_normalized_logdet = 0.0
        self._last_game_step_info: Dict[str, float] = self._default_game_info()

    def seed(self, seed: int | None = None) -> list[int | None]:
        """Set the environment random seed."""

        self._seed = seed
        self.rng = np.random.default_rng(seed)
        return [seed]

    def reset(self) -> np.ndarray:
        """Reset the hunting episode and return the initial state vector."""

        center = self.half_area
        uav_position = np.array([center, center, self.uav_height], dtype=float)
        usv_position = np.array([center, center, 0.0], dtype=float)
        uuv_center = np.array([center, center, self.uuv_initial_depth], dtype=float)

        target_position = self._initial_target_position()
        direction_gw = self._unit_vector(target_position - uuv_center)
        target_distance = self._distance(uuv_center, target_position)

        self.state = ThreeUState(
            uav_position=uav_position,
            usv_position=usv_position,
            uuv_center=uuv_center,
            target_position=target_position,
            direction_gw=direction_gw,
            total_voyage_distance=0.0,
            step_count=0,
            previous_target_distance=target_distance,
            total_energy_used=0.0,
            energy_budget=self.energy_budget,
            observed_target_position=target_position.copy(),
            belief_target_position=target_position.copy(),
        )
        self._update_sensing(initial=True)
        self.history = {
            "energy": [],
            "us_distance": [],
            "sg_distance": [],
            "connected_fraction": [],
            "target_distance": [target_distance],
            "fim_logdet": [],
            "fim_trace_inv": [],
            "fim_min_eigenvalue": [],
            "fim_condition_number": [],
            "belief_error": [],
            "target_true_position": [],
            "target_belief_position": [],
            "lyapunov_value": [],
            "lyapunov_delta": [],
            "lyapunov_condition_satisfied": [],
            "safety_violation": [],
            "original_action": [],
            "executed_action": [],
            "action_replaced": [],
            "target_best_response_action": [],
            "stackelberg_selected_action": [],
            "stackelberg_leader_cost": [],
            "target_utility": [],
            "stackelberg_changed_action": [],
        }
        self._last_boundary_violation = False
        self._last_game_step_info = self._default_game_info()
        self.last_info = self._build_info(reward=0.0, done=False)
        self._attach_reset_safety_info()
        return self.get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        """Advance one step with an 8-direction UUV group-center action."""

        self._require_state()
        action = int(np.clip(action, 0, self.action_space_n - 1))
        original_action = action
        previous_state_vector = self.get_state().copy()
        previous_metrics = dict(self.last_info)
        selection_info = self._select_executed_action(action)
        action = int(selection_info.get("executed_action", action))
        target_decision = self._select_target_response(action, selection_info)

        previous_center = self.state.uuv_center.copy()
        self.state.previous_target_distance = self._target_distance()
        self._last_boundary_violation = False

        self._move_uuv_group(action)
        actual_move = self._distance(previous_center, self.state.uuv_center)
        self.state.total_voyage_distance += actual_move

        self._move_usv_randomly()
        self._move_target(target_decision)
        self.state.direction_gw = self._unit_vector(self.state.target_position - self.state.uuv_center)
        self._update_sensing(initial=False)

        step_energy = self._compute_step_energy(actual_move)
        self.state.total_energy_used += step_energy

        constraints_ok = self.check_constraints()
        self.state.constraint_violation = not constraints_ok
        captured = self._target_captured()
        escaped = self._target_escaped()
        reward = self.compute_reward()

        self.state.step_count += 1
        done = bool(captured or escaped or self.state.step_count >= self.max_steps)

        self._last_game_step_info = self._merge_game_step_info(selection_info, target_decision)
        self._append_logs(step_energy)
        info = self._build_info(reward=reward, done=done)
        self._attach_step_safety_info(
            info=info,
            previous_state=previous_state_vector,
            previous_metrics=previous_metrics,
            original_action=original_action,
            executed_action=action,
            selection_info=selection_info,
        )
        self.last_info = info
        return self.get_state(), reward, done, info

    def get_state(self) -> np.ndarray:
        """Return ``[U, S, G, W, direction_GW, L]`` as a float32 vector."""

        self._require_state()
        target_for_policy = self._policy_target_position()
        direction_for_policy = self._unit_vector(target_for_policy - self.state.uuv_center)
        return np.concatenate(
            [
                self.state.uav_position,
                self.state.usv_position,
                self.state.uuv_center,
                target_for_policy,
                direction_for_policy,
                np.array([self.state.total_voyage_distance], dtype=float),
            ]
        ).astype(np.float32)

    def compute_reward(self) -> float:
        """Compute the paper-style sparse hunting reward."""

        self._require_state()
        if self.state.constraint_violation or self.state.target_escaped:
            return -1.0
        if self._target_captured():
            return float(10.0 + self._information_reward())
        if self._target_distance() < self.state.previous_target_distance:
            return float(0.1 + self._information_reward())
        return float(-1.0 + self._information_reward())

    def check_constraints(self) -> bool:
        """Check boundary and simple communication constraints."""

        self._require_state()
        if self._last_boundary_violation:
            return False

        positions_ok = (
            self._xy_in_bounds(self.state.uav_position)
            and self._xy_in_bounds(self.state.usv_position)
            and self._xy_in_bounds(self.state.uuv_center)
            and self._xy_in_bounds(self.state.target_position)
        )
        depths_ok = (
            self.state.uav_position[2] >= 0.0
            and np.isclose(self.state.usv_position[2], 0.0)
            and self.state.uuv_center[2] <= 0.0
            and self.state.target_position[2] <= 0.0
        )
        links_ok = self._us_distance() <= self.uav_usv_range and self._sg_distance() <= self.usv_uuv_range
        return bool(positions_ok and depths_ok and links_ok)

    def render(self, mode: str = "human") -> str:
        """Render a compact text summary of the current episode state."""

        self._require_state()
        summary = (
            f"step={self.state.step_count} "
            f"G=({self.state.uuv_center[0]:.1f}, {self.state.uuv_center[1]:.1f}, {self.state.uuv_center[2]:.1f}) "
            f"W=({self.state.target_position[0]:.1f}, {self.state.target_position[1]:.1f}, "
            f"{self.state.target_position[2]:.1f}) "
            f"d_GW={self._target_distance():.1f} belief_err={self.current_belief_error:.1f} "
            f"L={self.state.total_voyage_distance:.1f}"
        )
        if mode == "human":
            print(summary)
        return summary

    def greedy_action_toward_target(self) -> int:
        """Return the compass action whose direction best points from G to W."""

        self._require_state()
        return self.action_from_vector(self._policy_target_position()[:2] - self.state.uuv_center[:2])

    def action_from_vector(self, vector_xy: np.ndarray) -> int:
        """Map a 2D vector to the nearest of the eight movement actions."""

        vector_xy = np.asarray(vector_xy, dtype=float)
        norm = np.linalg.norm(vector_xy)
        if norm <= 1e-12:
            return 0
        unit = vector_xy / norm
        scores = self.ACTION_DIRECTIONS @ unit
        return int(np.argmax(scores))

    def _initial_target_position(self) -> np.ndarray:
        configured = self.env_config.get("target_initial_position")
        if configured is not None:
            target = np.asarray(configured, dtype=float)
            if target.size != 3:
                raise ValueError("target_initial_position must have three coordinates.")
            target[2] = self.hunting_depth
            return self._clip_position(target)

        center = self.half_area
        radius = self.rng.uniform(self.area_size * 0.22, self.area_size * 0.34)
        angle = self.rng.uniform(0.0, 2.0 * np.pi)
        xy = np.array([center + radius * np.cos(angle), center + radius * np.sin(angle)], dtype=float)
        xy = np.clip(xy, 0.0, self.area_size)
        return np.array([xy[0], xy[1], self.hunting_depth], dtype=float)

    def _move_uuv_group(self, action: int) -> None:
        move_xy = self.ACTION_DIRECTIONS[action] * self.uuv_speed * self.dt
        candidate = self.state.uuv_center.copy()
        candidate[:2] += move_xy
        clipped = self._clip_position(candidate)
        self._last_boundary_violation = self._last_boundary_violation or not np.allclose(candidate, clipped)
        self.state.uuv_center = clipped
        self.state.uuv_center[2] = self.uuv_initial_depth

    def _move_usv_randomly(self) -> None:
        angle = self.rng.uniform(0.0, 2.0 * np.pi)
        move = np.array([np.cos(angle), np.sin(angle), 0.0], dtype=float) * self.usv_speed * self.dt
        self.state.usv_position = self._clip_position(self.state.usv_position + move)
        self.state.usv_position[2] = 0.0

    def _move_target_away_from_group(self) -> None:
        direction = self._unit_vector(self.state.target_position - self.state.uuv_center)
        horizontal = direction[:2]
        if np.linalg.norm(horizontal) <= 1e-12:
            angle = self.rng.uniform(0.0, 2.0 * np.pi)
            horizontal = np.array([np.cos(angle), np.sin(angle)], dtype=float)
        else:
            horizontal = horizontal / np.linalg.norm(horizontal)

        candidate = self.state.target_position.copy()
        candidate[:2] += horizontal * self.target_speed * self.dt
        escaped_boundary = not self._xy_in_bounds(candidate)
        self.state.target_position = self._clip_position(candidate)
        self.state.target_position[2] = self.hunting_depth
        self.state.target_escaped = bool(escaped_boundary or self._target_distance() >= self.escape_distance)

    def _move_target(self, target_decision: Dict[str, float]) -> None:
        """Move the target with either the original heuristic or game response."""

        if str(target_decision.get("target_motion_model", "simple")) != "intelligent":
            self._move_target_away_from_group()
            return

        target_action = int(target_decision.get("target_best_response_action", 0))
        if target_action >= self.action_space_n:
            candidate = self.state.target_position.copy()
        else:
            move_xy = self.ACTION_DIRECTIONS[target_action] * self.target_speed * self.dt
            candidate = self.state.target_position.copy()
            candidate[:2] += move_xy

        escaped_boundary = not self._xy_in_bounds(candidate)
        self.state.target_position = self._clip_position(candidate)
        self.state.target_position[2] = self.hunting_depth
        self.state.target_escaped = bool(escaped_boundary or self._target_distance() >= self.escape_distance)

    def _compute_step_energy(self, actual_move: float) -> float:
        speed = actual_move / max(self.dt, 1e-12)
        per_uuv = self.energy_base * self.dt + self.energy_linear * actual_move + self.energy_quadratic * speed**2
        return float(self.num_uuvs * per_uuv)

    def _append_logs(self, step_energy: float) -> None:
        us_distance = self._us_distance()
        sg_distance = self._sg_distance()
        connected_fraction = 0.5 * float(us_distance <= self.uav_usv_range) + 0.5 * float(
            sg_distance <= self.usv_uuv_range
        )
        self.history["energy"].append(float(step_energy))
        self.history["us_distance"].append(us_distance)
        self.history["sg_distance"].append(sg_distance)
        self.history["connected_fraction"].append(connected_fraction)
        self.history["target_distance"].append(self._target_distance())
        self.history["fim_logdet"].append(float(self.current_fim_metrics.get("logdet", np.nan)))
        self.history["fim_trace_inv"].append(float(self.current_fim_metrics.get("trace_inv", np.nan)))
        self.history["fim_min_eigenvalue"].append(float(self.current_fim_metrics.get("min_eigenvalue", np.nan)))
        self.history["fim_condition_number"].append(float(self.current_fim_metrics.get("condition_number", np.nan)))
        self.history["belief_error"].append(float(self.current_belief_error))
        self.history["target_true_position"].append(self.state.target_position.copy())
        self.history["target_belief_position"].append(self._belief_target_position().copy())
        game_info = self._last_game_step_info or self._default_game_info()
        self.history["target_best_response_action"].append(float(game_info.get("target_best_response_action", -1.0)))
        self.history["stackelberg_selected_action"].append(float(game_info.get("stackelberg_selected_action", -1.0)))
        self.history["stackelberg_leader_cost"].append(float(game_info.get("stackelberg_leader_cost", 0.0)))
        self.history["target_utility"].append(float(game_info.get("target_utility", 0.0)))
        self.history["stackelberg_changed_action"].append(float(game_info.get("stackelberg_changed_action", 0.0)))

    def _build_info(self, reward: float, done: bool) -> Dict[str, float]:
        captured = self._target_captured()
        escaped = self._target_escaped()
        us_distance = self._us_distance()
        sg_distance = self._sg_distance()
        target_distance = self._target_distance()
        belief_position = self._belief_target_position()
        observed_position = self._observed_target_position()
        connected_fraction = 0.5 * float(us_distance <= self.uav_usv_range) + 0.5 * float(
            sg_distance <= self.usv_uuv_range
        )
        info = {
            "step": float(self.state.step_count),
            "reward": float(reward),
            "done": float(done),
            "captured": float(captured),
            "target_escaped": float(escaped),
            "constraints_satisfied": float(self.check_constraints()),
            "step_energy": float(self.history["energy"][-1]) if self.history["energy"] else 0.0,
            "total_energy_used": float(self.state.total_energy_used),
            "total_voyage_distance": float(self.state.total_voyage_distance),
            "us_distance": us_distance,
            "sg_distance": sg_distance,
            "mean_uuv_usv_distance": sg_distance,
            "usv_uav_distance": us_distance,
            "target_distance": target_distance,
            "mean_target_distance": target_distance,
            "min_target_distance": target_distance,
            "connected_fraction": connected_fraction,
            "all_connected": float(connected_fraction == 1.0),
            "remaining_energy_mean": float(np.mean(self.state.uuv_energy)),
            "remaining_energy_min": float(np.min(self.state.uuv_energy)),
            "fim_logdet": float(self.current_fim_metrics.get("logdet", np.nan)),
            "fim_trace_inv": float(self.current_fim_metrics.get("trace_inv", np.nan)),
            "fim_min_eigenvalue": float(self.current_fim_metrics.get("min_eigenvalue", np.nan)),
            "fim_condition_number": float(self.current_fim_metrics.get("condition_number", np.nan)),
            "normalized_logdet_fim": float(self.current_normalized_logdet),
            "belief_error": float(self.current_belief_error),
            "target_true_position": self.state.target_position.copy(),
            "target_belief_position": belief_position.copy(),
            "target_observed_position": observed_position.copy(),
            "target_true_x": float(self.state.target_position[0]),
            "target_true_y": float(self.state.target_position[1]),
            "target_true_z": float(self.state.target_position[2]),
            "target_belief_x": float(belief_position[0]),
            "target_belief_y": float(belief_position[1]),
            "target_belief_z": float(belief_position[2]),
            "target_observed_x": float(observed_position[0]),
            "target_observed_y": float(observed_position[1]),
            "target_observed_z": float(observed_position[2]),
            "use_fim": float(self.use_fim),
            "use_belief_state": float(self.use_belief_state),
        }
        game_info = dict(self._last_game_step_info or self._default_game_info())
        predicted_fim_trace_inv = game_info.pop("fim_trace_inv", np.nan)
        game_info["stackelberg_predicted_fim_trace_inv"] = float(predicted_fim_trace_inv)
        info.update(game_info)
        return info

    def _update_sensing(self, initial: bool = False) -> None:
        """Update noisy target observation, belief, and FIM diagnostics."""

        self._require_state()
        true_target = self.state.target_position.copy()

        if self.use_belief_state:
            fused_observation = self._fused_noisy_position_observation(true_target)
            previous_belief = self._belief_target_position()
            if initial:
                belief = fused_observation
            else:
                alpha = float(np.clip(self.belief_update_alpha, 0.0, 1.0))
                belief = alpha * previous_belief + (1.0 - alpha) * fused_observation
            belief = self._clip_target_belief(belief)
        else:
            fused_observation = true_target.copy()
            belief = true_target.copy()

        self.state.observed_target_position = fused_observation.copy()
        self.state.belief_target_position = belief.copy()
        self.current_belief_error = self._distance(true_target, belief)

        if self.use_fim:
            sensor_positions, noise_covariances = self._fim_sensor_inputs(true_target)
            self.current_fim = fisher_information_matrix(true_target, sensor_positions, noise_covariances)
            self.current_fim_metrics = fim_metrics(self.current_fim, regularization=self.fim_regularization)
            logdet = float(self.current_fim_metrics.get("logdet", 0.0))
            self.current_normalized_logdet = float(np.tanh(logdet / 10.0))
        else:
            self.current_fim = np.zeros((3, 3), dtype=float)
            self.current_fim_metrics = self._empty_fim_metrics()
            self.current_normalized_logdet = 0.0

    def _fused_noisy_position_observation(self, true_target: np.ndarray) -> np.ndarray:
        estimates = []
        variances = []

        estimates.append(self._cartesian_position_observation(true_target, self.observation_noise_uav, z_scale=0.5))
        variances.append(max(self.observation_noise_uav**2, 1e-9))

        estimates.append(self._cartesian_position_observation(true_target, self.observation_noise_usv, z_scale=0.35))
        variances.append(max(self.observation_noise_usv**2, 1e-9))

        if self._target_distance() <= self.uuv_observation_range:
            estimates.append(self._uuv_range_bearing_position_observation(true_target))
            variances.append(max(self.observation_noise_uuv**2, 1e-9))

        weights = 1.0 / np.asarray(variances, dtype=float)
        weights = weights / max(float(np.sum(weights)), 1e-12)
        fused = np.zeros(3, dtype=float)
        for weight, estimate in zip(weights, estimates):
            fused += float(weight) * estimate
        return self._clip_target_belief(fused)

    def _cartesian_position_observation(self, true_target: np.ndarray, noise_std: float, z_scale: float) -> np.ndarray:
        std = max(float(noise_std), 0.0)
        noise = self.rng.normal(0.0, std, size=3)
        noise[2] *= float(z_scale)
        observation = true_target + noise
        return self._clip_target_belief(observation)

    def _uuv_range_bearing_position_observation(self, true_target: np.ndarray) -> np.ndarray:
        measurement = range_bearing_measurement(true_target, self.state.uuv_center)
        horizontal_distance = max(float(np.linalg.norm(true_target[:2] - self.state.uuv_center[:2])), 1.0)
        range_noise = self.rng.normal(0.0, self.observation_noise_uuv)
        bearing_noise = self.rng.normal(0.0, self.observation_noise_uuv / horizontal_distance)
        noisy_range = max(0.0, float(measurement[0] + range_noise))
        noisy_bearing = float(measurement[1] + bearing_noise)

        estimate = self.state.uuv_center.copy()
        estimate[0] += noisy_range * np.cos(noisy_bearing)
        estimate[1] += noisy_range * np.sin(noisy_bearing)
        estimate[2] = true_target[2] + self.rng.normal(0.0, self.observation_noise_uuv * 0.25)
        return self._clip_target_belief(estimate)

    def _fim_sensor_inputs(self, true_target: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        sensor_positions = [self.state.uav_position.copy(), self.state.usv_position.copy()]
        noise_covariances = [
            self._range_bearing_covariance(true_target, self.state.uav_position, self.observation_noise_uav),
            self._range_bearing_covariance(true_target, self.state.usv_position, self.observation_noise_usv),
        ]
        if self._target_distance() <= self.uuv_observation_range:
            sensor_positions.append(self.state.uuv_center.copy())
            noise_covariances.append(
                self._range_bearing_covariance(true_target, self.state.uuv_center, self.observation_noise_uuv)
            )
        return sensor_positions, noise_covariances

    def _range_bearing_covariance(self, target: np.ndarray, sensor: np.ndarray, position_noise_std: float) -> np.ndarray:
        std = max(float(position_noise_std), 1e-9)
        horizontal_distance = max(float(np.linalg.norm(target[:2] - sensor[:2])), 1.0)
        bearing_std = std / horizontal_distance
        return np.diag([std**2, bearing_std**2]).astype(float)

    def _information_reward(self) -> float:
        if not self.use_fim:
            return 0.0
        return float(self.info_reward_weight * self.current_normalized_logdet)

    def _empty_fim_metrics(self) -> Dict[str, float]:
        return {
            "logdet": np.nan,
            "trace_inv": np.nan,
            "min_eigenvalue": np.nan,
            "condition_number": np.nan,
        }

    def _policy_target_position(self) -> np.ndarray:
        if self.use_belief_state:
            return self._belief_target_position()
        return self.state.target_position

    def _belief_target_position(self) -> np.ndarray:
        if self.state.belief_target_position is None:
            return self.state.target_position
        return self.state.belief_target_position

    def _observed_target_position(self) -> np.ndarray:
        if self.state.observed_target_position is None:
            return self._belief_target_position()
        return self.state.observed_target_position

    def _clip_target_belief(self, position: np.ndarray) -> np.ndarray:
        clipped = self._clip_position(np.asarray(position, dtype=float).copy())
        clipped[2] = np.clip(clipped[2], -self.max_depth, 0.0)
        return clipped

    def _select_executed_action(self, action: int) -> Dict[str, float]:
        action = int(action)
        game_selected_action = action
        game_info = self._default_game_info(action)

        if self.use_stackelberg:
            from game.stackelberg import stackelberg_select_action

            game_selected_action, game_info = stackelberg_select_action(
                self,
                None,
                action,
                self.config,
                return_info=True,
            )
            game_selected_action = int(game_selected_action)

        if not self.use_lyapunov:
            game_info.update(
                {
                    "original_action": float(action),
                    "executed_action": float(game_selected_action),
                    "action_replaced": float(action != game_selected_action),
                    "proposed_was_safe": 1.0,
                    "safety_filter_active": 0.0,
                    "lyapunov_changed_action": 0.0,
                }
            )
            return game_info

        from safety.lyapunov import select_safe_action

        executed_action, safety_info = select_safe_action(
            self, game_selected_action, range(self.action_space_n), self.config
        )
        safety_info = dict(safety_info)
        safety_info.update(game_info)
        safety_info["original_action"] = float(action)
        safety_info["executed_action"] = float(executed_action)
        safety_info["safety_filter_active"] = 1.0
        safety_info["action_replaced"] = float(int(executed_action) != action)
        safety_info["lyapunov_changed_action"] = float(int(executed_action) != game_selected_action)
        return safety_info

    def _select_target_response(self, executed_action: int, selection_info: Dict[str, float]) -> Dict[str, float]:
        if not self.use_intelligent_target:
            return self._default_game_info(executed_action)

        from game.stackelberg import target_best_response_details

        response = target_best_response_details(self, executed_action, self.config)
        target_info = self._default_game_info(executed_action)
        target_info.update(
            {
                "target_motion_model": "intelligent",
                "target_best_response_action": float(response["target_action"]),
                "target_utility": float(response["target_utility"]),
                "target_uncertainty_score": float(response.get("uncertainty_score", 0.0)),
                "target_weak_connectivity_score": float(response.get("weak_connectivity_score", 0.0)),
                "target_boundary_penalty": float(response.get("target_boundary_penalty", 0.0)),
                "fim_trace_inv": float(response.get("fim_trace_inv", self.current_fim_metrics.get("trace_inv", np.nan))),
            }
        )
        for key in (
            "stackelberg_active",
            "stackelberg_proposed_action",
            "stackelberg_selected_action",
            "stackelberg_changed_action",
            "stackelberg_leader_cost",
            "stackelberg_energy_cost",
            "stackelberg_connectivity_cost",
            "stackelberg_information_cost",
            "stackelberg_lyapunov_penalty",
            "stackelberg_evaluated_actions",
        ):
            if key in selection_info:
                target_info[key] = float(selection_info[key])
        return target_info

    def _merge_game_step_info(
        self,
        selection_info: Dict[str, float],
        target_decision: Dict[str, float],
    ) -> Dict[str, float]:
        executed_action = int(selection_info.get("executed_action", target_decision.get("stackelberg_selected_action", 0)))
        merged = self._default_game_info(executed_action)
        allowed_keys = set(merged) | {"target_motion_model"}
        merged.update({key: value for key, value in selection_info.items() if key in allowed_keys})
        merged.update({key: value for key, value in target_decision.items() if key in allowed_keys})
        if "fim_trace_inv" not in merged or not np.isfinite(float(merged.get("fim_trace_inv", np.nan))):
            merged["fim_trace_inv"] = float(self.current_fim_metrics.get("trace_inv", np.nan))
        return merged

    def _default_game_info(self, action: int = -1) -> Dict[str, float]:
        return {
            "stackelberg_active": float(self.use_stackelberg),
            "stackelberg_proposed_action": float(action),
            "stackelberg_selected_action": float(action),
            "stackelberg_changed_action": 0.0,
            "stackelberg_leader_cost": 0.0,
            "target_best_response_action": -1.0,
            "target_utility": 0.0,
            "target_uncertainty_score": 0.0,
            "target_weak_connectivity_score": 0.0,
            "target_boundary_penalty": 0.0,
            "stackelberg_energy_cost": 0.0,
            "stackelberg_connectivity_cost": 0.0,
            "stackelberg_information_cost": 0.0,
            "stackelberg_lyapunov_penalty": 0.0,
            "stackelberg_evaluated_actions": 0.0,
            "fim_trace_inv": float(self.current_fim_metrics.get("trace_inv", np.nan)),
        }

    def _attach_reset_safety_info(self) -> None:
        from safety.lyapunov import lyapunov_value

        value = lyapunov_value(self.get_state(), self.last_info, self.config)
        self.last_info.update(
            {
                "lyapunov_value": float(value),
                "lyapunov_delta": 0.0,
                "lyapunov_condition_satisfied": 1.0,
                "safety_violation": 0.0,
                "safety_filter_active": float(self.use_lyapunov),
                "original_action": -1.0,
                "executed_action": -1.0,
                "action_replaced": 0.0,
            }
        )

    def _attach_step_safety_info(
        self,
        info: Dict[str, float],
        previous_state: np.ndarray,
        previous_metrics: Dict[str, float],
        original_action: int,
        executed_action: int,
        selection_info: Dict[str, float],
    ) -> None:
        from safety.lyapunov import is_safe_transition, lyapunov_delta, lyapunov_value

        current_state = self.get_state()
        value = lyapunov_value(current_state, info, self.config)
        delta = lyapunov_delta(previous_state, current_state, previous_metrics, info, self.config)
        condition_satisfied = is_safe_transition(previous_state, current_state, previous_metrics, info, self.config)
        safety_violation = float(not condition_satisfied)
        action_replaced = float(original_action != executed_action)

        info.update(
            {
                "lyapunov_value": float(value),
                "lyapunov_delta": float(delta),
                "lyapunov_condition_satisfied": float(condition_satisfied),
                "safety_violation": safety_violation,
                "safety_filter_active": float(self.use_lyapunov),
                "original_action": float(original_action),
                "executed_action": float(executed_action),
                "action_replaced": action_replaced,
                "proposed_was_safe": float(selection_info.get("proposed_was_safe", condition_satisfied)),
                "predicted_lyapunov_margin": float(selection_info.get("lyapunov_margin", np.nan)),
                "predicted_lyapunov_value": float(selection_info.get("lyapunov_value", np.nan)),
            }
        )

        self.history["lyapunov_value"].append(float(value))
        self.history["lyapunov_delta"].append(float(delta))
        self.history["lyapunov_condition_satisfied"].append(float(condition_satisfied))
        self.history["safety_violation"].append(safety_violation)
        self.history["original_action"].append(float(original_action))
        self.history["executed_action"].append(float(executed_action))
        self.history["action_replaced"].append(action_replaced)

    def _target_captured(self) -> bool:
        return bool(self._target_distance() <= self.search_radius)

    def _target_escaped(self) -> bool:
        return bool(self.state.target_escaped)

    def _target_distance(self) -> float:
        return self._distance(self.state.uuv_center, self.state.target_position)

    def _us_distance(self) -> float:
        return self._distance(self.state.uav_position, self.state.usv_position)

    def _sg_distance(self) -> float:
        return self._distance(self.state.usv_position, self.state.uuv_center)

    def _clip_position(self, position: np.ndarray) -> np.ndarray:
        clipped = np.asarray(position, dtype=float).copy()
        clipped[0] = np.clip(clipped[0], 0.0, self.area_size)
        clipped[1] = np.clip(clipped[1], 0.0, self.area_size)
        return clipped

    def _xy_in_bounds(self, position: np.ndarray) -> bool:
        return bool(0.0 <= position[0] <= self.area_size and 0.0 <= position[1] <= self.area_size)

    def _unit_vector(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=float)
        norm = np.linalg.norm(vector)
        if norm <= 1e-12:
            return np.zeros_like(vector, dtype=float)
        return vector / norm

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))

    def _require_state(self) -> None:
        if self.state is None:
            raise RuntimeError("Call reset() before using the environment.")
