"""NumPy-only UAV-USV-UUV cooperative target hunting environment.

This environment follows a compact abstraction of the 3U hunting problem:

- one UAV monitors the 400 m x 400 m area from altitude ``h``;
- one USV performs a random surface relay walk at ``z = 0``;
- three UUVs are represented by a single underwater group center ``G``;
- the target stays at the hunting depth and moves away from ``G``.

The API is intentionally Gym-like but does not depend on Gym/Gymnasium.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


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
    target_escaped: bool = False
    constraint_violation: bool = False

    @property
    def target_estimate(self) -> np.ndarray:
        """Compatibility alias for algorithms that expect an estimated target."""

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

        self.state: ThreeUState | None = None
        self.last_info: Dict[str, float] = {}
        self.history: Dict[str, list[float]] = {}
        self.action_tuples = [(action,) for action in range(self.action_space_n)]
        self._last_boundary_violation = False

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
        )
        self.history = {
            "energy": [],
            "us_distance": [],
            "sg_distance": [],
            "target_distance": [target_distance],
        }
        self._last_boundary_violation = False
        self.last_info = self._build_info(reward=0.0, done=False)
        return self.get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        """Advance one step with an 8-direction UUV group-center action."""

        self._require_state()
        action = int(np.clip(action, 0, self.action_space_n - 1))

        previous_center = self.state.uuv_center.copy()
        self.state.previous_target_distance = self._target_distance()
        self._last_boundary_violation = False

        self._move_uuv_group(action)
        actual_move = self._distance(previous_center, self.state.uuv_center)
        self.state.total_voyage_distance += actual_move

        self._move_usv_randomly()
        self._move_target_away_from_group()
        self.state.direction_gw = self._unit_vector(self.state.target_position - self.state.uuv_center)

        step_energy = self._compute_step_energy(actual_move)
        self.state.total_energy_used += step_energy

        constraints_ok = self.check_constraints()
        self.state.constraint_violation = not constraints_ok
        captured = self._target_captured()
        escaped = self._target_escaped()
        reward = self.compute_reward()

        self.state.step_count += 1
        done = bool(captured or escaped or self.state.step_count >= self.max_steps)

        self._append_logs(step_energy)
        info = self._build_info(reward=reward, done=done)
        self.last_info = info
        return self.get_state(), reward, done, info

    def get_state(self) -> np.ndarray:
        """Return ``[U, S, G, W, direction_GW, L]`` as a float32 vector."""

        self._require_state()
        return np.concatenate(
            [
                self.state.uav_position,
                self.state.usv_position,
                self.state.uuv_center,
                self.state.target_position,
                self.state.direction_gw,
                np.array([self.state.total_voyage_distance], dtype=float),
            ]
        ).astype(np.float32)

    def compute_reward(self) -> float:
        """Compute the paper-style sparse hunting reward."""

        self._require_state()
        if self.state.constraint_violation or self.state.target_escaped:
            return -1.0
        if self._target_captured():
            return 10.0
        if self._target_distance() < self.state.previous_target_distance:
            return 0.1
        return -1.0

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
            f"d_GW={self._target_distance():.1f} L={self.state.total_voyage_distance:.1f}"
        )
        if mode == "human":
            print(summary)
        return summary

    def greedy_action_toward_target(self) -> int:
        """Return the compass action whose direction best points from G to W."""

        self._require_state()
        return self.action_from_vector(self.state.target_position[:2] - self.state.uuv_center[:2])

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

    def _compute_step_energy(self, actual_move: float) -> float:
        speed = actual_move / max(self.dt, 1e-12)
        per_uuv = self.energy_base * self.dt + self.energy_linear * actual_move + self.energy_quadratic * speed**2
        return float(self.num_uuvs * per_uuv)

    def _append_logs(self, step_energy: float) -> None:
        self.history["energy"].append(float(step_energy))
        self.history["us_distance"].append(self._us_distance())
        self.history["sg_distance"].append(self._sg_distance())
        self.history["target_distance"].append(self._target_distance())

    def _build_info(self, reward: float, done: bool) -> Dict[str, float]:
        captured = self._target_captured()
        escaped = self._target_escaped()
        us_distance = self._us_distance()
        sg_distance = self._sg_distance()
        target_distance = self._target_distance()
        connected_fraction = 0.5 * float(us_distance <= self.uav_usv_range) + 0.5 * float(
            sg_distance <= self.usv_uuv_range
        )
        return {
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
        }

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
