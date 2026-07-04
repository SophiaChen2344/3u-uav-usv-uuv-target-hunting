"""Ant Colony Optimization baseline for UUV group-center path planning.

This is an approximate reproduction baseline. The paper describes an ACO
comparison but does not provide source code, so this implementation uses a
standard grid-based pheromone and distance-heuristic planner.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


GridPoint = Tuple[int, int]


@dataclass
class ACOPathResult:
    path: List[GridPoint]
    length: float


class ACOPlanner:
    """Grid-based ACO planner.

    The planner operates on a 2D surface projection of the underwater problem.
    It gives the baseline a classical shortest-path flavor while the simulator
    still accounts for 3D motion, energy, target movement, and connectivity.
    """

    def __init__(
        self,
        area_size: float,
        grid_size: int = 25,
        ants: int = 100,
        iterations: int = 100,
        evaporation: float = 0.2,
        alpha: float = 1.0,
        beta: float = 2.5,
        q: float = 80.0,
        seed: int | None = None,
    ) -> None:
        self.area_size = float(area_size)
        self.grid_size = int(grid_size)
        self.ants = int(ants)
        self.iterations = int(iterations)
        self.evaporation = float(evaporation)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.q = float(q)
        self.rng = np.random.default_rng(seed)
        self.pheromone = np.ones((self.grid_size, self.grid_size), dtype=float)

    def plan(self, start_xy: np.ndarray, goal_xy: np.ndarray) -> ACOPathResult:
        start = self._xy_to_grid(start_xy)
        goal = self._xy_to_grid(goal_xy)
        best_path: List[GridPoint] = self._fallback_astar(start, goal)
        best_length = self._path_length(best_path)

        for _ in range(self.iterations):
            paths = []
            for _ant in range(self.ants):
                path = self._construct_path(start, goal)
                length = self._path_length(path)
                paths.append((path, length))
                if length < best_length:
                    best_path, best_length = path, length

            self.pheromone *= 1.0 - self.evaporation
            for path, length in paths:
                deposit = self.q / max(length, 1.0)
                for point in path:
                    self.pheromone[point] += deposit
            for point in best_path:
                self.pheromone[point] += self.q / max(best_length, 1.0)

        return ACOPathResult(path=best_path, length=best_length)

    def path_to_waypoints(self, path: Sequence[GridPoint]) -> np.ndarray:
        return np.asarray([self._grid_to_xy(point) for point in path], dtype=float)

    def _construct_path(self, start: GridPoint, goal: GridPoint) -> List[GridPoint]:
        current = start
        path = [current]
        visited = {current}
        max_hops = self.grid_size * self.grid_size

        for _ in range(max_hops):
            if current == goal:
                return path

            candidates = [p for p in self._neighbors(current) if p not in visited]
            if not candidates:
                fallback = self._fallback_astar(current, goal)
                return path + fallback[1:]

            probabilities = self._transition_probabilities(candidates, goal)
            next_idx = int(self.rng.choice(len(candidates), p=probabilities))
            current = candidates[next_idx]
            visited.add(current)
            path.append(current)

        fallback = self._fallback_astar(path[-1], goal)
        return path + fallback[1:]

    def _transition_probabilities(self, candidates: Sequence[GridPoint], goal: GridPoint) -> np.ndarray:
        weights = []
        for point in candidates:
            pheromone = self.pheromone[point] ** self.alpha
            heuristic = (1.0 / (self._grid_distance(point, goal) + 1.0)) ** self.beta
            weights.append(pheromone * heuristic)
        weights_arr = np.asarray(weights, dtype=float)
        total = float(weights_arr.sum())
        if total <= 0.0:
            return np.full(len(candidates), 1.0 / len(candidates), dtype=float)
        return weights_arr / total

    def _neighbors(self, point: GridPoint) -> Iterable[GridPoint]:
        x, y = point
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                yield (nx, ny)

    def _fallback_astar(self, start: GridPoint, goal: GridPoint) -> List[GridPoint]:
        frontier: List[Tuple[float, GridPoint]] = []
        heapq.heappush(frontier, (0.0, start))
        came_from: Dict[GridPoint, GridPoint | None] = {start: None}
        cost_so_far = {start: 0.0}

        while frontier:
            _priority, current = heapq.heappop(frontier)
            if current == goal:
                break
            for next_point in self._neighbors(current):
                new_cost = cost_so_far[current] + self._grid_distance(current, next_point)
                if next_point not in cost_so_far or new_cost < cost_so_far[next_point]:
                    cost_so_far[next_point] = new_cost
                    priority = new_cost + self._grid_distance(next_point, goal)
                    heapq.heappush(frontier, (priority, next_point))
                    came_from[next_point] = current

        if goal not in came_from:
            return [start]

        path = [goal]
        while path[-1] != start:
            parent = came_from[path[-1]]
            if parent is None:
                break
            path.append(parent)
        path.reverse()
        return path

    def _path_length(self, path: Sequence[GridPoint]) -> float:
        if len(path) < 2:
            return 0.0
        cell = self.area_size / max(self.grid_size - 1, 1)
        return sum(self._grid_distance(a, b) * cell for a, b in zip(path[:-1], path[1:]))

    def _grid_distance(self, a: GridPoint, b: GridPoint) -> float:
        return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))

    def _xy_to_grid(self, xy: np.ndarray) -> GridPoint:
        xy = np.asarray(xy, dtype=float)
        scale = max(self.grid_size - 1, 1) / self.area_size
        x = int(np.clip(round(xy[0] * scale), 0, self.grid_size - 1))
        y = int(np.clip(round(xy[1] * scale), 0, self.grid_size - 1))
        return x, y

    def _grid_to_xy(self, point: GridPoint) -> np.ndarray:
        cell = self.area_size / max(self.grid_size - 1, 1)
        return np.array([point[0] * cell, point[1] * cell], dtype=float)


class ACOBaselinePolicy:
    """Convert ACO waypoints into UUV group-center actions."""

    def __init__(self, planner: ACOPlanner, replan_interval: int = 10) -> None:
        self.planner = planner
        self.replan_interval = int(replan_interval)
        self.waypoints: np.ndarray = np.empty((0, 2), dtype=float)
        self.index = 0
        self.last_plan_step = -1

    def reset(self, env) -> None:
        self.index = 0
        self.last_plan_step = -1
        self._plan_from_env(env)

    def select_action(self, env) -> int:
        if self._should_replan(env):
            self._plan_from_env(env)

        waypoint = self._current_waypoint(env.state.uuv_center)
        delta = waypoint - env.state.uuv_center[:2]
        if np.linalg.norm(delta) < float(env.env_config.get("capture_radius", 18.0)) * 0.45:
            self.index = min(self.index + 1, len(self.waypoints) - 1)
            waypoint = self._current_waypoint(env.state.uuv_center)
            delta = waypoint - env.state.uuv_center[:2]
        if np.linalg.norm(delta) <= 1e-9:
            delta = env.state.target_position[:2] - env.state.uuv_center[:2]
        return env.action_from_vector(delta)

    def _should_replan(self, env) -> bool:
        if self.waypoints.size == 0:
            return True
        if self.replan_interval <= 0:
            return False
        return (int(env.state.step_count) - self.last_plan_step) >= self.replan_interval

    def _plan_from_env(self, env) -> None:
        result = self.planner.plan(env.state.uuv_center[:2], env.state.target_position[:2])
        waypoints = self.planner.path_to_waypoints(result.path)
        if len(waypoints) == 0:
            waypoints = np.asarray([env.state.target_position[:2]], dtype=float)
        self.waypoints = waypoints
        self.index = min(1, len(self.waypoints) - 1)
        self.last_plan_step = int(env.state.step_count)

    def _current_waypoint(self, position: np.ndarray) -> np.ndarray:
        if self.waypoints.size == 0:
            return position[:2]
        return self.waypoints[self.index]
