"""Synthetic trajectory dataset generation for Flow Matching.

The paper does not release expert UUV trajectory data, so this module builds a
small supervised dataset from the simulator itself: DQN-like policy rollouts
when a policy is supplied, ACO waypoints when enabled, and heuristic pursuit
trajectories filtered through the environment's projection and safety logic.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from agents.aco_baseline import ACOBaselinePolicy, ACOPlanner
from envs.three_u_env import ThreeUEnv
from generative.flow_matching import heuristic_candidate_trajectories


def collect_rollouts(
    env: ThreeUEnv,
    policy: Any,
    n_episodes: int,
    horizon: int | None = None,
) -> Dict[str, np.ndarray]:
    """Collect condition/trajectory pairs from a policy in the environment.

    ``policy`` may be a DQN-style object with ``select_action(state)`` or an
    ACO-style policy with ``select_action(env)``.
    """

    horizon = int(horizon or env.config.get("flow_matching", {}).get("horizon", 10))
    conditions: list[np.ndarray] = []
    trajectories: list[np.ndarray] = []
    sources: list[str] = []

    for _episode in range(int(n_episodes)):
        state = env.reset()
        if hasattr(policy, "reset"):
            policy.reset(env)
        for _step in range(env.max_steps):
            action = _policy_action(policy, env, state)
            condition = env.get_flow_condition_vector(coarse_action=action)
            future = _rollout_future(env, policy, action, horizon)
            conditions.append(condition)
            trajectories.append(future)
            sources.append("policy")

            state, _reward, done, _info = env.step(action)
            if done:
                break

    return _pack_dataset(conditions, trajectories, sources)


def generate_heuristic_trajectories(
    env: ThreeUEnv,
    n_samples: int,
    horizon: int | None = None,
) -> Dict[str, np.ndarray]:
    """Generate pursuit trajectories with small lateral and coarse-action bias."""

    horizon = int(horizon or env.config.get("flow_matching", {}).get("horizon", 10))
    conditions: list[np.ndarray] = []
    trajectories: list[np.ndarray] = []
    sources: list[str] = []
    remaining = int(n_samples)

    while remaining > 0:
        env.reset()
        for _step in range(env.max_steps):
            coarse_action = env.greedy_action_toward_target()
            condition = env.get_flow_condition_vector(coarse_action=coarse_action)
            batch_size = min(remaining, 8)
            batch = heuristic_candidate_trajectories(env, batch_size, horizon, coarse_action=coarse_action)
            for trajectory in batch:
                conditions.append(condition)
                trajectories.append(trajectory)
                sources.append("heuristic")
            remaining -= batch_size
            if remaining <= 0:
                break
            _state, _reward, done, _info = env.step(coarse_action)
            if done:
                break

    return _pack_dataset(conditions, trajectories, sources)


def generate_aco_trajectories(
    env: ThreeUEnv,
    n_samples: int,
    horizon: int | None = None,
    config: Dict[str, Any] | None = None,
) -> Dict[str, np.ndarray]:
    """Generate short trajectory labels by following an ACO waypoint policy."""

    config = config or env.config
    horizon = int(horizon or config.get("flow_matching", {}).get("horizon", 10))
    aco_cfg = dict(config.get("experiments", {}).get("aco", {}))
    planner = ACOPlanner(
        area_size=env.area_size,
        grid_size=int(aco_cfg.get("grid_size", 25)),
        ants=int(aco_cfg.get("ants", 100)),
        iterations=int(aco_cfg.get("iterations", 100)),
        evaporation=float(aco_cfg.get("evaporation", 0.2)),
        alpha=float(aco_cfg.get("alpha", 1.0)),
        beta=float(aco_cfg.get("beta", 2.5)),
        q=float(aco_cfg.get("q", 80.0)),
        seed=int(config.get("seed", 7)) + 30_000,
    )
    policy = ACOBaselinePolicy(planner, replan_interval=int(aco_cfg.get("replan_interval", 10)))
    rollout_data = collect_rollouts(env, policy, n_episodes=max(1, int(np.ceil(n_samples / max(env.max_steps, 1)))), horizon=horizon)
    return {key: value[:n_samples] for key, value in rollout_data.items()}


def build_trajectory_dataset(
    config: Dict[str, Any],
    env: ThreeUEnv | None = None,
    policy: Any | None = None,
    save_path: str | Path | None = None,
) -> Dict[str, np.ndarray]:
    """Create and optionally save a mixed synthetic trajectory dataset."""

    flow_cfg = dict(config.get("flow_matching", {}))
    env = env or ThreeUEnv(config, seed=int(config.get("seed", 7)) + 20_000)
    horizon = int(flow_cfg.get("horizon", 10))
    dataset_size = int(flow_cfg.get("dataset_size", 5000))
    chunks: list[Dict[str, np.ndarray]] = []

    if bool(flow_cfg.get("use_dqn_rollouts", True)) and policy is not None:
        rollout_samples = max(1, dataset_size // 4)
        episodes = max(1, int(np.ceil(rollout_samples / max(env.max_steps, 1))))
        chunks.append(collect_rollouts(env, policy, n_episodes=episodes, horizon=horizon))

    if bool(flow_cfg.get("use_aco_rollouts", False)):
        chunks.append(generate_aco_trajectories(env, max(1, dataset_size // 4), horizon=horizon, config=config))

    if bool(flow_cfg.get("use_heuristic_rollouts", True)) or not chunks:
        heuristic_samples = dataset_size - sum(chunk["conditions"].shape[0] for chunk in chunks)
        heuristic_samples = max(1, heuristic_samples)
        chunks.append(generate_heuristic_trajectories(env, heuristic_samples, horizon=horizon))

    dataset = _concat_chunks(chunks, dataset_size)
    if save_path is None:
        save_path = config.get("results", {}).get("datasets", "results/datasets")
        save_path = Path(save_path) / "trajectory_dataset.npz"
    save_dataset(dataset, save_path)
    return dataset


def save_dataset(dataset: Dict[str, np.ndarray], path: str | Path) -> None:
    """Save the dataset in compressed ``npz`` format."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **dataset)


def load_dataset(path: str | Path) -> Dict[str, np.ndarray]:
    """Load a trajectory dataset saved by :func:`save_dataset`."""

    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _rollout_future(env: ThreeUEnv, policy: Any, first_action: int, horizon: int) -> np.ndarray:
    trial_env = deepcopy(env)
    positions = []
    action = int(first_action)
    state = trial_env.get_state()
    for step in range(int(horizon)):
        if step > 0:
            action = _policy_action(policy, trial_env, state)
        state, _reward, done, _info = trial_env.step(action)
        positions.append(trial_env.state.uuv_center.copy())
        if done:
            positions.extend([trial_env.state.uuv_center.copy()] * (int(horizon) - len(positions)))
            break
    return trial_env.project_trajectory(np.asarray(positions, dtype=float))


def _policy_action(policy: Any, env: ThreeUEnv, state: np.ndarray) -> int:
    if policy is None:
        return env.greedy_action_toward_target()
    if isinstance(policy, ACOBaselinePolicy):
        return int(policy.select_action(env))
    try:
        return int(policy.select_action(state, epsilon=0.0))
    except TypeError:
        try:
            return int(policy.select_action(state))
        except TypeError:
            return int(policy.select_action(env))


def _pack_dataset(
    conditions: list[np.ndarray],
    trajectories: list[np.ndarray],
    sources: list[str],
) -> Dict[str, np.ndarray]:
    return {
        "conditions": np.asarray(conditions, dtype=np.float32),
        "trajectories": np.asarray(trajectories, dtype=np.float32),
        "sources": np.asarray(sources),
    }


def _concat_chunks(chunks: list[Dict[str, np.ndarray]], dataset_size: int) -> Dict[str, np.ndarray]:
    conditions = np.concatenate([chunk["conditions"] for chunk in chunks if chunk["conditions"].size], axis=0)
    trajectories = np.concatenate([chunk["trajectories"] for chunk in chunks if chunk["trajectories"].size], axis=0)
    sources = np.concatenate([chunk["sources"] for chunk in chunks if chunk["sources"].size], axis=0)
    if conditions.shape[0] > int(dataset_size):
        indices = np.arange(conditions.shape[0])
        np.random.default_rng(12345).shuffle(indices)
        indices = indices[: int(dataset_size)]
        conditions = conditions[indices]
        trajectories = trajectories[indices]
        sources = sources[indices]
    return {"conditions": conditions, "trajectories": trajectories, "sources": sources}
