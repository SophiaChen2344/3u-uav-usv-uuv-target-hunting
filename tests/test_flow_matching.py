from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from envs.three_u_env import ThreeUEnv
from generative.flow_matching import (
    FlowMatchingPlanner,
    _differentiable_trajectory_information_gain,
    sample_trajectories,
    train_flow_matching,
)
from generative.trajectory_dataset import generate_heuristic_trajectories


def load_flow_test_config() -> dict:
    with open(PROJECT_ROOT / "configs" / "default.yaml", "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["environment"]["max_steps"] = 6
    config["training"]["progress_bar"] = False
    config["flow_matching"].update(
        {
            "horizon": 4,
            "num_candidates": 4,
            "hidden_dim": 32,
            "num_layers": 2,
            "batch_size": 8,
            "epochs": 1,
            "dataset_size": 24,
            "use_dqn_rollouts": False,
            "use_heuristic_rollouts": True,
            "use_aco_rollouts": False,
        }
    )
    return config


def test_flow_matching_training_and_sampling_shape() -> None:
    config = load_flow_test_config()
    env = ThreeUEnv(config, seed=321)
    dataset = generate_heuristic_trajectories(env, n_samples=24, horizon=4)

    model, history = train_flow_matching(dataset, config)
    assert len(history) == 1
    assert np.isfinite(history["loss"].iloc[-1])
    assert np.isfinite(history["information_gain"].iloc[-1])
    assert np.isfinite(history["speed_loss"].iloc[-1])
    assert np.isfinite(history["step_loss"].iloc[-1])
    assert np.isfinite(history["smoothness_loss"].iloc[-1])
    assert np.isfinite(history["boundary_loss"].iloc[-1])
    assert np.isfinite(history["connectivity_loss"].iloc[-1])
    assert np.isfinite(history["energy_safety_loss"].iloc[-1])

    env.reset()
    condition = env.get_flow_condition_vector(coarse_action=env.greedy_action_toward_target())
    assert condition.shape == (39,)
    samples = sample_trajectories(model, condition, num_samples=3, horizon=4, env=env, config=config)

    assert samples.shape == (3, 4, 3)
    assert np.all(samples[:, :, 0] >= 0.0)
    assert np.all(samples[:, :, 0] <= env.area_size)
    assert np.all(samples[:, :, 1] >= 0.0)
    assert np.all(samples[:, :, 1] <= env.area_size)
    assert np.allclose(samples[:, :, 2], env.uuv_initial_depth)


def test_differentiable_fim_uses_heterogeneous_sensors_with_smooth_range() -> None:
    config = load_flow_test_config()
    config["sensing"]["uuv_observation_range"] = 1.0
    config["sensing"]["observation_range_smoothing"] = 5.0
    env = ThreeUEnv(config, seed=432)
    env.reset()
    condition = env.get_flow_condition_vector(coarse_action=env.greedy_action_toward_target())
    trajectory = np.repeat(env.state.uuv_center[None, :], repeats=4, axis=0).astype(np.float32)

    gain = _differentiable_trajectory_information_gain(
        torch.as_tensor(trajectory[None, :, :], dtype=torch.float32),
        torch.as_tensor(condition[None, :], dtype=torch.float32),
        config=config,
    )

    assert gain.shape == (1,)
    assert torch.isfinite(gain).all()
    assert float(gain.item()) > 0.0


def test_differentiable_fim_uses_predicted_target_response_path() -> None:
    config = load_flow_test_config()
    config["sensing"]["uav_observation_range"] = 1000.0
    config["sensing"]["usv_observation_range"] = 1000.0
    config["sensing"]["uuv_observation_range"] = 1000.0
    trajectory = np.repeat(np.array([[180.0, 190.0, -120.0]], dtype=np.float32), repeats=4, axis=0)

    condition = np.zeros((39,), dtype=np.float32)
    condition[0:3] = np.array([50.0, 50.0, 120.0], dtype=np.float32)
    condition[3:6] = np.array([70.0, 330.0, 0.0], dtype=np.float32)
    condition[6:9] = trajectory[0]
    condition[9:12] = np.array([210.0, 210.0, -120.0], dtype=np.float32)
    condition[12:18] = np.array([100.0, 80.0, 25.0, 10.0, 0.0, 0.0], dtype=np.float32)

    static_condition = condition.copy()
    escaping_condition = condition.copy()
    escaping_condition[33:36] = np.array([15.0, -10.0, 0.0], dtype=np.float32)

    static_gain = _differentiable_trajectory_information_gain(
        torch.as_tensor(trajectory[None, :, :], dtype=torch.float32),
        torch.as_tensor(static_condition[None, :], dtype=torch.float32),
        config=config,
    )
    escaping_gain = _differentiable_trajectory_information_gain(
        torch.as_tensor(trajectory[None, :, :], dtype=torch.float32),
        torch.as_tensor(escaping_condition[None, :], dtype=torch.float32),
        config=config,
    )

    assert torch.isfinite(static_gain).all()
    assert torch.isfinite(escaping_gain).all()
    assert not torch.allclose(static_gain, escaping_gain)


def test_full_flow_matching_planner_runs_five_steps() -> None:
    config = load_flow_test_config()
    env = ThreeUEnv(config, seed=654)
    state = env.reset()
    planner = FlowMatchingPlanner(model=None, config=config, mode="full")

    for _ in range(5):
        action = planner.select_action(env)
        assert planner.last_trajectories is not None
        assert planner.last_trajectories.shape[0] == 1
        assert planner.last_score is not None
        assert planner.last_score.information_cost == 0.0
        state, reward, done, info = env.step(action)
        assert state.shape == (16,)
        assert isinstance(reward, float)
        assert 0 <= action < env.action_space_n
        assert np.isfinite(info["total_energy_used"])
        if done:
            break
