from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from envs.three_u_env import ThreeUEnv
from generative.flow_matching import FlowMatchingPlanner, sample_trajectories, train_flow_matching
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

    env.reset()
    condition = env.get_flow_condition_vector(coarse_action=env.greedy_action_toward_target())
    samples = sample_trajectories(model, condition, num_samples=3, horizon=4, env=env, config=config)

    assert samples.shape == (3, 4, 3)
    assert np.all(samples[:, :, 0] >= 0.0)
    assert np.all(samples[:, :, 0] <= env.area_size)
    assert np.all(samples[:, :, 1] >= 0.0)
    assert np.all(samples[:, :, 1] <= env.area_size)
    assert np.allclose(samples[:, :, 2], env.uuv_initial_depth)


def test_full_flow_matching_planner_runs_five_steps() -> None:
    config = load_flow_test_config()
    env = ThreeUEnv(config, seed=654)
    state = env.reset()
    planner = FlowMatchingPlanner(model=None, config=config, mode="full")

    for _ in range(5):
        action = planner.select_action(env)
        state, reward, done, info = env.step(action)
        assert state.shape == (16,)
        assert isinstance(reward, float)
        assert 0 <= action < env.action_space_n
        assert np.isfinite(info["total_energy_used"])
        if done:
            break
