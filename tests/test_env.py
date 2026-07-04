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


def load_test_config() -> dict:
    with open(PROJECT_ROOT / "configs" / "default.yaml", "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["environment"]["max_steps"] = 5
    return config


def test_environment_reset_observation_is_finite() -> None:
    env = ThreeUEnv(load_test_config(), seed=123)
    observation = env.reset()
    assert observation.ndim == 1
    assert observation.size == 16
    assert np.all(np.isfinite(observation))
    assert env.action_space_n == 8
    assert env.check_constraints()
    assert np.allclose(env.state.uav_position, np.array([200.0, 200.0, 120.0]))
    assert np.allclose(env.state.uuv_center, np.array([200.0, 200.0, -120.0]))


def test_environment_step_updates_energy_and_info() -> None:
    env = ThreeUEnv(load_test_config(), seed=123)
    observation = env.reset()
    next_observation, reward, done, info = env.step(env.greedy_action_toward_target())

    assert next_observation.shape == observation.shape
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert info["total_energy_used"] > 0.0
    assert env.state.total_voyage_distance > 0.0
    assert len(env.history["energy"]) == 1
    assert len(env.history["us_distance"]) == 1
    assert len(env.history["sg_distance"]) == 1
    assert len(env.history["target_distance"]) == 2
    assert 0.0 <= info["connected_fraction"] <= 1.0
