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


def test_environment_runs_with_fim_disabled_and_enabled() -> None:
    disabled = load_test_config()
    disabled["sensing"]["use_fim"] = False
    disabled["sensing"]["use_belief_state"] = False
    env_disabled = ThreeUEnv(disabled, seed=123)
    obs_disabled = env_disabled.reset()
    next_obs_disabled, _reward, _done, info_disabled = env_disabled.step(0)

    enabled = load_test_config()
    enabled["sensing"]["use_fim"] = True
    enabled["sensing"]["use_belief_state"] = True
    env_enabled = ThreeUEnv(enabled, seed=123)
    obs_enabled = env_enabled.reset()
    next_obs_enabled, _reward, _done, info_enabled = env_enabled.step(0)

    assert obs_disabled.shape == obs_enabled.shape == (16,)
    assert next_obs_disabled.shape == next_obs_enabled.shape == (16,)
    assert np.isnan(info_disabled["fim_logdet"])
    assert np.isfinite(info_enabled["fim_logdet"])
    assert np.isfinite(info_enabled["belief_error"])
    assert info_enabled["target_belief_covariance"].shape == (3, 3)
    assert np.all(np.linalg.eigvalsh(info_enabled["target_belief_covariance"]) > 0.0)


def test_belief_state_hides_true_target_position_from_policy_state() -> None:
    config = load_test_config()
    config["sensing"]["use_fim"] = True
    config["sensing"]["use_belief_state"] = True
    config["sensing"]["observation_noise_uav"] = 30.0
    config["sensing"]["observation_noise_usv"] = 25.0
    config["sensing"]["observation_noise_uuv"] = 20.0

    env = ThreeUEnv(config, seed=999)
    observation = env.reset()

    policy_target = observation[9:12]
    true_target = env.state.target_position.astype(np.float32)
    belief_target = env.state.belief_target_position.astype(np.float32)
    condition = env.get_flow_condition_vector(coarse_action=env.greedy_action_toward_target())

    assert np.allclose(policy_target, belief_target)
    assert not np.allclose(policy_target, true_target)
    assert condition.shape == (39,)
    assert np.all(condition[12:15] > 0.0)
