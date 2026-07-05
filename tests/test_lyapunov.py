from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from envs.three_u_env import ThreeUEnv
from safety.lyapunov import (
    filter_action,
    is_safe_transition,
    lyapunov_components,
    lyapunov_delta,
    lyapunov_value,
    select_safe_action,
)


def load_test_config(use_lyapunov: bool = True) -> dict:
    with open(PROJECT_ROOT / "configs" / "default.yaml", "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["environment"]["max_steps"] = 5
    config.setdefault("training", {})["iterations"] = 5
    config.setdefault("safety", {})["use_lyapunov"] = use_lyapunov
    return config


def test_lyapunov_value_is_finite_and_nonnegative() -> None:
    config = load_test_config()
    env = ThreeUEnv(config, seed=11)
    state = env.reset()

    value = lyapunov_value(state, env.last_info, config)
    components = lyapunov_components(state, env.last_info, config)

    assert np.isfinite(value)
    assert value >= 0.0
    assert components["target_error"] >= 0.0
    assert components["connectivity_risk"] >= 0.0


def test_lyapunov_delta_and_transition_condition_return_scalars() -> None:
    config = load_test_config(use_lyapunov=False)
    env = ThreeUEnv(config, seed=12)
    previous_state = env.reset()
    previous_metrics = dict(env.last_info)

    next_state, _reward, _done, next_metrics = env.step(env.greedy_action_toward_target())
    delta = lyapunov_delta(previous_state, next_state, previous_metrics, next_metrics, config)
    safe = is_safe_transition(previous_state, next_state, previous_metrics, next_metrics, config)

    assert isinstance(delta, float)
    assert isinstance(safe, bool)
    assert np.isfinite(delta)


def test_filter_and_select_safe_action_do_not_mutate_environment() -> None:
    config = load_test_config()
    env = ThreeUEnv(config, seed=13)
    state = env.reset()
    action = env.greedy_action_toward_target()

    result = filter_action(env, action, config)
    selected_action, selected_info = select_safe_action(env, action, range(env.action_space_n), config)

    assert np.allclose(env.get_state(), state)
    assert 0 <= selected_action < env.action_space_n
    assert result["candidate_action"] == action
    assert "lyapunov_value" in selected_info
    assert "safety_violation" in selected_info


def test_disabled_filter_executes_original_action() -> None:
    config = load_test_config(use_lyapunov=False)
    env = ThreeUEnv(config, seed=14)
    env.reset()
    action = 0

    _state, _reward, _done, info = env.step(action)

    assert info["safety_filter_active"] == 0.0
    assert info["original_action"] == float(action)
    assert info["executed_action"] == float(action)
    assert info["action_replaced"] == 0.0


def test_boundary_risk_increases_near_edges() -> None:
    config = load_test_config()
    env = ThreeUEnv(config, seed=15)
    center_state = env.reset()
    edge_state = deepcopy(center_state)
    edge_state[6:8] = np.array([5.0, 5.0])
    edge_state[9:11] = np.array([395.0, 395.0])

    center_components = lyapunov_components(center_state, env.last_info, config)
    edge_components = lyapunov_components(edge_state, env.last_info, config)

    assert edge_components["boundary_risk"] > center_components["boundary_risk"]

