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
from game.stackelberg import stackelberg_select_action, target_best_response


def load_test_config() -> dict:
    with open(PROJECT_ROOT / "configs" / "default.yaml", "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["environment"]["max_steps"] = 5
    config.setdefault("safety", {})["use_lyapunov"] = False
    config.setdefault("sensing", {})["use_fim"] = False
    config.setdefault("sensing", {})["use_belief_state"] = False
    return config


def test_target_best_response_returns_valid_action() -> None:
    config = load_test_config()
    config.setdefault("game", {})["use_intelligent_target"] = True
    env = ThreeUEnv(config, seed=123)
    env.reset()

    action = target_best_response(env, leader_action=env.greedy_action_toward_target(), config=config)

    assert isinstance(action, int)
    assert 0 <= action < 8


def test_stackelberg_select_action_returns_valid_action() -> None:
    config = load_test_config()
    config.setdefault("game", {})["use_stackelberg"] = True
    config.setdefault("game", {})["use_intelligent_target"] = True
    env = ThreeUEnv(config, seed=123)
    env.reset()

    proposed = 0
    selected, info = stackelberg_select_action(env, range(env.action_space_n), proposed, config, return_info=True)

    assert 0 <= selected < env.action_space_n
    assert selected == proposed
    assert info["stackelberg_active"] == 1.0
    assert info["stackelberg_changed_action"] == 0.0
    assert info["stackelberg_evaluated_actions"] == 1.0


def test_disabled_game_matches_original_target_motion() -> None:
    disabled_config = load_test_config()
    disabled_config.setdefault("game", {})["use_stackelberg"] = False
    disabled_config.setdefault("game", {})["use_intelligent_target"] = False

    no_game_config = deepcopy(disabled_config)
    no_game_config.pop("game", None)

    env_disabled = ThreeUEnv(disabled_config, seed=321)
    env_no_game = ThreeUEnv(no_game_config, seed=321)
    obs_disabled = env_disabled.reset()
    obs_no_game = env_no_game.reset()

    assert np.allclose(obs_disabled, obs_no_game)

    next_disabled, reward_disabled, done_disabled, info_disabled = env_disabled.step(0)
    next_no_game, reward_no_game, done_no_game, info_no_game = env_no_game.step(0)

    assert np.allclose(next_disabled, next_no_game)
    assert reward_disabled == reward_no_game
    assert done_disabled == done_no_game
    assert info_disabled["target_best_response_action"] == -1.0
    assert info_no_game["target_best_response_action"] == -1.0
