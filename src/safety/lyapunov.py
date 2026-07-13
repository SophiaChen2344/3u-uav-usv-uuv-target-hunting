"""Lyapunov-inspired risk filtering for the simplified 3U environment.

The functions in this module evaluate a scalar Lyapunov-style score for the
current UAV-USV-UUV state and use one-step lookahead to reject only candidate
UUV actions that create safety risk: leaving bounds, breaking the relay chain,
or running too low on UUV energy.

This is an educational safety filter. It is useful for comparing safer action
selection heuristics, but it is not a formal proof of global stability for the
simplified simulator or for a physical 3U system.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, Tuple

import numpy as np


StateParts = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]


def lyapunov_value(state: Any, metrics: Dict[str, Any] | None, config: Dict[str, Any] | None) -> float:
    """Return the scalar Lyapunov-inspired risk score.

    The score intentionally excludes target-distance or hunting-progress terms
    so the filter cannot take over the tracking task from the trajectory
    generator.
    """

    uav, usv, uuv_center, _target, _direction, _voyage = _state_parts(state)
    safety, _env = _config_parts(config)

    connectivity_risk = _connectivity_risk(uav, usv, uuv_center, metrics, config)
    energy_risk = _energy_shortfall_risk(state, metrics, config)
    boundary_risk = _boundary_risk(uuv_center, config)

    value = (
        float(safety.get("alpha_conn", 0.5)) * connectivity_risk
        + float(safety.get("alpha_energy", 0.2)) * energy_risk
        + float(safety.get("alpha_boundary", 0.2)) * boundary_risk
    )
    return float(np.nan_to_num(value, nan=np.inf, posinf=np.inf, neginf=np.inf))


def lyapunov_delta(
    prev_state: Any,
    next_state: Any,
    prev_metrics: Dict[str, Any] | None,
    next_metrics: Dict[str, Any] | None,
    config: Dict[str, Any] | None,
) -> float:
    """Return ``V(next_state) - V(prev_state)``."""

    prev_value = lyapunov_value(prev_state, prev_metrics, config)
    next_value = lyapunov_value(next_state, next_metrics, config)
    return float(next_value - prev_value)


def is_safe_transition(
    prev_state: Any,
    next_state: Any,
    prev_metrics: Dict[str, Any] | None,
    next_metrics: Dict[str, Any] | None,
    config: Dict[str, Any] | None,
) -> bool:
    """Check whether a transition stays within the configured risk envelope."""

    return bool(_condition_margin(prev_state, next_state, prev_metrics, next_metrics, config) <= 0.0)


def filter_action(env: Any, candidate_action: int, config: Dict[str, Any] | None) -> Dict[str, Any]:
    """Evaluate one candidate action without mutating ``env``.

    A deep copy of the environment is stepped with Lyapunov filtering disabled.
    Copying preserves the random generator state, so every candidate sees the
    same one-step random USV perturbation and target escape draw.
    """

    candidate_action = _action_to_int(candidate_action)
    prev_state = env.get_state().copy()
    prev_metrics = dict(getattr(env, "last_info", {}) or {})

    trial_env = deepcopy(env)
    _disable_trial_safety(trial_env)
    next_state, _reward, done, next_metrics = trial_env.step(candidate_action)

    prev_value = lyapunov_value(prev_state, prev_metrics, config)
    next_value = lyapunov_value(next_state, next_metrics, config)
    delta = float(next_value - prev_value)
    risk_tolerance = _risk_tolerance(config)
    margin = _condition_margin(prev_state, next_state, prev_metrics, next_metrics, config)
    is_safe = bool(margin <= 0.0)

    return {
        "candidate_action": int(candidate_action),
        "executed_action": int(candidate_action),
        "is_safe": float(is_safe),
        "safety_violation": float(not is_safe),
        "lyapunov_value": float(next_value),
        "lyapunov_prev_value": float(prev_value),
        "lyapunov_delta": float(delta),
        "lyapunov_bound": float(risk_tolerance),
        "lyapunov_margin": float(margin),
        "predicted_done": float(done),
        "predicted_target_distance": float(next_metrics.get("target_distance", np.nan)),
        "predicted_connected_fraction": float(next_metrics.get("connected_fraction", np.nan)),
    }


def select_safe_action(
    env: Any,
    proposed_action: int,
    all_actions: Iterable[int],
    config: Dict[str, Any] | None,
) -> Tuple[int, Dict[str, Any]]:
    """Return a Lyapunov-filtered action and diagnostics.

    The proposed action is accepted when it satisfies the one-step condition. If
    it fails, every other discrete action is checked and the safe action with
    the lowest predicted Lyapunov value is selected. If no action is safe, the
    least unsafe action is returned and ``safety_violation`` is set.
    """

    proposed_action = _action_to_int(proposed_action)
    actions = [_action_to_int(action) for action in all_actions]
    if proposed_action not in actions:
        actions.insert(0, proposed_action)

    proposed = filter_action(env, proposed_action, config)
    proposed["original_action"] = int(proposed_action)
    proposed["action_replaced"] = 0.0
    proposed["proposed_was_safe"] = proposed["is_safe"]
    if bool(proposed["is_safe"]):
        proposed["safety_violation"] = 0.0
        return proposed_action, proposed

    candidates = [proposed]
    for action in actions:
        if action == proposed_action:
            continue
        result = filter_action(env, action, config)
        result["original_action"] = int(proposed_action)
        candidates.append(result)

    safe_candidates = [candidate for candidate in candidates if bool(candidate["is_safe"])]
    if safe_candidates:
        chosen = min(safe_candidates, key=lambda item: (item["lyapunov_value"], abs(item["lyapunov_delta"])))
        chosen["safety_violation"] = 0.0
    else:
        _safety, _env_config = _config_parts(config)
        # "least_unsafe" is currently the only fallback, but the config value is
        # still read so experiments can document the intended behavior.
        _fallback_mode = str(_safety.get("fallback_mode", "least_unsafe"))
        chosen = min(candidates, key=lambda item: (item["lyapunov_margin"], item["lyapunov_value"]))
        chosen["safety_violation"] = 1.0

    executed_action = int(chosen["candidate_action"])
    chosen["executed_action"] = executed_action
    chosen["original_action"] = int(proposed_action)
    chosen["action_replaced"] = float(executed_action != proposed_action)
    chosen["proposed_was_safe"] = 0.0
    return executed_action, chosen


def lyapunov_components(
    state: Any,
    metrics: Dict[str, Any] | None,
    config: Dict[str, Any] | None,
) -> Dict[str, float]:
    """Expose component values for diagnostics and tests."""

    uav, usv, uuv_center, _target, _direction, _voyage = _state_parts(state)
    return {
        "target_error": 0.0,
        "formation_error": 0.0,
        "connectivity_risk": _connectivity_risk(uav, usv, uuv_center, metrics, config),
        "energy_imbalance_risk": 0.0,
        "energy_shortfall_risk": _energy_shortfall_risk(state, metrics, config),
        "boundary_risk": _boundary_risk(uuv_center, config),
    }


def _state_parts(state: Any) -> StateParts:
    if hasattr(state, "uav_position"):
        return (
            np.asarray(state.uav_position, dtype=float),
            np.asarray(state.usv_position, dtype=float),
            np.asarray(state.uuv_center, dtype=float),
            np.asarray(state.target_position, dtype=float),
            np.asarray(state.direction_gw, dtype=float),
            float(getattr(state, "total_voyage_distance", 0.0)),
        )

    if isinstance(state, dict):
        return (
            np.asarray(state["uav_position"], dtype=float),
            np.asarray(state["usv_position"], dtype=float),
            np.asarray(state["uuv_center"], dtype=float),
            np.asarray(state["target_position"], dtype=float),
            np.asarray(state.get("direction_gw", np.zeros(3)), dtype=float),
            float(state.get("total_voyage_distance", 0.0)),
        )

    array_state = np.asarray(state, dtype=float).ravel()
    if array_state.size < 15:
        raise ValueError("Lyapunov state vector must contain at least 15 elements.")
    voyage = float(array_state[15]) if array_state.size > 15 else 0.0
    return (
        array_state[0:3],
        array_state[3:6],
        array_state[6:9],
        array_state[9:12],
        array_state[12:15],
        voyage,
    )


def _config_parts(config: Dict[str, Any] | None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    config = config or {}
    safety = dict(config.get("safety", config if "alpha_target" in config else {}))
    env_config = dict(config.get("environment", {}))
    return safety, env_config


def _metric(metrics: Dict[str, Any] | None, name: str, default: Any = None) -> Any:
    if metrics is None:
        return default
    return metrics.get(name, default)


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def _connectivity_risk(
    uav: np.ndarray,
    usv: np.ndarray,
    uuv_center: np.ndarray,
    metrics: Dict[str, Any] | None,
    config: Dict[str, Any] | None,
) -> float:
    _safety, env_config = _config_parts(config)
    uav_usv_range = float(env_config.get("uav_usv_range", 700.0))
    usv_uuv_range = float(env_config.get("usv_uuv_range", env_config.get("uuv_acoustic_range", 700.0)))

    us_distance = float(_metric(metrics, "us_distance", _metric(metrics, "usv_uav_distance", _distance(uav, usv))))
    sg_distance = float(
        _metric(metrics, "sg_distance", _metric(metrics, "mean_uuv_usv_distance", _distance(usv, uuv_center)))
    )

    us_ratio = us_distance / max(uav_usv_range, 1e-9)
    sg_ratio = sg_distance / max(usv_uuv_range, 1e-9)
    distance_risk = us_ratio**2 + sg_ratio**2
    connected_fraction = _metric(metrics, "connected_fraction", None)
    if connected_fraction is None:
        disconnected_risk = float(us_ratio > 1.0) + float(sg_ratio > 1.0)
    else:
        disconnected_risk = max(0.0, 1.0 - float(connected_fraction))
    return float(distance_risk + 2.0 * disconnected_risk)


def _energy_imbalance_risk(state: Any, metrics: Dict[str, Any] | None) -> float:
    energy_values = _metric(metrics, "uuv_energy", None)
    if energy_values is None and hasattr(state, "uuv_energy"):
        energy_values = state.uuv_energy
    if energy_values is not None:
        energy_array = np.asarray(energy_values, dtype=float).ravel()
        if energy_array.size >= 2:
            mean_energy = float(np.mean(np.abs(energy_array)))
            return float(np.std(energy_array) / max(mean_energy, 1e-9))

    remaining_mean = _metric(metrics, "remaining_energy_mean", None)
    remaining_min = _metric(metrics, "remaining_energy_min", None)
    if remaining_mean is not None and remaining_min is not None:
        return float(max(0.0, float(remaining_mean) - float(remaining_min)) / max(abs(float(remaining_mean)), 1e-9))
    return 0.0


def _energy_shortfall_risk(state: Any, metrics: Dict[str, Any] | None, config: Dict[str, Any] | None) -> float:
    safety, env_config = _config_parts(config)
    reserve_fraction = float(safety.get("energy_reserve_fraction", 0.05))
    budget = float(_metric(metrics, "energy_budget", env_config.get("uuv_energy_budget", 65_000.0)))
    budget = max(abs(budget), 1e-9)

    remaining_min = _metric(metrics, "remaining_energy_min", None)
    if remaining_min is None and hasattr(state, "uuv_energy"):
        energy_array = np.asarray(state.uuv_energy, dtype=float).ravel()
        if energy_array.size:
            remaining_min = float(np.min(energy_array))
    if remaining_min is None:
        total_used = float(_metric(metrics, "total_energy_used", 0.0))
        remaining_min = budget - total_used / 3.0

    reserve = reserve_fraction * budget
    return float(max(0.0, reserve - float(remaining_min)) / budget)


def _boundary_risk(uuv_center: np.ndarray, config: Dict[str, Any] | None) -> float:
    safety, env_config = _config_parts(config)
    area_size = float(env_config.get("area_size", 400.0))
    margin = float(safety.get("boundary_margin", 0.1 * area_size))
    margin = max(margin, 1e-9)
    return float(_point_boundary_risk(uuv_center, area_size, margin))


def _point_boundary_risk(position: np.ndarray, area_size: float, margin: float) -> float:
    position = np.asarray(position, dtype=float)
    min_edge_distance = min(position[0], position[1], area_size - position[0], area_size - position[1])
    if min_edge_distance >= margin:
        return 0.0
    return float(((margin - min_edge_distance) / margin) ** 2)


def _risk_tolerance(config: Dict[str, Any] | None) -> float:
    safety, _env_config = _config_parts(config)
    return float(safety.get("risk_tolerance", safety.get("epsilon", 1.0)))


def _hard_safety_violation(
    next_state: Any,
    next_metrics: Dict[str, Any] | None,
    config: Dict[str, Any] | None,
) -> float:
    uav, usv, uuv_center, _target, _direction, _voyage = _state_parts(next_state)
    _safety, env_config = _config_parts(config)
    area_size = float(env_config.get("area_size", 400.0))
    uav_usv_range = float(env_config.get("uav_usv_range", 700.0))
    usv_uuv_range = float(env_config.get("usv_uuv_range", env_config.get("uuv_acoustic_range", 700.0)))
    connected_fraction = _metric(next_metrics, "connected_fraction", None)
    if connected_fraction is None:
        connected_fraction = 0.5 * float(_distance(uav, usv) <= uav_usv_range)
        connected_fraction += 0.5 * float(_distance(usv, uuv_center) <= usv_uuv_range)

    remaining_min = _metric(next_metrics, "remaining_energy_min", None)
    energy_shortfall = _energy_shortfall_risk(next_state, next_metrics, config)
    constraints_satisfied = float(_metric(next_metrics, "constraints_satisfied", 1.0))
    in_bounds = 0.0 <= uuv_center[0] <= area_size and 0.0 <= uuv_center[1] <= area_size
    disconnected = float(connected_fraction) < 1.0
    energy_low = energy_shortfall > 0.0 or (remaining_min is not None and float(remaining_min) <= 0.0)
    violated = (constraints_satisfied < 1.0) or (not in_bounds) or disconnected or energy_low
    return float(violated)


def _condition_margin(
    prev_state: Any,
    next_state: Any,
    prev_metrics: Dict[str, Any] | None,
    next_metrics: Dict[str, Any] | None,
    config: Dict[str, Any] | None,
) -> float:
    delta = lyapunov_delta(prev_state, next_state, prev_metrics, next_metrics, config)
    hard_violation = _hard_safety_violation(next_state, next_metrics, config)
    return float(delta - _risk_tolerance(config) + 1_000.0 * hard_violation)


def _action_to_int(action: Any) -> int:
    if isinstance(action, (tuple, list, np.ndarray)):
        return int(np.asarray(action).ravel()[0])
    return int(action)


def _disable_trial_safety(env: Any) -> None:
    if hasattr(env, "use_lyapunov"):
        env.use_lyapunov = False
    if hasattr(env, "safety_config"):
        env.safety_config = dict(getattr(env, "safety_config", {}) or {})
        env.safety_config["use_lyapunov"] = False
    if hasattr(env, "use_stackelberg"):
        env.use_stackelberg = False
    if hasattr(env, "game_config"):
        env.game_config = dict(getattr(env, "game_config", {}) or {})
        env.game_config["use_stackelberg"] = False
