"""One-step Stackelberg pursuit-evasion game for the 3U simulator.

The leader action is supplied by the trajectory generator or policy. This
module predicts the underwater target follower's escape response after
observing that action; it does not replace the leader action.

This is an approximate educational model. It intentionally uses a cheap
one-step rollout so simulation remains practical while still replacing the old
purely reactive "move away from the UUV" target rule with a rational
best-response target.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import numpy as np

from utils.energy import uuv_group_step_energy


TARGET_STAY_ACTION = 8
EPS = 1e-12


def target_best_response(env: Any, leader_action: int, config: Dict[str, Any] | None) -> int:
    """Return the target action that maximizes follower utility.

    The target evaluates the eight horizontal escape headings, plus an optional
    stay action when configured, after observing the UUV leader action.
    """

    return int(target_best_response_details(env, leader_action, config)["target_action"])


def evaluate_leader_action(
    env: Any,
    leader_action: int,
    target_response: int | Dict[str, Any],
    config: Dict[str, Any] | None,
) -> float:
    """Return the scalar one-step leader cost for a leader-target pair."""

    return float(evaluate_leader_action_details(env, leader_action, target_response, config)["leader_cost"])


def stackelberg_select_action(
    env: Any,
    candidate_actions: Iterable[int] | None,
    proposed_action: int,
    config: Dict[str, Any] | None,
    return_info: bool = False,
) -> int | Tuple[int, Dict[str, float]]:
    """Return ``proposed_action`` plus target-response diagnostics.

    This compatibility function used to reselect the 3U leader action. The
    integrated model now treats Flow Matching or the caller as the sole leader
    action source; Stackelberg only predicts the target follower response for
    that supplied action.
    """

    proposed_action = _action_to_int(proposed_action)
    if not bool(_game_config(config).get("use_stackelberg", True)):
        info = _disabled_info(proposed_action)
        return (proposed_action, info) if return_info else proposed_action

    response = target_best_response_details(env, proposed_action, config)
    leader = evaluate_leader_action_details(env, proposed_action, response, config)
    chosen = {**response, **leader}
    info = {
        "stackelberg_active": 1.0,
        "stackelberg_proposed_action": float(proposed_action),
        "stackelberg_selected_action": float(proposed_action),
        "stackelberg_changed_action": 0.0,
        "stackelberg_leader_cost": float(chosen["leader_cost"]),
        "target_best_response_action": float(chosen["target_action"]),
        "target_utility": float(chosen["target_utility"]),
        "target_uncertainty_score": float(chosen["uncertainty_score"]),
        "target_weak_connectivity_score": float(chosen["weak_connectivity_score"]),
        "target_boundary_penalty": float(chosen["target_boundary_penalty"]),
        "stackelberg_energy_cost": float(chosen["energy_cost"]),
        "stackelberg_connectivity_cost": float(chosen["connectivity_cost"]),
        "stackelberg_information_cost": float(chosen["information_cost"]),
        "stackelberg_lyapunov_penalty": float(chosen["lyapunov_violation_penalty"]),
        "fim_trace_inv": float(chosen["fim_trace_inv"]),
        "stackelberg_evaluated_actions": 1.0,
    }
    return (proposed_action, info) if return_info else proposed_action


def simulate_one_step(env: Any, uuv_action: int, target_action: int) -> Dict[str, Any]:
    """Roll out one deterministic UUV-target step without mutating ``env``.

    The USV random walk is not included in this cheap game rollout; the current
    USV position is used for connectivity scoring. Boundary penalties record
    whether an unclipped candidate would have left the square region.
    """

    _require_state(env)
    uuv_action = _action_to_int(uuv_action)
    target_action = _action_to_int(target_action)
    directions = _action_directions(env)
    area_size = _area_size(env)
    dt = float(getattr(env, "dt", _env_config(getattr(env, "config", {})).get("dt", 1.0)))

    uuv_start = np.asarray(env.state.uuv_center, dtype=float)
    target_start = np.asarray(env.state.target_position, dtype=float)

    uuv_candidate = uuv_start.copy()
    uuv_candidate[:2] += directions[uuv_action] * float(getattr(env, "uuv_speed", 8.0)) * dt
    uuv_next, uuv_clipped = _clip_xy(uuv_candidate, area_size)
    uuv_next[2] = uuv_start[2]

    target_candidate = target_start.copy()
    if target_action != TARGET_STAY_ACTION:
        target_candidate[:2] += directions[target_action] * float(getattr(env, "target_speed", 3.0)) * dt
    target_next, target_clipped = _clip_xy(target_candidate, area_size)
    target_next[2] = target_start[2]

    return {
        "uuv_action": int(uuv_action),
        "target_action": int(target_action),
        "uuv_next": uuv_next,
        "target_next": target_next,
        "uuv_move_distance": _distance(uuv_start, uuv_next),
        "target_move_distance": _distance(target_start, target_next),
        "uuv_boundary_penalty": _boundary_penalty(uuv_candidate, area_size, uuv_clipped),
        "target_boundary_penalty": _boundary_penalty(target_candidate, area_size, target_clipped),
        "target_distance": _distance(uuv_next, target_next),
    }


def target_best_response_details(env: Any, leader_action: int, config: Dict[str, Any] | None) -> Dict[str, Any]:
    """Return the best target action plus diagnostic utility terms."""

    leader_action = _action_to_int(leader_action)
    game = _game_config(config)
    weights = {
        "dist": float(game.get("w_target_dist", 1.0)),
        "info": float(game.get("w_target_info", 0.2)),
        "conn": float(game.get("w_target_conn", 0.2)),
        "boundary": float(game.get("w_target_boundary", 1.0)),
    }

    candidates = []
    for target_action in _target_actions(config):
        sim = simulate_one_step(env, leader_action, target_action)
        target_next = sim["target_next"]
        uuv_next = sim["uuv_next"]
        uncertainty = _information_uncertainty(env, target_next, uuv_next, config)
        weak_conn = _weak_connectivity_score(env, target_next, uuv_next, config)
        utility = (
            weights["dist"] * sim["target_distance"]
            + weights["info"] * uncertainty
            + weights["conn"] * weak_conn
            - weights["boundary"] * sim["target_boundary_penalty"]
        )
        candidates.append(
            {
                "leader_action": int(leader_action),
                "target_action": int(target_action),
                "target_next": target_next,
                "uuv_next": uuv_next,
                "target_utility": float(utility),
                "uncertainty_score": float(uncertainty),
                "weak_connectivity_score": float(weak_conn),
                "target_boundary_penalty": float(sim["target_boundary_penalty"]),
                "fim_trace_inv": float(approximate_fim_trace_inv(env, target_next, uuv_next, config)),
            }
        )

    return max(candidates, key=lambda item: (item["target_utility"], -item["target_boundary_penalty"]))


def evaluate_leader_action_details(
    env: Any,
    leader_action: int,
    target_response: int | Dict[str, Any],
    config: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """Return leader cost terms for a leader action and target response."""

    leader_action = _action_to_int(leader_action)
    if isinstance(target_response, dict):
        target_action = _action_to_int(target_response.get("target_action", 0))
    else:
        target_action = _action_to_int(target_response)

    sim = simulate_one_step(env, leader_action, target_action)
    game = _game_config(config)

    hunt = sim["target_distance"]
    energy = _leader_energy_cost(env, sim)
    comm = _connectivity_cost(env, sim["uuv_next"], config)
    info = _information_cost(env, sim["target_next"], sim["uuv_next"], config)
    lyapunov = _lyapunov_penalty(env, sim["uuv_next"], sim["target_next"], sim, config)

    cost = (
        float(game.get("w_leader_hunt", 1.0)) * hunt
        + float(game.get("w_leader_energy", 0.1)) * energy
        + float(game.get("w_leader_comm", 0.2)) * comm
        + float(game.get("w_leader_info", 0.1)) * info
        + float(game.get("w_leader_lyapunov", 1.0)) * lyapunov
    )

    details = {
        "leader_action": int(leader_action),
        "target_action": int(target_action),
        "leader_cost": float(cost),
        "hunt_distance": float(hunt),
        "energy_cost": float(energy),
        "connectivity_cost": float(comm),
        "information_cost": float(info),
        "lyapunov_violation_penalty": float(lyapunov),
        "uuv_boundary_penalty": float(sim["uuv_boundary_penalty"]),
        "fim_trace_inv": float(approximate_fim_trace_inv(env, sim["target_next"], sim["uuv_next"], config)),
    }
    if isinstance(target_response, dict):
        details.update({key: value for key, value in target_response.items() if key not in details})
    return details


def approximate_fim_trace_inv(
    env: Any,
    target_position: np.ndarray | None = None,
    uuv_position: np.ndarray | None = None,
    config: Dict[str, Any] | None = None,
) -> float:
    """Return inverse FIM trace from the environment's range-bearing model.

    The fallback below is retained for legacy callers that do not expose
    ``compute_fim_metrics_for``.
    """

    _require_state(env)
    target = np.asarray(target_position if target_position is not None else env.state.target_position, dtype=float)
    uuv = np.asarray(uuv_position if uuv_position is not None else env.state.uuv_center, dtype=float)
    if hasattr(env, "compute_fim_metrics_for"):
        try:
            return float(env.compute_fim_metrics_for(uuv, target).get("fim_trace_inv", 0.0))
        except Exception:
            pass

    usv = np.asarray(env.state.usv_position, dtype=float)
    uav = np.asarray(env.state.uav_position, dtype=float)

    area = max(_area_size(env), 1.0)
    search_radius = max(float(getattr(env, "search_radius", 18.0)), 1.0)
    footprint = max(_camera_footprint_radius(env, config), 1.0)
    regularization = float(_info_config(config).get("fim_regularization", 1e-3))

    uuv_strength = 1.0 / (1.0 + (_distance(uuv, target) / search_radius) ** 2)
    usv_strength = 0.35 / (1.0 + (_distance(usv, target) / (0.5 * area)) ** 2)
    uav_xy_distance = _distance(uav[:2], target[:2])
    uav_strength = 0.25 / (1.0 + (uav_xy_distance / footprint) ** 2)
    strength = regularization + uuv_strength + usv_strength + uav_strength
    return float(1.0 / max(strength, EPS))


def _target_actions(config: Dict[str, Any] | None) -> list[int]:
    game = _game_config(config)
    target_action_space = int(game.get("target_action_space", 8))
    actions = list(range(min(target_action_space, 8)))
    if target_action_space > 8 or bool(game.get("allow_target_stay", False)):
        actions.append(TARGET_STAY_ACTION)
    return actions or list(range(8))


def _information_uncertainty(
    env: Any,
    target_position: np.ndarray,
    uuv_position: np.ndarray,
    config: Dict[str, Any] | None,
) -> float:
    if not _fim_enabled(config):
        return 0.0
    return float(approximate_fim_trace_inv(env, target_position, uuv_position, config))


def _information_cost(
    env: Any,
    target_position: np.ndarray,
    uuv_position: np.ndarray,
    config: Dict[str, Any] | None,
) -> float:
    return _information_uncertainty(env, target_position, uuv_position, config)


def _weak_connectivity_score(
    env: Any,
    target_position: np.ndarray,
    uuv_position: np.ndarray,
    config: Dict[str, Any] | None,
) -> float:
    env_cfg = _env_config(config)
    usv = np.asarray(env.state.usv_position, dtype=float)
    uav = np.asarray(env.state.uav_position, dtype=float)
    usv_range = max(float(getattr(env, "usv_uuv_range", env_cfg.get("usv_uuv_range", 700.0))), 1.0)
    uav_range = max(_camera_footprint_radius(env, config), 1.0)
    hunting_range = max(float(getattr(env, "escape_distance", env_cfg.get("escape_distance", 288.0))), 1.0)

    away_from_relay = _distance(usv, target_position) / usv_range
    away_from_uav_footprint = _distance(uav[:2], target_position[:2]) / uav_range
    away_from_hunters = _distance(uuv_position, target_position) / hunting_range
    return float(np.clip(0.45 * away_from_relay + 0.35 * away_from_uav_footprint + 0.20 * away_from_hunters, 0.0, 3.0))


def _leader_energy_cost(env: Any, sim: Dict[str, Any]) -> float:
    env_cfg = _env_config(getattr(env, "config", None))
    uav_usv_range = max(float(getattr(env, "uav_usv_range", env_cfg.get("uav_usv_range", 700.0))), 1.0)
    usv_uuv_range = max(float(getattr(env, "usv_uuv_range", env_cfg.get("usv_uuv_range", 700.0))), 1.0)
    connected = (
        _distance(env.state.uav_position, env.state.usv_position) <= uav_usv_range
        and _distance(env.state.usv_position, sim["uuv_next"]) <= usv_uuv_range
    )
    breakdown = uuv_group_step_energy(
        displacement=float(sim["uuv_move_distance"]),
        dt=max(float(getattr(env, "dt", 1.0)), EPS),
        num_uuvs=max(int(getattr(env, "num_uuvs", 3)), 1),
        communication_distance_m=_distance(env.state.usv_position, sim["uuv_next"]),
        connected=connected,
        energy_config=getattr(env, "energy_config", None),
    )
    return float(breakdown.total)


def _connectivity_cost(env: Any, uuv_next: np.ndarray, config: Dict[str, Any] | None) -> float:
    env_cfg = _env_config(config)
    uav = np.asarray(env.state.uav_position, dtype=float)
    usv = np.asarray(env.state.usv_position, dtype=float)
    uav_usv_range = max(float(getattr(env, "uav_usv_range", env_cfg.get("uav_usv_range", 700.0))), 1.0)
    usv_uuv_range = max(float(getattr(env, "usv_uuv_range", env_cfg.get("usv_uuv_range", 700.0))), 1.0)
    us_ratio = _distance(uav, usv) / uav_usv_range
    sg_ratio = _distance(usv, uuv_next) / usv_uuv_range
    disconnect_penalty = float(us_ratio > 1.0) + float(sg_ratio > 1.0)
    return float(us_ratio**2 + sg_ratio**2 + 2.0 * disconnect_penalty)


def _lyapunov_penalty(
    env: Any,
    uuv_next: np.ndarray,
    target_next: np.ndarray,
    sim: Dict[str, Any],
    config: Dict[str, Any] | None,
) -> float:
    safety = dict((config or {}).get("safety", {}))
    if not bool(safety.get("use_lyapunov", False)):
        return 0.0

    try:
        from safety.lyapunov import is_safe_transition, lyapunov_delta, lyapunov_value
    except Exception:
        return 0.0

    previous_state = env.get_state().copy()
    direction = _safe_unit(target_next - uuv_next)
    next_state = np.concatenate(
        [
            env.state.uav_position,
            env.state.usv_position,
            uuv_next,
            target_next,
            direction,
            np.array([env.state.total_voyage_distance + sim["uuv_move_distance"]], dtype=float),
        ]
    )
    next_metrics = dict(getattr(env, "last_info", {}) or {})
    next_metrics.update(
        {
            "target_distance": _distance(uuv_next, target_next),
            "us_distance": _distance(env.state.uav_position, env.state.usv_position),
            "sg_distance": _distance(env.state.usv_position, uuv_next),
            "total_voyage_distance": env.state.total_voyage_distance + sim["uuv_move_distance"],
        }
    )
    previous_metrics = dict(getattr(env, "last_info", {}) or {})
    safe = is_safe_transition(previous_state, next_state, previous_metrics, next_metrics, config)
    if safe:
        return 0.0
    delta = lyapunov_delta(previous_state, next_state, previous_metrics, next_metrics, config)
    current_value = lyapunov_value(previous_state, previous_metrics, config)
    return float(max(0.0, delta) + 0.01 * max(0.0, current_value))


def _disabled_info(action: int) -> Dict[str, float]:
    return {
        "stackelberg_active": 0.0,
        "stackelberg_proposed_action": float(action),
        "stackelberg_selected_action": float(action),
        "stackelberg_changed_action": 0.0,
        "stackelberg_leader_cost": 0.0,
        "target_best_response_action": -1.0,
        "target_utility": 0.0,
        "target_uncertainty_score": 0.0,
        "target_weak_connectivity_score": 0.0,
        "target_boundary_penalty": 0.0,
        "stackelberg_energy_cost": 0.0,
        "stackelberg_connectivity_cost": 0.0,
        "stackelberg_information_cost": 0.0,
        "stackelberg_lyapunov_penalty": 0.0,
        "fim_trace_inv": 0.0,
        "stackelberg_evaluated_actions": 0.0,
    }


def _action_directions(env: Any) -> np.ndarray:
    directions = np.asarray(getattr(env, "ACTION_DIRECTIONS"), dtype=float)
    if directions.shape[0] < 8 or directions.shape[1] != 2:
        raise ValueError("env.ACTION_DIRECTIONS must contain eight 2D directions.")
    return directions


def _clip_xy(position: np.ndarray, area_size: float) -> Tuple[np.ndarray, bool]:
    candidate = np.asarray(position, dtype=float).copy()
    clipped = candidate.copy()
    clipped[0] = np.clip(clipped[0], 0.0, area_size)
    clipped[1] = np.clip(clipped[1], 0.0, area_size)
    return clipped, bool(not np.allclose(candidate[:2], clipped[:2]))


def _boundary_penalty(position: np.ndarray, area_size: float, clipped: bool) -> float:
    position = np.asarray(position, dtype=float)
    min_edge_distance = min(position[0], position[1], area_size - position[0], area_size - position[1])
    margin = max(0.08 * area_size, 1.0)
    edge_risk = max(0.0, margin - min_edge_distance) / margin
    return float((5.0 if clipped else 0.0) + edge_risk**2)


def _camera_footprint_radius(env: Any, config: Dict[str, Any] | None) -> float:
    env_cfg = _env_config(config)
    height = float(getattr(env, "uav_height", env_cfg.get("uav_height", 120.0)))
    fov_deg = float(env_cfg.get("camera_fov_deg", 70.0))
    min_radius = float(env_cfg.get("min_camera_radius", 120.0))
    half_angle = np.deg2rad(fov_deg) / 2.0
    return float(max(min_radius, height * np.tan(half_angle)))


def _fim_enabled(config: Dict[str, Any] | None) -> bool:
    info = _info_config(config)
    game = _game_config(config)
    return bool(info.get("use_fim", game.get("use_fim", False)))


def _game_config(config: Dict[str, Any] | None) -> Dict[str, Any]:
    return dict((config or {}).get("game", {}))


def _env_config(config: Dict[str, Any] | None) -> Dict[str, Any]:
    return dict((config or {}).get("environment", {}))


def _info_config(config: Dict[str, Any] | None) -> Dict[str, Any]:
    config = config or {}
    info = dict(config.get("sensing", {}))
    info.update(dict(config.get("information", {})))
    return info


def _area_size(env: Any) -> float:
    return float(getattr(env, "area_size", _env_config(getattr(env, "config", {})).get("area_size", 400.0)))


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def _safe_unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm <= EPS:
        return np.zeros_like(vector, dtype=float)
    return vector / norm


def _action_to_int(action: Any) -> int:
    if isinstance(action, (tuple, list, np.ndarray)):
        return int(np.asarray(action).ravel()[0])
    return int(action)


def _require_state(env: Any) -> None:
    if getattr(env, "state", None) is None:
        raise RuntimeError("Call env.reset() before evaluating the Stackelberg game.")


__all__ = [
    "TARGET_STAY_ACTION",
    "target_best_response",
    "target_best_response_details",
    "evaluate_leader_action",
    "evaluate_leader_action_details",
    "stackelberg_select_action",
    "simulate_one_step",
    "approximate_fim_trace_inv",
]
