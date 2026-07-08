"""Conditional Flow Matching trajectory proposals for 3U target hunting.

The model in this file is intentionally lightweight. It learns a vector field
that transports noisy short-horizon UUV-center trajectories toward synthetic
expert trajectories conditioned on the current 3U state and diagnostics. At
runtime it proposes smooth candidate trajectories; a separate cost function
selects the safest low-cost candidate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from safety.lyapunov import lyapunov_value
from utils.energy import motion_energy
from utils.physics import distance, safe_norm, safe_unit_vector


@dataclass
class TrajectoryScore:
    """Cost and safety diagnostics for one candidate trajectory."""

    total_cost: float
    hunting_cost: float
    energy_cost: float
    communication_risk: float
    information_cost: float
    game_cost: float
    lyapunov_penalty: float
    smoothness: float
    is_safe: bool
    safety_violations: int


class FlowMatchingTrajectoryModel(nn.Module):
    """Small MLP vector field ``v_theta(x_t, t, condition)``.

    ``x_t`` is a flattened short-horizon trajectory, ``t`` is a scalar in
    ``[0, 1]``, and ``condition`` is the 3U context vector produced by the
    environment. The output has the same dimension as ``x_t``.
    """

    def __init__(
        self,
        condition_dim: int,
        horizon: int = 10,
        point_dim: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.condition_dim = int(condition_dim)
        self.horizon = int(horizon)
        self.point_dim = int(point_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.trajectory_dim = self.horizon * self.point_dim

        layers: list[nn.Module] = []
        input_dim = self.trajectory_dim + 1 + self.condition_dim
        for layer_idx in range(max(1, self.num_layers)):
            layers.append(nn.Linear(input_dim if layer_idx == 0 else self.hidden_dim, self.hidden_dim))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(self.hidden_dim, self.trajectory_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Evaluate the vector field for a batch of interpolated trajectories."""

        if x_t.ndim == 3:
            x_t = x_t.reshape(x_t.shape[0], -1)
        if t.ndim == 1:
            t = t.unsqueeze(1)
        if condition.ndim == 1:
            condition = condition.unsqueeze(0)
        if condition.shape[0] == 1 and x_t.shape[0] > 1:
            condition = condition.repeat(x_t.shape[0], 1)

        model_input = torch.cat([x_t, t, condition], dim=1)
        return self.net(model_input)

    @torch.no_grad()
    def sample_trajectories(
        self,
        condition: np.ndarray | torch.Tensor,
        num_samples: int = 16,
        horizon: int | None = None,
        start_position: np.ndarray | None = None,
        area_size: float = 400.0,
        depth: float = -120.0,
        noise_scale: float = 18.0,
        integration_steps: int = 20,
        device: str | torch.device | None = None,
    ) -> np.ndarray:
        """Draw candidate trajectories with simple Euler integration."""

        horizon = int(horizon or self.horizon)
        if horizon != self.horizon:
            raise ValueError(f"Model horizon is {self.horizon}, received {horizon}.")

        model_device = next(self.parameters()).device
        target_device = torch.device(device) if device is not None else model_device
        self.to(target_device)
        was_training = self.training
        self.eval()

        condition_t = torch.as_tensor(condition, dtype=torch.float32, device=target_device)
        if condition_t.ndim == 1:
            condition_t = condition_t.unsqueeze(0)
        condition_t = condition_t.repeat(int(num_samples), 1)

        if start_position is None:
            start_position = _start_from_condition(np.asarray(condition_t[0].detach().cpu().numpy(), dtype=float))
        start_position = np.asarray(start_position, dtype=np.float32)
        x = _initial_noise_trajectories(
            start_position=start_position,
            num_samples=int(num_samples),
            horizon=horizon,
            area_size=float(area_size),
            depth=float(depth),
            noise_scale=float(noise_scale),
        )
        x_t = torch.as_tensor(x.reshape(int(num_samples), -1), dtype=torch.float32, device=target_device)

        steps = max(1, int(integration_steps))
        dt = 1.0 / steps
        for step in range(steps):
            t_value = torch.full((int(num_samples), 1), float(step) / steps, dtype=torch.float32, device=target_device)
            velocity = self.forward(x_t, t_value, condition_t)
            x_t = x_t + velocity * dt

        trajectories = x_t.reshape(int(num_samples), horizon, self.point_dim).detach().cpu().numpy()
        trajectories = project_trajectories(trajectories, area_size=area_size, depth=depth)
        if was_training:
            self.train()
        return trajectories

    def save(self, path: str | Path) -> None:
        """Save model weights and shape metadata."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "condition_dim": self.condition_dim,
                "horizon": self.horizon,
                "point_dim": self.point_dim,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        hidden_dim: int = 256,
        num_layers: int = 3,
        map_location: str | torch.device = "cpu",
    ) -> "FlowMatchingTrajectoryModel":
        """Load a saved Flow Matching trajectory model."""

        checkpoint = torch.load(path, map_location=map_location)
        model = cls(
            condition_dim=int(checkpoint["condition_dim"]),
            horizon=int(checkpoint["horizon"]),
            point_dim=int(checkpoint.get("point_dim", 3)),
            hidden_dim=int(checkpoint.get("hidden_dim", hidden_dim)),
            num_layers=int(checkpoint.get("num_layers", num_layers)),
        )
        model.load_state_dict(checkpoint["state_dict"])
        return model


def sample_trajectories(
    model: FlowMatchingTrajectoryModel,
    condition: np.ndarray,
    num_samples: int = 16,
    horizon: int | None = None,
    env: Any | None = None,
    config: Dict[str, Any] | None = None,
) -> np.ndarray:
    """Convenience wrapper that samples and projects candidate trajectories."""

    cfg = _flow_config(config)
    area_size = float(getattr(env, "area_size", cfg.get("area_size", 400.0)))
    depth = float(getattr(env, "uuv_initial_depth", cfg.get("depth", -120.0)))
    start_position = None
    if env is not None and getattr(env, "state", None) is not None:
        start_position = env.state.uuv_center
    return model.sample_trajectories(
        condition=condition,
        num_samples=int(num_samples),
        horizon=int(horizon or cfg.get("horizon", model.horizon)),
        start_position=start_position,
        area_size=area_size,
        depth=depth,
        noise_scale=float(cfg.get("noise_scale", 18.0)),
        integration_steps=int(cfg.get("integration_steps", 20)),
    )


def train_flow_matching(
    dataset: Dict[str, np.ndarray] | str | Path,
    config: Dict[str, Any] | None = None,
) -> Tuple[FlowMatchingTrajectoryModel, pd.DataFrame]:
    """Train the conditional vector field with the CFM objective.

    The synthetic dataset stores absolute future UUV-center positions. Noisy
    start trajectories are sampled around the current group center stored in
    the condition vector, then linearly interpolated toward the target
    trajectory.
    """

    data = _load_dataset(dataset)
    conditions = np.asarray(data["conditions"], dtype=np.float32)
    trajectories = np.asarray(data["trajectories"], dtype=np.float32)
    if conditions.ndim != 2:
        raise ValueError("conditions must have shape [N, condition_dim].")
    if trajectories.ndim != 3 or trajectories.shape[2] != 3:
        raise ValueError("trajectories must have shape [N, horizon, 3].")

    cfg = _flow_config(config)
    device = _torch_device(cfg.get("device", "auto"))
    model = FlowMatchingTrajectoryModel(
        condition_dim=conditions.shape[1],
        horizon=trajectories.shape[1],
        hidden_dim=int(cfg.get("hidden_dim", 256)),
        num_layers=int(cfg.get("num_layers", 3)),
    ).to(device)

    condition_t = torch.as_tensor(conditions, dtype=torch.float32)
    trajectory_t = torch.as_tensor(trajectories.reshape(trajectories.shape[0], -1), dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(condition_t, trajectory_t),
        batch_size=int(cfg.get("batch_size", 128)),
        shuffle=True,
        drop_last=False,
    )
    optimizer = Adam(model.parameters(), lr=float(cfg.get("lr", 1e-3)))
    loss_fn = nn.MSELoss()
    records = []

    epochs = int(cfg.get("epochs", 50))
    noise_scale = float(cfg.get("noise_scale", 18.0))
    area_size = float(_env_config(config).get("area_size", 400.0))
    depth = -abs(float(_env_config(config).get("initial_uuv_depth", 120.0)))

    for epoch in range(epochs):
        epoch_losses = []
        for condition_batch, target_batch in loader:
            condition_batch = condition_batch.to(device)
            target_batch = target_batch.to(device)
            batch_size = condition_batch.shape[0]

            start_positions = condition_batch[:, 6:9].detach().cpu().numpy()
            noise = np.stack(
                [
                    _initial_noise_trajectories(
                        start_position=start,
                        num_samples=1,
                        horizon=trajectories.shape[1],
                        area_size=area_size,
                        depth=depth,
                        noise_scale=noise_scale,
                    )[0]
                    for start in start_positions
                ],
                axis=0,
            )
            x0 = torch.as_tensor(noise.reshape(batch_size, -1), dtype=torch.float32, device=device)
            t = torch.rand(batch_size, 1, dtype=torch.float32, device=device)
            x_t = (1.0 - t) * x0 + t * target_batch
            target_velocity = target_batch - x0

            prediction = model(x_t, t, condition_batch)
            loss = loss_fn(prediction, target_velocity)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("gradient_clip", 5.0)))
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        records.append({"epoch": epoch, "loss": float(np.mean(epoch_losses)) if epoch_losses else np.nan})

    history = pd.DataFrame.from_records(records)
    checkpoint_path = cfg.get("checkpoint_path")
    if checkpoint_path:
        model.save(checkpoint_path)
    return model, history


def heuristic_candidate_trajectories(
    env: Any,
    num_candidates: int,
    horizon: int,
    coarse_action: int | None = None,
) -> np.ndarray:
    """Generate smooth pursuit-style fallback trajectories.

    These candidates are used both for synthetic expert data and as a runtime
    guardrail when the learned vector field is untrained or uncertain.
    """

    if getattr(env, "state", None) is None:
        env.reset()

    trajectories = []
    current = env.state.uuv_center.copy()
    target = env.state.target_position.copy()
    speed = float(env.uuv_speed) * float(env.dt)
    target_step = float(env.target_speed) * float(env.dt)
    coarse_direction = _coarse_direction(env, coarse_action)
    lateral = np.array([-coarse_direction[1], coarse_direction[0], 0.0], dtype=float)
    if safe_norm(lateral) <= 0.0:
        lateral = np.array([0.0, 1.0, 0.0], dtype=float)

    offsets = np.linspace(-0.6, 0.6, max(1, int(num_candidates)))
    for candidate_idx in range(int(num_candidates)):
        position = current.copy()
        target_prediction = target.copy()
        trajectory = []
        offset = offsets[candidate_idx % len(offsets)]
        for step in range(int(horizon)):
            pursuit = safe_unit_vector(target_prediction - position)
            blend = 0.72 * pursuit + 0.20 * coarse_direction + 0.08 * offset * lateral
            if candidate_idx % 5 == 1:
                blend = 0.55 * pursuit + 0.35 * coarse_direction + 0.10 * lateral
            elif candidate_idx % 5 == 2:
                blend = 0.60 * pursuit + 0.25 * coarse_direction - 0.15 * lateral
            elif candidate_idx % 5 == 3:
                blend = pursuit
            elif candidate_idx % 5 == 4:
                blend = 0.85 * pursuit + 0.15 * offset * lateral
            direction = safe_unit_vector(blend)
            position = position + direction * speed
            position = env.project_position(position)
            position[2] = env.uuv_initial_depth
            trajectory.append(position.copy())

            escape_direction = safe_unit_vector(target_prediction - position)
            target_prediction = env.project_position(target_prediction + escape_direction * target_step)
            target_prediction[2] = env.hunting_depth
        trajectories.append(np.asarray(trajectory, dtype=float))

    return np.asarray(trajectories, dtype=np.float32)


def project_trajectories(
    trajectories: np.ndarray,
    area_size: float = 400.0,
    depth: float = -120.0,
) -> np.ndarray:
    """Project generated trajectories into the square environment bounds."""

    projected = np.asarray(trajectories, dtype=np.float32).copy()
    projected[..., 0] = np.clip(projected[..., 0], 0.0, float(area_size))
    projected[..., 1] = np.clip(projected[..., 1], 0.0, float(area_size))
    projected[..., 2] = float(depth)
    return projected


def predict_target_response(env: Any, trajectory: np.ndarray) -> np.ndarray:
    """Predict a simple Stackelberg-style target best response.

    The target moves away from each proposed UUV center, matching the simplified
    escape dynamics used by the environment.
    """

    target = env.state.target_position.copy()
    responses = []
    step = float(env.target_speed) * float(env.dt)
    for center in np.asarray(trajectory, dtype=float):
        direction = safe_unit_vector(target - center)
        if safe_norm(direction[:2]) <= 0.0:
            direction = np.array([1.0, 0.0, 0.0], dtype=float)
        direction[2] = 0.0
        target = env.project_position(target + safe_unit_vector(direction) * step)
        target[2] = env.hunting_depth
        responses.append(target.copy())
    return np.asarray(responses, dtype=float)


def score_candidate_trajectory(
    env: Any,
    trajectory: np.ndarray,
    config: Dict[str, Any] | None = None,
    use_fim: bool = True,
    use_stackelberg: bool = True,
    use_lyapunov: bool = True,
) -> TrajectoryScore:
    """Compute the weighted trajectory score requested by the integration plan."""

    flow_cfg = _flow_config(config)
    scoring = dict(flow_cfg.get("scoring", {}))
    weights = {
        "w_hunt": float(scoring.get("w_hunt", 1.0)),
        "w_energy": float(scoring.get("w_energy", 0.001)),
        "w_comm": float(scoring.get("w_comm", 3.0)),
        "w_info": float(scoring.get("w_info", 0.25)),
        "w_game": float(scoring.get("w_game", 0.5)),
        "w_lyapunov": float(scoring.get("w_lyapunov", 2.0)),
        "w_smooth": float(scoring.get("w_smooth", 0.05)),
    }

    projected = env.project_trajectory(np.asarray(trajectory, dtype=float))
    target_path = predict_target_response(env, projected) if use_stackelberg else np.repeat(
        env.state.target_position[None, :], projected.shape[0], axis=0
    )
    final_distance = distance(projected[-1], target_path[-1])

    positions = np.vstack([env.state.uuv_center[None, :], projected])
    displacements = np.diff(positions, axis=0)
    speeds = np.linalg.norm(displacements, axis=1) / max(float(env.dt), 1e-12)
    energy_cost = float(sum(motion_energy(speed, env.dt) for speed in speeds))

    comm_risks = []
    info_costs = []
    lyapunov_penalty = 0.0
    safety_violations = 0
    previous_state = env.get_state().copy()
    previous_metrics = dict(env.last_info)
    previous_value = lyapunov_value(previous_state, previous_metrics, config)

    for center, target in zip(projected, target_path):
        us_distance = distance(env.state.uav_position, env.state.usv_position)
        sg_distance = distance(env.state.usv_position, center)
        us_ratio = us_distance / max(float(env.uav_usv_range), 1e-9)
        sg_ratio = sg_distance / max(float(env.usv_uuv_range), 1e-9)
        comm_risk = us_ratio**2 + sg_ratio**2 + float(us_ratio > 1.0) + float(sg_ratio > 1.0)
        comm_risks.append(comm_risk)

        if use_fim and hasattr(env, "compute_fim_metrics_for"):
            fim = env.compute_fim_metrics_for(center, target, comm_risk=comm_risk)
            info_costs.append(float(fim["trace_inv"]))
        else:
            info_costs.append(float(distance(center, target) / max(env.search_radius, 1e-9)))

        next_state = previous_state.copy()
        next_state[6:9] = center
        next_state[9:12] = target
        next_state[12:15] = safe_unit_vector(target - center)
        next_metrics = dict(previous_metrics)
        next_metrics.update(
            {
                "us_distance": us_distance,
                "sg_distance": sg_distance,
                "target_distance": distance(center, target),
                "connected_fraction": 0.5 * float(us_ratio <= 1.0) + 0.5 * float(sg_ratio <= 1.0),
            }
        )
        next_value = lyapunov_value(next_state, next_metrics, config)
        target_error = float(np.sum((target - center) ** 2))
        safety_cfg = dict((config or {}).get("safety", {}))
        bound = -float(safety_cfg.get("c", 0.001)) * target_error + float(safety_cfg.get("epsilon", 1.0))
        margin = float(next_value - previous_value - bound)
        if use_lyapunov and margin > 0.0:
            lyapunov_penalty += margin
            safety_violations += 1
        previous_state = next_state
        previous_metrics = next_metrics
        previous_value = next_value

    smoothness = trajectory_smoothness(projected)
    communication_risk = float(np.mean(comm_risks)) if comm_risks else 0.0
    information_cost = float(np.mean(info_costs)) if info_costs else 0.0
    game_cost = float(np.mean(np.linalg.norm(target_path - projected, axis=1)))

    total = (
        weights["w_hunt"] * final_distance
        + weights["w_energy"] * energy_cost
        + weights["w_comm"] * communication_risk
        + weights["w_info"] * information_cost
        + weights["w_game"] * game_cost
        + weights["w_lyapunov"] * lyapunov_penalty
        + weights["w_smooth"] * smoothness
    )
    is_safe = bool(safety_violations == 0 and np.all(np.isfinite(projected)))
    return TrajectoryScore(
        total_cost=float(total),
        hunting_cost=float(final_distance),
        energy_cost=float(energy_cost),
        communication_risk=float(communication_risk),
        information_cost=float(information_cost),
        game_cost=float(game_cost),
        lyapunov_penalty=float(lyapunov_penalty),
        smoothness=float(smoothness),
        is_safe=is_safe,
        safety_violations=int(safety_violations),
    )


def select_best_trajectory(
    env: Any,
    trajectories: np.ndarray,
    config: Dict[str, Any] | None = None,
    use_fim: bool = True,
    use_stackelberg: bool = True,
    use_lyapunov: bool = True,
) -> Tuple[np.ndarray, TrajectoryScore, pd.DataFrame]:
    """Select the lowest-cost safe trajectory, falling back to least unsafe."""

    rows = []
    scored = []
    for idx, trajectory in enumerate(np.asarray(trajectories, dtype=float)):
        score = score_candidate_trajectory(
            env,
            trajectory,
            config=config,
            use_fim=use_fim,
            use_stackelberg=use_stackelberg,
            use_lyapunov=use_lyapunov,
        )
        scored.append((idx, trajectory, score))
        row = asdict(score)
        row["candidate"] = idx
        rows.append(row)

    safe = [item for item in scored if item[2].is_safe]
    pool = safe if safe else scored
    chosen_idx, chosen_traj, chosen_score = min(
        pool,
        key=lambda item: (item[2].total_cost, item[2].safety_violations, item[2].lyapunov_penalty),
    )
    diagnostics = pd.DataFrame.from_records(rows)
    diagnostics["selected"] = diagnostics["candidate"] == chosen_idx
    return env.project_trajectory(chosen_traj), chosen_score, diagnostics


def trajectory_smoothness(trajectory: np.ndarray) -> float:
    """Return a simple squared-acceleration smoothness score."""

    trajectory = np.asarray(trajectory, dtype=float)
    if trajectory.shape[0] < 3:
        return 0.0
    accelerations = trajectory[2:] - 2.0 * trajectory[1:-1] + trajectory[:-2]
    return float(np.mean(np.sum(accelerations**2, axis=1)))


class FlowMatchingPlanner:
    """Trajectory proposal planner that can wrap a DQN coarse policy."""

    def __init__(
        self,
        model: FlowMatchingTrajectoryModel | None = None,
        config: Dict[str, Any] | None = None,
        mode: str = "flow_matching",
    ) -> None:
        self.model = model
        self.config = config or {}
        self.mode = str(mode)
        flow_cfg = _flow_config(config)
        self.horizon = int(flow_cfg.get("horizon", 10))
        self.num_candidates = int(flow_cfg.get("num_candidates", 16))
        self.last_trajectories: np.ndarray | None = None
        self.last_selected: np.ndarray | None = None
        self.last_score: TrajectoryScore | None = None
        self.last_diagnostics: pd.DataFrame | None = None

    def select_action(self, env: Any, coarse_action: int | None = None, agent: Any | None = None) -> int:
        """Generate, score, and convert the selected trajectory to one action."""

        if coarse_action is None:
            coarse_action = _coarse_action_from_agent(env, agent)
        condition = env.get_flow_condition_vector(coarse_action=coarse_action)

        candidates = []
        if self.model is not None:
            candidates.append(
                sample_trajectories(
                    self.model,
                    condition,
                    num_samples=self.num_candidates,
                    horizon=self.horizon,
                    env=env,
                    config=self.config,
                )
            )
        heuristic_count = max(4, min(self.num_candidates, 8))
        candidates.append(heuristic_candidate_trajectories(env, heuristic_count, self.horizon, coarse_action))
        trajectories = np.concatenate(candidates, axis=0)
        trajectories = env.project_trajectory(trajectories)

        use_full_terms = self.mode in {"full", "flow_matching_full", "dqn_fim_stackelberg_lyapunov"}
        selected, score, diagnostics = select_best_trajectory(
            env,
            trajectories,
            config=self.config,
            use_fim=use_full_terms or self.mode == "flow_matching",
            use_stackelberg=use_full_terms,
            use_lyapunov=use_full_terms or "lyapunov" in self.mode,
        )
        self.last_trajectories = trajectories
        self.last_selected = selected
        self.last_score = score
        self.last_diagnostics = diagnostics
        return env.trajectory_to_action(selected)


def _flow_config(config: Dict[str, Any] | None) -> Dict[str, Any]:
    config = config or {}
    return dict(config.get("flow_matching", config if "horizon" in config else {}))


def _env_config(config: Dict[str, Any] | None) -> Dict[str, Any]:
    config = config or {}
    return dict(config.get("environment", {}))


def _torch_device(requested: Any) -> torch.device:
    requested = str(requested).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _load_dataset(dataset: Dict[str, np.ndarray] | str | Path) -> Dict[str, np.ndarray]:
    if isinstance(dataset, (str, Path)):
        with np.load(dataset) as data:
            return {key: data[key] for key in data.files}
    return dataset


def _initial_noise_trajectories(
    start_position: np.ndarray,
    num_samples: int,
    horizon: int,
    area_size: float,
    depth: float,
    noise_scale: float,
) -> np.ndarray:
    start = np.asarray(start_position, dtype=np.float32)
    random_steps = np.random.normal(0.0, float(noise_scale), size=(int(num_samples), int(horizon), 3)).astype(np.float32)
    random_steps[..., 2] = 0.0
    trajectories = start[None, None, :] + np.cumsum(random_steps, axis=1) / max(float(horizon), 1.0)
    trajectories[..., 2] = float(depth)
    return project_trajectories(trajectories, area_size=area_size, depth=depth)


def _start_from_condition(condition: np.ndarray) -> np.ndarray:
    condition = np.asarray(condition, dtype=float).ravel()
    if condition.size >= 9:
        return condition[6:9].copy()
    return np.array([200.0, 200.0, -120.0], dtype=float)


def _coarse_action_from_agent(env: Any, agent: Any | None) -> int:
    if agent is not None:
        return int(agent.select_action(env.get_state(), epsilon=0.0))
    return int(env.greedy_action_toward_target())


def _coarse_direction(env: Any, coarse_action: int | None) -> np.ndarray:
    if coarse_action is None:
        coarse_action = env.greedy_action_toward_target()
    direction_xy = np.asarray(env.ACTION_DIRECTIONS[int(coarse_action)], dtype=float)
    return np.array([direction_xy[0], direction_xy[1], 0.0], dtype=float)
