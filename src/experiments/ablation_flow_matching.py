"""Ablation study for Conditional Flow Matching trajectory proposals."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys
import time
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.three_u_env import ThreeUEnv
from experiments.run_dqn import train_dqn
from generative.flow_matching import FlowMatchingPlanner, train_flow_matching, trajectory_smoothness
from generative.trajectory_dataset import build_trajectory_dataset
from utils.plotting import (
    save_dataframe,
    save_flow_ablation_plot,
    save_flow_trajectory_plot,
    save_smoothness_plot,
)
from utils.seed import set_global_seed


def load_config(path: str | Path = "configs/default.yaml") -> Dict:
    """Load YAML configuration."""

    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _output_dirs(config: Dict) -> Tuple[Path, Path, Path, Path]:
    results = config.get("results", {})
    figures_dir = Path(results.get("figures", "results/figures"))
    tables_dir = Path(results.get("tables", "results/tables"))
    datasets_dir = Path(results.get("datasets", "results/datasets"))
    checkpoints_dir = Path(results.get("checkpoints", "results/checkpoints"))
    for path in (figures_dir, tables_dir, datasets_dir, checkpoints_dir):
        path.mkdir(parents=True, exist_ok=True)
    return figures_dir, tables_dir, datasets_dir, checkpoints_dir


def _planner_config(config: Dict, use_lyapunov: bool, use_fim: bool, use_stackelberg: bool) -> Dict:
    run_config = deepcopy(config)
    run_config.setdefault("safety", {})["use_lyapunov"] = bool(use_lyapunov)
    run_config.setdefault("sensing", {})["use_fim"] = bool(use_fim)
    run_config.setdefault("sensing", {})["use_belief_state"] = bool(use_fim)
    run_config.setdefault("game", {})["use_stackelberg"] = bool(use_stackelberg)
    run_config.setdefault("game", {})["use_intelligent_target"] = bool(use_stackelberg)
    return run_config


def evaluate_planner(
    config: Dict,
    planner_mode: str,
    agent: Any | None = None,
    flow_model: Any | None = None,
    episodes: int = 5,
    seed_offset: int = 90_000,
) -> tuple[pd.DataFrame, FlowMatchingPlanner | None, ThreeUEnv | None]:
    """Evaluate one planner mode for a small number of episodes."""

    records = []
    last_planner = None
    last_env = None
    for episode in range(int(episodes)):
        env = ThreeUEnv(config, seed=int(config.get("seed", 7)) + seed_offset + episode)
        state = env.reset()
        positions = [env.state.uuv_center.copy()]
        episode_reward = 0.0
        final_info = env.last_info
        planner = None
        if planner_mode in {"flow_matching", "full"}:
            planner = FlowMatchingPlanner(flow_model, config=config, mode=planner_mode)

        start_time = time.perf_counter()
        for step in range(env.max_steps):
            if planner_mode == "dqn":
                action = int(agent.select_action(state, epsilon=0.0)) if agent is not None else env.greedy_action_toward_target()
            elif planner_mode == "dqn_lyapunov":
                action = int(agent.select_action(state, epsilon=0.0)) if agent is not None else env.greedy_action_toward_target()
            elif planner_mode == "dqn_fim_stackelberg_lyapunov":
                action = int(agent.select_action(state, epsilon=0.0)) if agent is not None else env.greedy_action_toward_target()
            elif planner_mode in {"flow_matching", "full"}:
                action = int(planner.select_action(env, agent=agent if planner_mode == "full" else None))
            else:
                raise ValueError(f"Unknown planner mode: {planner_mode}")

            state, reward, done, info = env.step(action)
            positions.append(env.state.uuv_center.copy())
            episode_reward += reward
            final_info = info
            if done:
                break
        runtime = time.perf_counter() - start_time

        if planner is not None and planner.last_score is not None:
            smoothness = float(planner.last_score.smoothness)
        else:
            smoothness = trajectory_smoothness(np.asarray(positions, dtype=float))
        fim_trace_inv = _mean_finite(env.history.get("fim_trace_inv", []), default=final_info.get("fim_trace_inv_proxy", np.nan))
        records.append(
            {
                "episode": episode,
                "planner": planner_mode,
                "reward": float(episode_reward),
                "success": float(final_info.get("captured", 0.0)),
                "captured": float(final_info.get("captured", 0.0)),
                "energy": float(final_info.get("total_energy_used", np.nan)),
                "capture_time": float(final_info.get("step", step + 1)),
                "trajectory_smoothness": smoothness,
                "fim_trace_inv": fim_trace_inv,
                "safety_violations": _sum_finite(env.history.get("safety_violation", [])),
                "connectivity": _mean_finite(env.history.get("connected_fraction", []), default=final_info.get("connected_fraction", np.nan)),
                "runtime_s": float(runtime),
            }
        )
        last_planner = planner or last_planner
        last_env = env
    return pd.DataFrame.from_records(records), last_planner, last_env


def run_ablation(
    config: Dict,
    train_episodes: int | None = None,
    eval_episodes: int | None = None,
    dataset_size: int | None = None,
    flow_epochs: int | None = None,
    save_outputs: bool = True,
) -> pd.DataFrame:
    """Compare DQN, safety/game variants, and Flow Matching planners."""

    config = deepcopy(config)
    set_global_seed(int(config.get("seed", 7)))
    figures_dir, tables_dir, datasets_dir, checkpoints_dir = _output_dirs(config)
    ablation_cfg = dict(config.get("experiments", {}).get("flow_matching_ablation", {}))
    train_episodes = int(train_episodes or ablation_cfg.get("train_episodes", 8))
    eval_episodes = int(eval_episodes or ablation_cfg.get("eval_episodes", 5))

    base_dqn_config = _planner_config(config, use_lyapunov=False, use_fim=False, use_stackelberg=False)
    safe_dqn_config = _planner_config(config, use_lyapunov=True, use_fim=False, use_stackelberg=False)
    full_dqn_config = _planner_config(config, use_lyapunov=True, use_fim=True, use_stackelberg=True)
    for run_config in (base_dqn_config, safe_dqn_config, full_dqn_config):
        run_config.setdefault("training", {})["episodes"] = train_episodes
        run_config.setdefault("training", {})["progress_bar"] = False

    dqn_start = time.perf_counter()
    dqn_agent, _dqn_history = train_dqn(base_dqn_config, episodes=train_episodes, save_outputs=False, seed_offset=80_000)
    dqn_training_time = time.perf_counter() - dqn_start

    safe_start = time.perf_counter()
    safe_agent, _safe_history = train_dqn(safe_dqn_config, episodes=train_episodes, save_outputs=False, seed_offset=81_000)
    safe_training_time = time.perf_counter() - safe_start

    full_start = time.perf_counter()
    full_agent, _full_history = train_dqn(full_dqn_config, episodes=train_episodes, save_outputs=False, seed_offset=82_000)
    full_training_time = time.perf_counter() - full_start

    flow_config = deepcopy(config)
    flow_config.setdefault("flow_matching", {})
    flow_config["flow_matching"]["dataset_size"] = int(dataset_size or ablation_cfg.get("dataset_size", 512))
    flow_config["flow_matching"]["epochs"] = int(flow_epochs or ablation_cfg.get("epochs", 5))
    flow_config["flow_matching"]["checkpoint_path"] = str(checkpoints_dir / "flow_matching_ablation.pt")
    flow_config["flow_matching"]["use_heuristic_rollouts"] = True
    dataset = build_trajectory_dataset(
        flow_config,
        env=ThreeUEnv(flow_config, seed=int(flow_config.get("seed", 7)) + 83_000),
        policy=dqn_agent,
        save_path=datasets_dir / "trajectory_dataset.npz",
    )
    flow_start = time.perf_counter()
    flow_model, _flow_history = train_flow_matching(dataset, flow_config)
    flow_training_time = time.perf_counter() - flow_start

    specs = [
        ("DQN", base_dqn_config, "dqn", dqn_agent, None, dqn_training_time),
        ("DQN + Lyapunov", safe_dqn_config, "dqn_lyapunov", safe_agent, None, safe_training_time),
        (
            "DQN + FIM + Stackelberg + Lyapunov",
            full_dqn_config,
            "dqn_fim_stackelberg_lyapunov",
            full_agent,
            None,
            full_training_time,
        ),
        ("Flow Matching only", base_dqn_config, "flow_matching", None, flow_model, flow_training_time),
        (
            "Flow Matching + FIM + Stackelberg + Lyapunov",
            full_dqn_config,
            "full",
            full_agent,
            flow_model,
            flow_training_time + full_training_time,
        ),
    ]

    summaries = []
    raw_frames = []
    trajectory_plot_written = False
    for idx, (method, run_config, planner_mode, agent, model, training_time) in enumerate(specs):
        eval_df, planner, env = evaluate_planner(
            run_config,
            planner_mode=planner_mode,
            agent=agent,
            flow_model=model,
            episodes=eval_episodes,
            seed_offset=84_000 + idx * 100,
        )
        eval_df["method"] = method
        raw_frames.append(eval_df)
        summaries.append(
            {
                "method": method,
                "planner": planner_mode,
                "success_rate": float(eval_df["success"].mean()),
                "average_energy": float(eval_df["energy"].mean()),
                "average_capture_time": float(eval_df["capture_time"].mean()),
                "average_trajectory_smoothness": float(eval_df["trajectory_smoothness"].mean()),
                "average_fim_trace_inv": float(eval_df["fim_trace_inv"].mean()),
                "safety_violations": float(eval_df["safety_violations"].sum()),
                "average_connectivity": float(eval_df["connectivity"].mean()),
                "runtime_per_episode_s": float(eval_df["runtime_s"].mean()),
                "training_time_s": float(training_time),
            }
        )
        if (
            save_outputs
            and not trajectory_plot_written
            and planner is not None
            and planner.last_trajectories is not None
            and env is not None
        ):
            save_flow_trajectory_plot(
                env,
                planner.last_trajectories,
                figures_dir / "flow_matching_trajectories.png",
                selected=planner.last_selected,
            )
            trajectory_plot_written = True

    summary = pd.DataFrame.from_records(summaries)
    raw = pd.concat(raw_frames, ignore_index=True)
    if save_outputs:
        save_dataframe(summary, tables_dir / "ablation_flow_matching.csv")
        save_dataframe(raw, tables_dir / "ablation_flow_matching_episodes.csv")
        save_flow_ablation_plot(summary, figures_dir / "flow_matching_ablation.png")
        save_smoothness_plot(summary, figures_dir / "trajectory_smoothness.png")
    return summary


def _mean_finite(values, default: float = np.nan) -> float:
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float(default)
    return float(np.mean(values))


def _sum_finite(values) -> float:
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    return float(np.sum(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--train-episodes", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--dataset-size", type=int, default=None)
    parser.add_argument("--flow-epochs", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    summary = run_ablation(
        config,
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        dataset_size=args.dataset_size,
        flow_epochs=args.flow_epochs,
        save_outputs=True,
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
