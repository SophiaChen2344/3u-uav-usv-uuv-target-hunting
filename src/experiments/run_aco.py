"""Run the ACO baseline for the simplified 3U target hunting environment.

The original paper reports an ACO comparison but does not release source code.
This script is therefore an approximate reproduction: it plans on a discretized
400 m x 400 m grid using pheromone and distance heuristics, then follows the
planned waypoints while the simulated target continues moving.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys
from typing import Dict

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.aco_baseline import ACOBaselinePolicy, ACOPlanner
from envs.three_u_env import ThreeUEnv
from utils.plotting import save_dataframe
from utils.seed import set_global_seed


def load_config(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _mean_finite(values, default: float = np.nan) -> float:
    values = np.asarray(list(values), dtype=float)
    if values.size == 0:
        return float(default)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float(default)
    return float(np.mean(values))


def evaluate_aco(config: Dict, episodes: int | None = None, save_outputs: bool = True) -> pd.DataFrame:
    """Evaluate the ACO baseline and save per-episode metrics."""

    config = deepcopy(config)
    seed = int(config.get("seed", 7))
    set_global_seed(seed)
    env_config = config.get("environment", {})
    aco_config = config.get("experiments", {}).get("aco", {})
    episodes = int(episodes if episodes is not None else aco_config.get("episodes", 8))

    records = []
    for episode in range(episodes):
        env = ThreeUEnv(config, seed=seed + 20_000 + episode)
        env.reset()

        planner = ACOPlanner(
            area_size=float(env_config.get("area_size", 400.0)),
            grid_size=int(aco_config.get("grid_size", 25)),
            ants=int(aco_config.get("ants", 100)),
            iterations=int(aco_config.get("iterations", 100)),
            evaporation=float(aco_config.get("evaporation", 0.2)),
            alpha=float(aco_config.get("alpha", 1.0)),
            beta=float(aco_config.get("beta", 2.5)),
            q=float(aco_config.get("q", 80.0)),
            seed=seed + episode,
        )
        policy = ACOBaselinePolicy(planner, replan_interval=int(aco_config.get("replan_interval", 10)))
        policy.reset(env)

        episode_reward = 0.0
        final_info = env.last_info
        for step in range(env.max_steps):
            action = policy.select_action(env)
            _state, reward, done, info = env.step(action)
            episode_reward += reward
            final_info = info
            if done:
                break

        energy_values = env.history.get("energy", [])
        us_distances = env.history.get("us_distance", [])
        sg_distances = env.history.get("sg_distance", [])
        connected_fractions = env.history.get("connected_fraction", [])
        target_distances = env.history.get("target_distance", [])
        lyapunov_values = env.history.get("lyapunov_value", [])
        safety_violations = env.history.get("safety_violation", [])
        action_replacements = env.history.get("action_replaced", [])
        fim_logdets = env.history.get("fim_logdet", [])
        fim_trace_invs = env.history.get("fim_trace_inv", [])
        belief_errors = env.history.get("belief_error", [])
        records.append(
            {
                "episode": episode,
                "algorithm": "ACO",
                "reward": float(episode_reward),
                "success": final_info.get("captured", 0.0),
                "captured": final_info.get("captured", 0.0),
                "failure": 1.0 - final_info.get("captured", 0.0),
                "steps": final_info.get("step", step + 1),
                "total_voyage_distance": final_info.get("total_voyage_distance", np.nan),
                "path_length": final_info.get("total_voyage_distance", np.nan),
                "motion_energy": float(np.sum(energy_values)) if energy_values else final_info.get("total_energy_used", np.nan),
                "energy_used": final_info.get("total_energy_used", np.nan),
                "avg_us_distance": float(np.mean(us_distances)) if us_distances else np.nan,
                "avg_sg_distance": float(np.mean(sg_distances)) if sg_distances else np.nan,
                "avg_connected_fraction": float(np.mean(connected_fractions)) if connected_fractions else np.nan,
                "mean_target_distance": final_info.get("mean_target_distance", np.nan),
                "avg_target_distance": float(np.mean(target_distances)) if target_distances else np.nan,
                "connected_fraction": final_info.get("connected_fraction", np.nan),
                "avg_lyapunov_value": float(np.mean(lyapunov_values)) if lyapunov_values else np.nan,
                "lyapunov_value": final_info.get("lyapunov_value", np.nan),
                "safety_violations": float(np.sum(safety_violations)) if safety_violations else 0.0,
                "action_replacements": float(np.sum(action_replacements)) if action_replacements else 0.0,
                "safety_filter_active": final_info.get("safety_filter_active", 0.0),
                "fim_logdet": final_info.get("fim_logdet", np.nan),
                "avg_fim_logdet": _mean_finite(fim_logdets),
                "fim_trace_inv": final_info.get("fim_trace_inv", np.nan),
                "avg_fim_trace_inv": _mean_finite(fim_trace_invs),
                "fim_min_eigenvalue": final_info.get("fim_min_eigenvalue", np.nan),
                "belief_error": final_info.get("belief_error", np.nan),
                "avg_belief_error": _mean_finite(belief_errors),
                "use_fim": final_info.get("use_fim", 0.0),
                "use_belief_state": final_info.get("use_belief_state", 0.0),
            }
        )

    results = pd.DataFrame.from_records(records)
    if save_outputs:
        tables_dir = Path(config.get("results", {}).get("tables", "results/tables"))
        save_dataframe(results, tables_dir / "aco_results.csv")
    return results


def print_summary(results: pd.DataFrame) -> None:
    """Print concise ACO aggregate metrics."""

    print(f"ACO success rate: {results['success'].mean():.3f}")
    print(f"ACO average voyage distance L: {results['total_voyage_distance'].mean():.3f}")
    print(f"ACO average motion energy: {results['motion_energy'].mean():.3f}")
    print(f"ACO average U-S distance: {results['avg_us_distance'].mean():.3f}")
    print(f"ACO average S-G distance: {results['avg_sg_distance'].mean():.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--episodes", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    results = evaluate_aco(config, episodes=args.episodes, save_outputs=True)
    print_summary(results)


if __name__ == "__main__":
    main()
