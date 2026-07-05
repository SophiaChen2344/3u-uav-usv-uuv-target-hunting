"""Fig. 2-style UAV-height sensitivity experiment.

Experiment A fixes UUV group speed at 7.8 knots, varies UAV height from 50 m
to 120 m, and compares ACO with DQN trained at two learning rates. The curves
are an approximate reproduction of the paper's experimental logic rather than
paper-exact values.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_aco import evaluate_aco
from experiments.run_dqn import evaluate_agent, train_dqn
from utils.physics import knots_to_mps
from utils.plotting import save_dataframe


DQN_LEARNING_RATES = (0.001, 0.01)
FIXED_UUV_SPEED_KNOTS = 7.8


def load_config(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _height_values(config: Dict, heights: Iterable[float] | None = None) -> list[float]:
    if heights is not None:
        return [float(value) for value in heights]
    fig2_config = config.get("experiments", {}).get("fig2", {})
    configured = fig2_config.get("heights_m")
    if configured is not None:
        return [float(value) for value in configured]
    return [50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0]


def _prepare_height_config(config: Dict, height_m: float) -> Dict:
    run_config = deepcopy(config)
    env_config = run_config.setdefault("environment", {})
    env_config["uav_height"] = float(height_m)
    env_config["uuv_speed"] = float(knots_to_mps(FIXED_UUV_SPEED_KNOTS))
    return run_config


def _dqn_eval_for_setting(
    config: Dict,
    learning_rate: float,
    train_episodes: int,
    eval_episodes: int,
    seed_offset: int,
) -> pd.DataFrame:
    run_config = deepcopy(config)
    run_config.setdefault("agent", {})["learning_rate"] = float(learning_rate)
    run_config.setdefault("training", {})["episodes"] = int(train_episodes)
    agent, _history = train_dqn(
        run_config,
        variant="dqn",
        episodes=train_episodes,
        save_outputs=False,
        seed_offset=seed_offset,
    )
    return evaluate_agent(run_config, agent, episodes=eval_episodes, seed_offset=seed_offset + 1000)


def _summarize(raw_df: pd.DataFrame, x_column: str) -> pd.DataFrame:
    grouped = raw_df.groupby([x_column, "algorithm"], as_index=False)
    return grouped.agg(
        min_energy=("energy_used", "min"),
        avg_energy=("energy_used", "mean"),
        avg_search_time=("search_time", "mean"),
        success_rate=("success", "mean"),
        avg_path_length=("path_length", "mean"),
        avg_safety_violations=("safety_violations", "mean"),
        avg_lyapunov_value=("avg_lyapunov_value", "mean"),
    )


def _summarize_distances(raw_df: pd.DataFrame, x_column: str) -> pd.DataFrame:
    grouped = raw_df.groupby([x_column, "algorithm"], as_index=False)
    return grouped.agg(
        avg_us_distance=("avg_us_distance", "mean"),
        avg_sg_distance=("avg_sg_distance", "mean"),
        avg_connected_fraction=("avg_connected_fraction", "mean"),
        avg_target_distance=("avg_target_distance", "mean"),
        avg_search_time=("search_time", "mean"),
        success_rate=("success", "mean"),
    )


def _save_energy_plot(summary: pd.DataFrame, x_column: str, output_path: Path, xlabel: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for algorithm, group in summary.groupby("algorithm"):
        group = group.sort_values(x_column)
        ax.plot(group[x_column], group["min_energy"], marker="o", linewidth=1.8, label=algorithm)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Minimum UUV energy consumption (J)")
    ax.set_title("Fig. 2-like Energy Sensitivity")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_distance_plot(summary: pd.DataFrame, x_column: str, output_path: Path, xlabel: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), sharex=True)
    for algorithm, group in summary.groupby("algorithm"):
        group = group.sort_values(x_column)
        axes[0].plot(group[x_column], group["avg_us_distance"], marker="o", linewidth=1.8, label=algorithm)
        axes[1].plot(group[x_column], group["avg_sg_distance"], marker="s", linewidth=1.8, label=algorithm)

    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("Average ||U - S|| (m)")
    axes[0].grid(True, alpha=0.25)
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("Average ||S - G|| (m)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    fig.suptitle("Fig. 3-like Path Connectivity Distances")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _write_combined_fig3_results(tables_dir: Path, new_summary: pd.DataFrame, experiment: str) -> None:
    """Update the shared Fig. 3 CSV with one experiment's distance summary."""

    output_path = tables_dir / "fig3_results.csv"
    new_summary = new_summary.copy()
    new_summary["experiment"] = experiment
    if output_path.exists():
        combined = pd.read_csv(output_path)
        combined = combined[combined["experiment"] != experiment]
        combined = pd.concat([combined, new_summary], ignore_index=True)
    else:
        combined = new_summary
    save_dataframe(combined, output_path)


def compare_uav_height(
    config: Dict,
    agent=None,
    heights: Iterable[float] | None = None,
    save_outputs: bool = True,
) -> pd.DataFrame:
    """Run Experiment A and return the summary table.

    ``agent`` is accepted for backward compatibility but this experiment trains
    fresh DQN baselines for each learning rate and height.
    """

    del agent
    fig2_config = config.get("experiments", {}).get("fig2", {})
    train_episodes = int(fig2_config.get("dqn_train_episodes", config.get("training", {}).get("episodes", 35)))
    eval_episodes = int(fig2_config.get("eval_episodes", config.get("training", {}).get("eval_episodes", 6)))
    height_values = _height_values(config, heights)

    raw_records = []
    for height_m in height_values:
        run_config = _prepare_height_config(config, height_m)

        aco_df = evaluate_aco(run_config, episodes=eval_episodes, save_outputs=False)
        for _, row in aco_df.iterrows():
            raw_records.append(
                {
                    "experiment": "height",
                    "uav_height_m": float(height_m),
                    "uuv_speed_knots": FIXED_UUV_SPEED_KNOTS,
                    "algorithm": "ACO",
                    "episode": int(row["episode"]),
                    "energy_used": float(row["energy_used"]),
                    "search_time": float(row["steps"]) * float(run_config.get("environment", {}).get("dt", 1.0)),
                    "success": float(row["success"]),
                    "path_length": float(row["path_length"]),
                    "avg_us_distance": float(row["avg_us_distance"]),
                    "avg_sg_distance": float(row["avg_sg_distance"]),
                    "avg_connected_fraction": float(row.get("avg_connected_fraction", row.get("connected_fraction", np.nan))),
                    "avg_target_distance": float(row.get("avg_target_distance", row.get("mean_target_distance", np.nan))),
                    "avg_lyapunov_value": float(row.get("avg_lyapunov_value", np.nan)),
                    "safety_violations": float(row.get("safety_violations", 0.0)),
                    "action_replacements": float(row.get("action_replacements", 0.0)),
                }
            )

        for learning_rate in DQN_LEARNING_RATES:
            eval_df = _dqn_eval_for_setting(
                run_config,
                learning_rate=learning_rate,
                train_episodes=train_episodes,
                eval_episodes=eval_episodes,
                seed_offset=int(40_000 + height_m * 10 + learning_rate * 100_000),
            )
            algorithm = f"DQN lr={learning_rate:g}"
            for _, row in eval_df.iterrows():
                raw_records.append(
                    {
                        "experiment": "height",
                        "uav_height_m": float(height_m),
                        "uuv_speed_knots": FIXED_UUV_SPEED_KNOTS,
                        "algorithm": algorithm,
                        "episode": int(row["episode"]),
                        "energy_used": float(row["energy_used"]),
                        "search_time": float(row["steps"]) * float(run_config.get("environment", {}).get("dt", 1.0)),
                        "success": float(row["success"]),
                        "path_length": float(row["path_length"]),
                        "avg_us_distance": float(row.get("avg_us_distance", row["us_distance"])),
                        "avg_sg_distance": float(row.get("avg_sg_distance", row["sg_distance"])),
                        "avg_connected_fraction": float(
                            row.get("avg_connected_fraction", row.get("connected_fraction", np.nan))
                        ),
                        "avg_target_distance": float(
                            row.get("avg_target_distance", row.get("mean_target_distance", np.nan))
                        ),
                        "avg_lyapunov_value": float(row.get("avg_lyapunov_value", np.nan)),
                        "safety_violations": float(row.get("safety_violations", 0.0)),
                        "action_replacements": float(row.get("action_replacements", 0.0)),
                    }
                )

    raw_df = pd.DataFrame.from_records(raw_records)
    summary = _summarize(raw_df, "uav_height_m")
    distance_summary = _summarize_distances(raw_df, "uav_height_m")

    if save_outputs:
        results = config.get("results", {})
        tables_dir = Path(results.get("tables", "results/tables"))
        figures_dir = Path(results.get("figures", "results/figures"))
        save_dataframe(raw_df, tables_dir / "fig2_height_raw_results.csv")
        save_dataframe(summary, tables_dir / "fig2_height_summary.csv")
        save_dataframe(summary, tables_dir / "height_comparison.csv")
        save_dataframe(distance_summary.assign(experiment="height"), tables_dir / "fig3_height_summary.csv")
        _write_combined_fig3_results(tables_dir, distance_summary, "height")
        _save_energy_plot(
            summary,
            x_column="uav_height_m",
            output_path=figures_dir / "fig2_energy_vs_height.png",
            xlabel="UAV height h (m)",
        )
        _save_distance_plot(
            distance_summary,
            x_column="uav_height_m",
            output_path=figures_dir / "fig3_distance_vs_height.png",
            xlabel="UAV height h (m)",
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--train-episodes", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    config.setdefault("experiments", {}).setdefault("fig2", {})
    if args.train_episodes is not None:
        config["experiments"]["fig2"]["dqn_train_episodes"] = args.train_episodes
    if args.eval_episodes is not None:
        config["experiments"]["fig2"]["eval_episodes"] = args.eval_episodes
    summary = compare_uav_height(config, save_outputs=True)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
