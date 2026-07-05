"""Ablation study for the Lyapunov-inspired safety filter."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys
import time
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_dqn import evaluate_agent, train_dqn
from utils.plotting import save_dataframe


def load_config(path: str | Path = "configs/default.yaml") -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _output_dirs(config: Dict) -> Tuple[Path, Path]:
    results = config.get("results", {})
    figures_dir = Path(results.get("figures", "results/figures"))
    tables_dir = Path(results.get("tables", "results/tables"))
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir, tables_dir


def _mean_column(df: pd.DataFrame, column: str, fallback: str | None = None) -> float:
    if column in df:
        return float(df[column].mean())
    if fallback is not None and fallback in df:
        return float(df[fallback].mean())
    return float("nan")


def _sum_column(df: pd.DataFrame, column: str) -> float:
    if column not in df:
        return 0.0
    return float(df[column].sum())


def _plot_lyapunov_curve(histories: Dict[str, pd.DataFrame], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for label, history in histories.items():
        if "avg_lyapunov_value" not in history:
            continue
        ax.plot(
            history["episode"],
            history["avg_lyapunov_value"],
            linewidth=1.8,
            marker="o",
            markersize=3.5,
            label=label,
        )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Lyapunov value")
    ax.set_title("Lyapunov Value During DQN Training")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_ablation(summary: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("success_rate", "Success rate"),
        ("average_energy", "Average energy (J)"),
        ("safety_violations", "Safety violations"),
        ("average_connectivity", "Average connectivity"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0))
    labels = summary["method"].tolist()
    colors = ["#4c78a8", "#59a14f"][: len(labels)]
    for ax, (column, title) in zip(axes.ravel(), metrics):
        ax.bar(labels, summary[column], color=colors)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=12)
    fig.suptitle("DQN Lyapunov Safety Ablation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_ablation(
    config: Dict,
    train_episodes: int | None = None,
    eval_episodes: int | None = None,
    save_outputs: bool = True,
) -> pd.DataFrame:
    """Compare DQN with and without the Lyapunov action filter."""

    config = deepcopy(config)
    ablation_config = config.get("experiments", {}).get("lyapunov_ablation", {})
    train_episodes = int(
        train_episodes
        if train_episodes is not None
        else ablation_config.get("train_episodes", config.get("training", {}).get("episodes", 12))
    )
    eval_episodes = int(
        eval_episodes
        if eval_episodes is not None
        else ablation_config.get("eval_episodes", config.get("training", {}).get("eval_episodes", 6))
    )

    histories: Dict[str, pd.DataFrame] = {}
    records = []
    settings = [
        ("DQN without Lyapunov", False, 70_000),
        ("DQN with Lyapunov", True, 71_000),
    ]

    for method, use_lyapunov, seed_offset in settings:
        run_config = deepcopy(config)
        run_config.setdefault("safety", {})["use_lyapunov"] = bool(use_lyapunov)
        run_config.setdefault("training", {})["episodes"] = train_episodes

        start_time = time.perf_counter()
        agent, history = train_dqn(
            run_config,
            variant="dqn",
            episodes=train_episodes,
            save_outputs=False,
            seed_offset=seed_offset,
        )
        training_time = time.perf_counter() - start_time
        eval_df = evaluate_agent(run_config, agent, episodes=eval_episodes, seed_offset=seed_offset + 1000)

        histories[method] = history.assign(method=method, use_lyapunov=use_lyapunov)
        records.append(
            {
                "method": method,
                "use_lyapunov": bool(use_lyapunov),
                "train_episodes": train_episodes,
                "eval_episodes": eval_episodes,
                "success_rate": _mean_column(eval_df, "success"),
                "average_energy": _mean_column(eval_df, "energy_used"),
                "safety_violations": _sum_column(eval_df, "safety_violations"),
                "average_connectivity": _mean_column(eval_df, "avg_connected_fraction", "connected_fraction"),
                "average_target_distance": _mean_column(eval_df, "avg_target_distance", "mean_target_distance"),
                "average_lyapunov_value": _mean_column(eval_df, "avg_lyapunov_value", "lyapunov_value"),
                "average_action_replacements": _mean_column(eval_df, "action_replacements"),
                "training_time_s": float(training_time),
            }
        )

    summary = pd.DataFrame.from_records(records)

    if save_outputs:
        figures_dir, tables_dir = _output_dirs(config)
        save_dataframe(summary, tables_dir / "ablation_lyapunov.csv")
        _plot_lyapunov_curve(histories, figures_dir / "lyapunov_curve.png")
        _plot_ablation(summary, figures_dir / "lyapunov_ablation.png")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--train-episodes", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    summary = run_ablation(
        config,
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        save_outputs=True,
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

