"""Ablation study for rational target motion and Stackelberg planning."""

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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.baselines.run_dqn import evaluate_agent, train_dqn
from utils.plotting import save_dataframe


def load_config(path: str | Path = "configs/default.yaml") -> Dict:
    """Load a YAML configuration file."""

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


def _configure_variant(
    config: Dict,
    use_intelligent_target: bool,
    use_stackelberg: bool,
    use_fim: bool,
    use_lyapunov: bool,
) -> Dict:
    run_config = deepcopy(config)
    run_config.setdefault("game", {})["use_intelligent_target"] = bool(use_intelligent_target)
    run_config.setdefault("game", {})["use_stackelberg"] = bool(use_stackelberg)
    run_config.setdefault("game", {})["target_action_space"] = int(run_config["game"].get("target_action_space", 8))
    run_config.setdefault("sensing", {})["use_fim"] = bool(use_fim)
    run_config.setdefault("safety", {})["use_lyapunov"] = bool(use_lyapunov)
    return run_config


def _plot_distance_curve(histories: Dict[str, pd.DataFrame], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for label, history in histories.items():
        if "avg_target_distance" not in history:
            continue
        ax.plot(
            history["episode"],
            history["avg_target_distance"],
            linewidth=1.7,
            marker="o",
            markersize=3.0,
            label=label,
        )
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Average target distance (m)")
    ax.set_title("Stackelberg Ablation Target Distance")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_success_rate(summary: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    colors = ["#4c78a8", "#59a14f", "#f28e2b", "#e15759", "#76b7b2"]
    ax.bar(summary["method"], summary["success_rate"], color=colors[: len(summary)])
    ax.set_ylabel("Success rate")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Stackelberg Ablation Success Rate")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_ablation(
    config: Dict,
    train_episodes: int | None = None,
    eval_episodes: int | None = None,
    save_outputs: bool = True,
) -> pd.DataFrame:
    """Compare simple, rational, Stackelberg, FIM, and Lyapunov variants."""

    config = deepcopy(config)
    ablation_config = config.get("experiments", {}).get("stackelberg_ablation", {})
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

    settings = [
        ("Simple target escape", False, False, False, False, 80_000),
        ("Intelligent target escape", True, False, False, False, 81_000),
        ("DQN + Stackelberg", True, True, False, False, 82_000),
        ("DQN + FIM + Stackelberg", True, True, True, False, 83_000),
        ("DQN + FIM + Stackelberg + Lyapunov", True, True, True, True, 84_000),
    ]

    histories: Dict[str, pd.DataFrame] = {}
    records = []

    for method, intelligent_target, stackelberg, fim, lyapunov, seed_offset in settings:
        run_config = _configure_variant(config, intelligent_target, stackelberg, fim, lyapunov)
        run_config.setdefault("training", {})["episodes"] = train_episodes

        start = time.perf_counter()
        agent, history = train_dqn(
            run_config,
            variant="dqn",
            episodes=train_episodes,
            save_outputs=False,
            seed_offset=seed_offset,
        )
        training_time = time.perf_counter() - start

        eval_df = evaluate_agent(run_config, agent, episodes=eval_episodes, seed_offset=seed_offset + 1000)
        histories[method] = history.assign(method=method)

        records.append(
            {
                "method": method,
                "use_intelligent_target": bool(intelligent_target),
                "use_stackelberg": bool(stackelberg),
                "use_fim": bool(fim),
                "use_lyapunov": bool(lyapunov),
                "train_episodes": train_episodes,
                "eval_episodes": eval_episodes,
                "success_rate": _mean_column(eval_df, "success"),
                "average_capture_time": _mean_column(eval_df, "steps"),
                "average_target_distance": _mean_column(eval_df, "avg_target_distance", "mean_target_distance"),
                "average_energy": _mean_column(eval_df, "energy_used"),
                "average_fim_trace_inv": _mean_column(eval_df, "avg_fim_trace_inv", "fim_trace_inv"),
                "safety_violations": _sum_column(eval_df, "safety_violations"),
                "average_stackelberg_changes": _mean_column(eval_df, "stackelberg_changed_actions"),
                "average_action_replacements": _mean_column(eval_df, "action_replacements"),
                "training_time_s": float(training_time),
            }
        )

    summary = pd.DataFrame.from_records(records)
    if save_outputs:
        figures_dir, tables_dir = _output_dirs(config)
        save_dataframe(summary, tables_dir / "ablation_stackelberg.csv")
        _plot_distance_curve(histories, figures_dir / "stackelberg_distance_curve.png")
        _plot_success_rate(summary, figures_dir / "stackelberg_success_rate.png")

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
