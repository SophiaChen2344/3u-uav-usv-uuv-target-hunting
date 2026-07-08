"""Table-II-style reproduction experiment.

The original paper's Table II compares methods under different initial
UUV-target distances. This script recreates that logic with the clean-room
simulator by sweeping ``H`` and reporting energy, voyage distance, path
distances, success rate, and elapsed training/planning time.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys
import time
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.baselines.run_aco import evaluate_aco
from experiments.baselines.run_dqn import evaluate_agent, train_dqn
from utils.physics import knots_to_mps
from utils.plotting import save_dataframe


H_VALUES = [50.0, 75.0, 100.0, 125.0]
DQN_VARIANTS = [
    ("dqn", "DQN"),
    ("double_dqn", "Double DQN"),
    ("dueling_dqn", "Dueling DQN"),
]


def load_config(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _table2_config(config: Dict, initial_distance_h: float) -> Dict:
    """Apply Table-II-style default parameters and target separation."""

    run_config = deepcopy(config)
    env_config = run_config.setdefault("environment", {})
    center = float(env_config.get("area_size", 400.0)) / 2.0
    depth = -abs(float(env_config.get("target_depth", 120.0)))

    env_config["uav_height"] = 100.0
    env_config["target_speed"] = float(knots_to_mps(1.0))
    env_config["usv_speed"] = float(knots_to_mps(3.9))
    env_config["uuv_speed"] = float(knots_to_mps(7.8))
    env_config["num_uuvs"] = 3
    env_config["target_initial_position"] = [center + float(initial_distance_h), center, depth]

    agent_config = run_config.setdefault("agent", {})
    agent_config["epsilon"] = 0.9
    agent_config["epsilon_start"] = 0.9
    agent_config["epsilon_end"] = 0.9
    agent_config["epsilon_min"] = 0.9
    agent_config["gamma"] = 0.95
    agent_config["learning_rate"] = 0.001

    return run_config


def _summarize_eval(
    method: str,
    initial_distance_h: float,
    eval_df: pd.DataFrame,
    training_time_s: float,
) -> Dict[str, float | str]:
    return {
        "H": float(initial_distance_h),
        "method": method,
        "E_UUV": float(eval_df["energy_used"].mean()),
        "L": float(eval_df["path_length"].mean()),
        "avg_U_S_distance": float(eval_df["avg_us_distance"].mean()),
        "avg_S_G_distance": float(eval_df["avg_sg_distance"].mean()),
        "success_rate": float(eval_df["success"].mean()),
        "training_time_s": float(training_time_s),
    }


def _markdown_table(df: pd.DataFrame) -> str:
    display = df.copy()
    for column in ("E_UUV", "L", "avg_U_S_distance", "avg_S_G_distance", "training_time_s"):
        display[column] = display[column].map(lambda value: f"{value:.3f}")
    display["success_rate"] = display["success_rate"].map(lambda value: f"{value:.3f}")
    columns = list(display.columns)
    rows = [[str(value) for value in row] for row in display.itertuples(index=False, name=None)]
    widths = [
        max(len(str(column)), *(len(row[idx]) for row in rows)) if rows else len(str(column))
        for idx, column in enumerate(columns)
    ]
    header = "| " + " | ".join(str(column).ljust(widths[idx]) for idx, column in enumerate(columns)) + " |"
    separator = "| " + " | ".join("-" * widths[idx] for idx in range(len(columns))) + " |"
    body = [
        "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(columns))) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def reproduce_table2(
    config: Dict,
    pretrained_dqn=None,
    save_outputs: bool = True,
    h_values: Iterable[float] | None = None,
) -> pd.DataFrame:
    """Run the Table-II-style sweep and return an aggregate DataFrame."""

    del pretrained_dqn
    table_config = config.get("experiments", {}).get("table2", {})
    h_values = [float(value) for value in (h_values if h_values is not None else table_config.get("H", H_VALUES))]
    train_episodes = int(table_config.get("train_episodes", config.get("training", {}).get("table_variant_episodes", 12)))
    eval_episodes = int(table_config.get("eval_episodes", config.get("training", {}).get("eval_episodes", 6)))

    rows = []
    for h_index, initial_distance_h in enumerate(h_values):
        run_config = _table2_config(config, initial_distance_h)

        start = time.perf_counter()
        aco_df = evaluate_aco(run_config, episodes=eval_episodes, save_outputs=False)
        rows.append(_summarize_eval("ACO", initial_distance_h, aco_df, time.perf_counter() - start))

        for variant_index, (variant, label) in enumerate(DQN_VARIANTS):
            seed_offset = 70_000 + h_index * 1000 + variant_index * 100
            start = time.perf_counter()
            agent, _history = train_dqn(
                run_config,
                variant=variant,
                episodes=train_episodes,
                save_outputs=False,
                seed_offset=seed_offset,
            )
            eval_df = evaluate_agent(run_config, agent, episodes=eval_episodes, seed_offset=seed_offset + 500)
            rows.append(_summarize_eval(label, initial_distance_h, eval_df, time.perf_counter() - start))

    result = pd.DataFrame.from_records(rows)
    result = result[
        [
            "H",
            "method",
            "E_UUV",
            "L",
            "avg_U_S_distance",
            "avg_S_G_distance",
            "success_rate",
            "training_time_s",
        ]
    ]

    if save_outputs:
        tables_dir = Path(config.get("results", {}).get("tables", "results/tables"))
        tables_dir.mkdir(parents=True, exist_ok=True)
        save_dataframe(result, tables_dir / "table2_reproduction.csv")
        markdown = _markdown_table(result)
        (tables_dir / "table2_reproduction.md").write_text(markdown + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--train-episodes", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    config.setdefault("experiments", {}).setdefault("table2", {})
    if args.train_episodes is not None:
        config["experiments"]["table2"]["train_episodes"] = args.train_episodes
    if args.eval_episodes is not None:
        config["experiments"]["table2"]["eval_episodes"] = args.eval_episodes

    result = reproduce_table2(config, save_outputs=True)
    markdown = _markdown_table(result)
    print(markdown)


if __name__ == "__main__":
    main()
