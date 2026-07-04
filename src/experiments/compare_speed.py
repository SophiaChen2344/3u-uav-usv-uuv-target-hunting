"""Fig. 2-style UUV-speed sensitivity experiment.

Experiment B fixes UAV height at 100 m, varies UUV group speed from 3.9 to
27.3 knots, and compares ACO with DQN trained at two learning rates.
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

from experiments.compare_height import (
    DQN_LEARNING_RATES,
    _dqn_eval_for_setting,
    _save_distance_plot,
    _save_energy_plot,
    _summarize,
    _summarize_distances,
    _write_combined_fig3_results,
)
from experiments.run_aco import evaluate_aco
from utils.physics import knots_to_mps
from utils.plotting import save_dataframe


FIXED_UAV_HEIGHT_M = 100.0


def load_config(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _speed_values(config: Dict, speeds: Iterable[float] | None = None) -> list[float]:
    if speeds is not None:
        return [float(value) for value in speeds]
    fig2_config = config.get("experiments", {}).get("fig2", {})
    configured = fig2_config.get("speeds_knots")
    if configured is not None:
        return [float(value) for value in configured]
    return [3.9, 7.8, 11.7, 15.6, 19.5, 23.4, 27.3]


def _prepare_speed_config(config: Dict, speed_knots: float) -> Dict:
    run_config = deepcopy(config)
    env_config = run_config.setdefault("environment", {})
    env_config["uav_height"] = FIXED_UAV_HEIGHT_M
    env_config["uuv_speed"] = float(knots_to_mps(speed_knots))
    return run_config


def compare_uuv_speed(
    config: Dict,
    agent=None,
    speeds: Iterable[float] | None = None,
    save_outputs: bool = True,
) -> pd.DataFrame:
    """Run Experiment B and return the summary table."""

    del agent
    fig2_config = config.get("experiments", {}).get("fig2", {})
    train_episodes = int(fig2_config.get("dqn_train_episodes", config.get("training", {}).get("episodes", 35)))
    eval_episodes = int(fig2_config.get("eval_episodes", config.get("training", {}).get("eval_episodes", 6)))
    speed_values = _speed_values(config, speeds)

    raw_records = []
    for speed_knots in speed_values:
        run_config = _prepare_speed_config(config, speed_knots)

        aco_df = evaluate_aco(run_config, episodes=eval_episodes, save_outputs=False)
        for _, row in aco_df.iterrows():
            raw_records.append(
                {
                    "experiment": "speed",
                    "uuv_speed_knots": float(speed_knots),
                    "uuv_speed_mps": float(knots_to_mps(speed_knots)),
                    "uav_height_m": FIXED_UAV_HEIGHT_M,
                    "algorithm": "ACO",
                    "episode": int(row["episode"]),
                    "energy_used": float(row["energy_used"]),
                    "search_time": float(row["steps"]) * float(run_config.get("environment", {}).get("dt", 1.0)),
                    "success": float(row["success"]),
                    "path_length": float(row["path_length"]),
                    "avg_us_distance": float(row["avg_us_distance"]),
                    "avg_sg_distance": float(row["avg_sg_distance"]),
                }
            )

        for learning_rate in DQN_LEARNING_RATES:
            eval_df = _dqn_eval_for_setting(
                run_config,
                learning_rate=learning_rate,
                train_episodes=train_episodes,
                eval_episodes=eval_episodes,
                seed_offset=int(50_000 + speed_knots * 100 + learning_rate * 100_000),
            )
            algorithm = f"DQN lr={learning_rate:g}"
            for _, row in eval_df.iterrows():
                raw_records.append(
                    {
                        "experiment": "speed",
                        "uuv_speed_knots": float(speed_knots),
                        "uuv_speed_mps": float(knots_to_mps(speed_knots)),
                        "uav_height_m": FIXED_UAV_HEIGHT_M,
                        "algorithm": algorithm,
                        "episode": int(row["episode"]),
                        "energy_used": float(row["energy_used"]),
                        "search_time": float(row["steps"]) * float(run_config.get("environment", {}).get("dt", 1.0)),
                        "success": float(row["success"]),
                        "path_length": float(row["path_length"]),
                        "avg_us_distance": float(row.get("avg_us_distance", row["us_distance"])),
                        "avg_sg_distance": float(row.get("avg_sg_distance", row["sg_distance"])),
                    }
                )

    raw_df = pd.DataFrame.from_records(raw_records)
    summary = _summarize(raw_df, "uuv_speed_knots")
    distance_summary = _summarize_distances(raw_df, "uuv_speed_knots")

    if save_outputs:
        results = config.get("results", {})
        tables_dir = Path(results.get("tables", "results/tables"))
        figures_dir = Path(results.get("figures", "results/figures"))
        save_dataframe(raw_df, tables_dir / "fig2_speed_raw_results.csv")
        save_dataframe(summary, tables_dir / "fig2_speed_summary.csv")
        save_dataframe(summary, tables_dir / "speed_comparison.csv")
        save_dataframe(distance_summary.assign(experiment="speed"), tables_dir / "fig3_speed_summary.csv")
        _write_combined_fig3_results(tables_dir, distance_summary, "speed")
        _save_energy_plot(
            summary,
            x_column="uuv_speed_knots",
            output_path=figures_dir / "fig2_energy_vs_speed.png",
            xlabel="UUV speed V_G (knots)",
        )
        _save_distance_plot(
            distance_summary,
            x_column="uuv_speed_knots",
            output_path=figures_dir / "fig3_distance_vs_speed.png",
            xlabel="UUV speed V_G (knots)",
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
    summary = compare_uuv_speed(config, save_outputs=True)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
