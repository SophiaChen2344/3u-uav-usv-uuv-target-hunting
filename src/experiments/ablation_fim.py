"""Ablation study for noisy belief state and FIM-aware reward shaping."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys
import time
from typing import Dict

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_dqn import evaluate_agent, train_dqn
from utils.plotting import save_dataframe, save_grouped_metric_curve


ABLATION_VARIANTS = [
    {
        "label": "true_target",
        "description": "DQN with true target state",
        "use_belief_state": False,
        "use_information_reward": False,
        "use_lyapunov": False,
    },
    {
        "label": "belief_only",
        "description": "DQN with noisy belief state only",
        "use_belief_state": True,
        "use_information_reward": False,
        "use_lyapunov": False,
    },
    {
        "label": "belief_fim",
        "description": "DQN with noisy belief and FIM reward",
        "use_belief_state": True,
        "use_information_reward": True,
        "use_lyapunov": False,
    },
    {
        "label": "belief_fim_lyapunov",
        "description": "DQN with noisy belief, FIM reward, and Lyapunov filter",
        "use_belief_state": True,
        "use_information_reward": True,
        "use_lyapunov": True,
    },
]


def load_config(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _variant_config(config: Dict, variant: Dict) -> Dict:
    run_config = deepcopy(config)
    sensing = run_config.setdefault("sensing", {})
    default_info_weight = float(config.get("sensing", {}).get("info_reward_weight", 0.01))

    # FIM is computed for diagnostics in every variant. The reward weight is set
    # to zero for the baselines that should not optimize information.
    sensing["use_fim"] = True
    sensing["use_belief_state"] = bool(variant["use_belief_state"])
    sensing["info_reward_weight"] = default_info_weight if variant["use_information_reward"] else 0.0

    run_config.setdefault("safety", {})["use_lyapunov"] = bool(variant["use_lyapunov"])
    return run_config


def _summarize_variant(label: str, description: str, eval_df: pd.DataFrame, training_time_s: float) -> Dict:
    return {
        "variant": label,
        "description": description,
        "success_rate": float(eval_df["success"].mean()),
        "average_belief_error": float(eval_df["avg_belief_error"].mean()),
        "average_fim_logdet": float(eval_df["avg_fim_logdet"].mean()),
        "average_energy": float(eval_df["energy_used"].mean()),
        "safety_violations": float(eval_df["safety_violations"].mean()),
        "connectivity": float(eval_df["avg_connected_fraction"].mean()),
        "training_time_s": float(training_time_s),
    }


def run_fim_ablation(config: Dict, save_outputs: bool = True) -> pd.DataFrame:
    """Train and evaluate the four FIM/belief ablation variants."""

    ablation_config = config.get("experiments", {}).get("fim_ablation", {})
    train_episodes = int(ablation_config.get("train_episodes", config.get("training", {}).get("episodes", 12)))
    eval_episodes = int(ablation_config.get("eval_episodes", config.get("training", {}).get("eval_episodes", 6)))

    rows = []
    training_curves = []
    for index, variant in enumerate(ABLATION_VARIANTS):
        run_config = _variant_config(config, variant)
        seed_offset = 90_000 + index * 1_000

        start = time.perf_counter()
        agent, history = train_dqn(
            run_config,
            variant="dqn",
            episodes=train_episodes,
            save_outputs=False,
            seed_offset=seed_offset,
        )
        training_time_s = time.perf_counter() - start
        eval_df = evaluate_agent(run_config, agent, episodes=eval_episodes, seed_offset=seed_offset + 500)

        rows.append(_summarize_variant(variant["label"], variant["description"], eval_df, training_time_s))

        curve = history.copy()
        curve["ablation_variant"] = variant["label"]
        training_curves.append(curve)

    result = pd.DataFrame.from_records(rows)

    if save_outputs:
        results = config.get("results", {})
        tables_dir = Path(results.get("tables", "results/tables"))
        figures_dir = Path(results.get("figures", "results/figures"))
        save_dataframe(result, tables_dir / "ablation_fim.csv")

        if training_curves:
            curves = pd.concat(training_curves, ignore_index=True)
            save_grouped_metric_curve(
                curves,
                x_column="episode",
                y_column="avg_fim_logdet",
                group_column="ablation_variant",
                output_path=figures_dir / "fim_logdet_curve.png",
                title="FIM Log-Determinant During Training",
                ylabel="Average FIM logdet",
            )
            save_grouped_metric_curve(
                curves,
                x_column="episode",
                y_column="avg_belief_error",
                group_column="ablation_variant",
                output_path=figures_dir / "belief_error_curve.png",
                title="Belief Error During Training",
                ylabel="Average belief error (m)",
            )

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--train-episodes", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    config.setdefault("experiments", {}).setdefault("fim_ablation", {})
    if args.train_episodes is not None:
        config["experiments"]["fim_ablation"]["train_episodes"] = int(args.train_episodes)
    if args.eval_episodes is not None:
        config["experiments"]["fim_ablation"]["eval_episodes"] = int(args.eval_episodes)

    result = run_fim_ablation(config, save_outputs=True)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
