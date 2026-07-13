"""Main entry point for the clean 3U reproduction project."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from experiments.ablations.ablation_flow_matching import run_ablation as run_flow_matching_ablation
from experiments.baselines.run_aco import evaluate_aco
from experiments.baselines.run_dqn import train_dqn
from experiments.reproduction.reproduce_table2 import reproduce_table2
from experiments.sensitivity.compare_height import compare_uav_height
from experiments.sensitivity.compare_speed import compare_uuv_speed
from utils.plotting import ensure_output_dirs


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 3U target hunting reproduction pipeline.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--episodes", type=int, default=None, help="Override DQN training episodes for this run.")
    parser.add_argument("--no-lyapunov", action="store_true", help="Disable the Lyapunov safety filter.")
    parser.add_argument(
        "--planner",
        default="full",
        choices=["dqn", "dqn_lyapunov", "dqn_fim_stackelberg_lyapunov", "flow_matching", "full"],
        help="Planner mode. Default full mode uses Flow Matching as the primary trajectory generator.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.episodes is not None:
        config.setdefault("training", {})["episodes"] = int(args.episodes)
        config.setdefault("experiments", {}).setdefault("fig2", {})["dqn_train_episodes"] = int(args.episodes)
        config.setdefault("experiments", {}).setdefault("table2", {})["train_episodes"] = int(args.episodes)
    if args.no_lyapunov:
        config.setdefault("safety", {})["use_lyapunov"] = False

    if args.planner in {"dqn_lyapunov", "dqn_fim_stackelberg_lyapunov", "flow_matching", "full"}:
        print(f"Running planner ablation with requested planner mode: {args.planner}")
        summary = run_flow_matching_ablation(
            config,
            train_episodes=args.episodes,
            eval_episodes=config.get("experiments", {}).get("flow_matching_ablation", {}).get("eval_episodes", 5),
            save_outputs=True,
        )
        print(summary.to_string(index=False))
        print("Done. Flow Matching ablation outputs are in the results directory.")
        return

    results = config.get("results", {})
    ensure_output_dirs(
        results.get("root", "results"),
        results.get("figures", "results/figures"),
        results.get("tables", "results/tables"),
    )

    print("Training DQN controller...")
    dqn_agent, dqn_history = train_dqn(config, variant="dqn", save_outputs=True)
    print(f"  Finished {len(dqn_history)} DQN episodes.")

    print("Evaluating ACO baseline...")
    aco_results = evaluate_aco(config, save_outputs=True)
    print(f"  Finished {len(aco_results)} ACO episodes.")

    print("Running UAV-height comparison...")
    height_results = compare_uav_height(config, agent=dqn_agent, save_outputs=True)
    print(f"  Saved {len(height_results)} height settings.")

    print("Running UUV-speed comparison...")
    speed_results = compare_uuv_speed(config, agent=dqn_agent, save_outputs=True)
    print(f"  Saved {len(speed_results)} speed settings.")

    print("Producing Table-II-style aggregate comparison...")
    table2 = reproduce_table2(config, pretrained_dqn=dqn_agent, save_outputs=True)
    print(table2.to_string(index=False))

    print("Done. Figures and CSV tables are in the results directory.")


if __name__ == "__main__":
    main()
