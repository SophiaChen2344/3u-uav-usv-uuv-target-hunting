"""Train the lightweight Conditional Flow Matching trajectory generator."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys
from typing import Dict

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.three_u_env import ThreeUEnv
from generative.flow_matching import sample_trajectories, train_flow_matching
from generative.trajectory_dataset import build_trajectory_dataset, load_dataset
from utils.plotting import save_dataframe, save_flow_trajectory_plot
from utils.seed import set_global_seed


def load_config(path: str | Path = "configs/default.yaml") -> Dict:
    """Load YAML configuration."""

    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def run_training(
    config: Dict,
    dataset_path: str | Path | None = None,
    regenerate_dataset: bool = False,
) -> tuple[object, pd.DataFrame]:
    """Build/load a trajectory dataset and train the Flow Matching model."""

    config = deepcopy(config)
    set_global_seed(int(config.get("seed", 7)))
    flow_cfg = config.setdefault("flow_matching", {})
    results = config.get("results", {})
    datasets_dir = Path(results.get("datasets", "results/datasets"))
    figures_dir = Path(results.get("figures", "results/figures"))
    tables_dir = Path(results.get("tables", "results/tables"))
    checkpoints_dir = Path(results.get("checkpoints", "results/checkpoints"))
    datasets_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(dataset_path or datasets_dir / "trajectory_dataset.npz")
    if regenerate_dataset or not dataset_path.exists():
        env = ThreeUEnv(config, seed=int(config.get("seed", 7)) + 21_000)
        dataset = build_trajectory_dataset(config, env=env, save_path=dataset_path)
    else:
        dataset = load_dataset(dataset_path)

    flow_cfg["checkpoint_path"] = str(checkpoints_dir / "flow_matching_final.pt")
    model, history = train_flow_matching(dataset, config)
    save_dataframe(history, tables_dir / "flow_matching_training.csv")

    env = ThreeUEnv(config, seed=int(config.get("seed", 7)) + 22_000)
    env.reset()
    condition = env.get_flow_condition_vector(coarse_action=env.greedy_action_toward_target())
    trajectories = sample_trajectories(
        model,
        condition,
        num_samples=int(flow_cfg.get("num_candidates", 16)),
        horizon=int(flow_cfg.get("horizon", 10)),
        env=env,
        config=config,
    )
    save_flow_trajectory_plot(env, trajectories, figures_dir / "flow_matching_trajectories.png")
    return model, history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Conditional Flow Matching trajectory proposals.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--regenerate-dataset", action="store_true")
    parser.add_argument("--dataset-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.dataset_size is not None:
        config.setdefault("flow_matching", {})["dataset_size"] = int(args.dataset_size)
    if args.epochs is not None:
        config.setdefault("flow_matching", {})["epochs"] = int(args.epochs)
    _model, history = run_training(config, dataset_path=args.dataset, regenerate_dataset=args.regenerate_dataset)
    print(history.tail(1).to_string(index=False))


if __name__ == "__main__":
    main()
