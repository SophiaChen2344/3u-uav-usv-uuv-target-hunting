"""Train and evaluate a DQN controller for the 3U target hunting environment."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from tqdm import trange

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.double_dqn_agent import DoubleDQNAgent
from agents.dqn_agent import DQNAgent
from agents.dueling_dqn_agent import DuelingDQNAgent
from envs.three_u_env import ThreeUEnv
from utils.plotting import save_dataframe, save_training_curve
from utils.seed import set_global_seed


def load_config(path: str | Path = "configs/default.yaml") -> Dict:
    """Load the YAML experiment configuration."""

    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def make_agent(variant: str, state_dim: int, action_dim: int, agent_config: Dict):
    """Create a DQN-family agent."""

    variant = normalize_agent_name(variant)
    if variant == "dqn":
        return DQNAgent(state_dim, action_dim, agent_config)
    if variant == "double_dqn":
        return DoubleDQNAgent(state_dim, action_dim, agent_config)
    if variant == "dueling_dqn":
        return DuelingDQNAgent(state_dim, action_dim, agent_config)
    raise ValueError(f"Unknown agent variant: {variant}")


def normalize_agent_name(name: str) -> str:
    """Normalize CLI aliases to the artifact-friendly agent name."""

    normalized = str(name).lower().replace("-", "_")
    aliases = {
        "ddqn": "double_dqn",
        "double": "double_dqn",
        "dueling": "dueling_dqn",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"dqn", "double_dqn", "dueling_dqn"}:
        raise ValueError(f"Unknown agent variant: {name}")
    return normalized


def epsilon_by_step(agent_config: Dict, global_step: int) -> float:
    """Return decayed epsilon, defaulting to 0.9 with optional decay."""

    start = float(agent_config.get("epsilon", agent_config.get("epsilon_start", 0.9)))
    end = float(agent_config.get("epsilon_min", agent_config.get("epsilon_end", 0.05)))
    decay_steps = agent_config.get("epsilon_decay_steps")
    if decay_steps is None:
        return start
    fraction = min(float(global_step) / max(int(decay_steps), 1), 1.0)
    return end + (start - end) * (1.0 - fraction)


def _output_dirs(config: Dict) -> Tuple[Path, Path, Path]:
    results = config.get("results", {})
    figures_dir = Path(results.get("figures", "results/figures"))
    tables_dir = Path(results.get("tables", "results/tables"))
    checkpoints_dir = Path(results.get("checkpoints", "results/checkpoints"))
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir, tables_dir, checkpoints_dir


def _save_energy_curve(energies: list[float], output_path: Path, title: str = "DQN Episode Energy") -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(energies, color="#2f855a", linewidth=1.6, label="Episode energy")
    if len(energies) >= 8:
        rolling = pd.Series(energies).rolling(window=8, min_periods=1).mean()
        ax.plot(rolling, color="#d69e2e", linewidth=2.0, label="Rolling mean")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Energy used (J)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def train_dqn(
    config: Dict,
    variant: str = "dqn",
    episodes: int | None = None,
    save_outputs: bool = True,
    seed_offset: int = 0,
) -> Tuple[object, pd.DataFrame]:
    """Train a DQN agent and return the trained agent plus episode log."""

    config = deepcopy(config)
    variant = normalize_agent_name(variant)
    seed = int(config.get("seed", 7)) + seed_offset
    set_global_seed(seed)

    env = ThreeUEnv(config, seed=seed)
    initial_state = env.reset()
    action_dim = int(getattr(env, "action_space_n", 8))
    agent_config = dict(config.get("agent", {}))
    agent_config.setdefault("replay_capacity", 10_000)
    agent_config.setdefault("batch_size", 128)
    agent_config.setdefault("gamma", 0.95)
    agent_config.setdefault("epsilon", 0.9)
    agent_config.setdefault("learning_rate", 1e-3)
    agent = make_agent(variant, len(initial_state), action_dim, agent_config)

    training_config = config.get("training", {})
    episodes = int(episodes if episodes is not None else training_config.get("episodes", 100))
    max_steps = int(training_config.get("iterations", env.max_steps))
    max_steps = min(max_steps, env.max_steps)
    progress_bar = bool(training_config.get("progress_bar", True))
    save_every = int(training_config.get("checkpoint_interval", 0))

    figures_dir, tables_dir, checkpoints_dir = _output_dirs(config)

    records = []
    global_step = 0
    iterator = trange(episodes, desc=f"training {variant}", disable=not progress_bar)
    for episode in iterator:
        state = env.reset()
        episode_reward = 0.0
        losses = []
        final_info = env.last_info

        for step in range(max_steps):
            epsilon = epsilon_by_step(agent_config, global_step)
            agent.epsilon = epsilon
            action = agent.select_action(state, epsilon=epsilon)
            next_state, reward, done, info = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            update_stats = agent.optimize_model()
            if update_stats is not None:
                losses.append(update_stats.loss)

            state = next_state
            episode_reward += reward
            final_info = info
            global_step += 1
            if done:
                break

        if "epsilon_decay_steps" not in agent_config:
            agent.decay_epsilon()

        record = {
            "episode": episode,
            "algorithm": variant,
            "reward": float(episode_reward),
            "epsilon": float(epsilon_by_step(agent_config, global_step)),
            "loss": float(np.mean(losses)) if losses else np.nan,
            "captured": final_info.get("captured", 0.0),
            "success": final_info.get("captured", 0.0),
            "steps": final_info.get("step", step + 1),
            "energy_used": final_info.get("total_energy_used", np.nan),
            "path_length": final_info.get("total_voyage_distance", np.nan),
            "us_distance": final_info.get("us_distance", final_info.get("usv_uav_distance", np.nan)),
            "sg_distance": final_info.get("sg_distance", final_info.get("mean_uuv_usv_distance", np.nan)),
            "avg_us_distance": float(np.mean(env.history.get("us_distance", [np.nan]))),
            "avg_sg_distance": float(np.mean(env.history.get("sg_distance", [np.nan]))),
            "mean_target_distance": final_info.get("mean_target_distance", np.nan),
            "connected_fraction": final_info.get("connected_fraction", np.nan),
        }
        records.append(record)

        if save_outputs and save_every > 0 and (episode + 1) % save_every == 0:
            agent.save(checkpoints_dir / f"{variant}_episode_{episode + 1}.pt")

        if progress_bar:
            iterator.set_postfix(
                reward=f"{episode_reward:.1f}",
                eps=f"{record['epsilon']:.2f}",
                success=int(record["success"]),
            )

    history = pd.DataFrame.from_records(records)

    if save_outputs:
        save_training_curve(history["reward"].tolist(), figures_dir / f"{variant}_reward_curve.png")
        if variant == "dqn":
            save_training_curve(history["reward"].tolist(), figures_dir / "training_reward_curve.png")
        _save_energy_curve(
            history["energy_used"].fillna(0.0).tolist(),
            figures_dir / f"{variant}_energy_curve.png",
            title=f"{variant.replace('_', ' ').title()} Episode Energy",
        )
        save_dataframe(history, tables_dir / f"{variant}_training.csv")
        agent.save(checkpoints_dir / f"{variant}_final.pt")

    return agent, history


def evaluate_agent(config: Dict, agent, episodes: int | None = None, seed_offset: int = 10_000) -> pd.DataFrame:
    """Evaluate an agent greedily and return per-episode metrics."""

    eval_config = deepcopy(config)
    episodes = int(episodes if episodes is not None else eval_config.get("training", {}).get("eval_episodes", 10))
    records = []
    for episode in range(episodes):
        env = ThreeUEnv(eval_config, seed=int(eval_config.get("seed", 7)) + seed_offset + episode)
        state = env.reset()
        episode_reward = 0.0
        final_info = env.last_info
        for step in range(env.max_steps):
            action = agent.select_action(state, epsilon=0.0)
            state, reward, done, info = env.step(action)
            episode_reward += reward
            final_info = info
            if done:
                break

        records.append(
            {
                "episode": episode,
                "reward": float(episode_reward),
                "captured": final_info.get("captured", 0.0),
                "success": final_info.get("captured", 0.0),
                "steps": final_info.get("step", step + 1),
                "energy_used": final_info.get("total_energy_used", np.nan),
                "path_length": final_info.get("total_voyage_distance", np.nan),
                "us_distance": final_info.get("us_distance", final_info.get("usv_uav_distance", np.nan)),
                "sg_distance": final_info.get("sg_distance", final_info.get("mean_uuv_usv_distance", np.nan)),
                "avg_us_distance": float(np.mean(env.history.get("us_distance", [np.nan]))),
                "avg_sg_distance": float(np.mean(env.history.get("sg_distance", [np.nan]))),
                "mean_target_distance": final_info.get("mean_target_distance", np.nan),
                "connected_fraction": final_info.get("connected_fraction", np.nan),
            }
        )
    return pd.DataFrame.from_records(records)


def print_summary(eval_df: pd.DataFrame) -> None:
    """Print final reproduction metrics requested by the experiment spec."""

    print(f"Final success rate: {eval_df['success'].mean():.3f}")
    print(f"Average energy: {eval_df['energy_used'].mean():.3f}")
    print(f"Average path length: {eval_df['path_length'].mean():.3f}")
    print(f"Average U-S distance: {eval_df['us_distance'].mean():.3f}")
    print(f"Average S-G distance: {eval_df['sg_distance'].mean():.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--agent", default=None, choices=["dqn", "double_dqn", "dueling_dqn"])
    parser.add_argument("--variant", default=None, choices=["dqn", "double_dqn", "dueling_dqn"])
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None, help="Override agent.learning_rate, e.g. 0.01.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.learning_rate is not None:
        config.setdefault("agent", {})["learning_rate"] = args.learning_rate
    agent_name = normalize_agent_name(args.agent or args.variant or "dqn")
    agent, _history = train_dqn(config, variant=agent_name, episodes=args.episodes, save_outputs=True)
    eval_df = evaluate_agent(config, agent, episodes=args.eval_episodes)
    print_summary(eval_df)


if __name__ == "__main__":
    main()
