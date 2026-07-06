"""Plotting and table output helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_output_dirs(*paths: str | Path) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def save_training_curve(rewards: Sequence[float], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(rewards, color="#2b6cb0", linewidth=1.6, label="Episode reward")
    if len(rewards) >= 8:
        rolling = pd.Series(rewards).rolling(window=8, min_periods=1).mean()
        ax.plot(rolling, color="#dd6b20", linewidth=2.0, label="Rolling mean")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("DQN Training Reward")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_energy_comparison(labels: Sequence[str], energy_values: Sequence[float], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, energy_values, color=["#2b6cb0", "#38a169", "#805ad5", "#d69e2e"][: len(labels)])
    ax.set_ylabel("Mean energy used (J)")
    ax.set_title("Energy Consumption Comparison")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_grouped_metric_curve(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    group_column: str,
    output_path: str | Path,
    title: str,
    ylabel: str,
    xlabel: str = "Episode",
) -> None:
    """Save one line per group for an episode-level metric."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for group_name, group in df.groupby(group_column):
        group = group.sort_values(x_column)
        ax.plot(group[x_column], group[y_column], linewidth=1.7, label=str(group_name))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_height_plot(df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(7.5, 4.6))
    ax1.plot(df["uav_height"], df["mean_energy"], marker="o", color="#2b6cb0", label="Energy")
    ax1.set_xlabel("UAV height (m)")
    ax1.set_ylabel("Mean energy used (J)", color="#2b6cb0")
    ax1.tick_params(axis="y", labelcolor="#2b6cb0")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(df["uav_height"], df["capture_rate"], marker="s", color="#c53030", label="Capture rate")
    ax2.set_ylabel("Capture rate", color="#c53030")
    ax2.tick_params(axis="y", labelcolor="#c53030")
    ax2.set_ylim(0.0, 1.05)

    fig.suptitle("UAV Height Sensitivity")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_speed_plot(df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(7.5, 4.6))
    ax1.plot(df["uuv_speed"], df["mean_target_distance"], marker="o", color="#2b6cb0")
    ax1.set_xlabel("UUV speed (m/s)")
    ax1.set_ylabel("Mean target distance (m)", color="#2b6cb0")
    ax1.tick_params(axis="y", labelcolor="#2b6cb0")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(df["uuv_speed"], df["mean_connectivity"], marker="s", color="#2f855a")
    ax2.set_ylabel("Mean connected fraction", color="#2f855a")
    ax2.tick_params(axis="y", labelcolor="#2f855a")
    ax2.set_ylim(0.0, 1.05)

    fig.suptitle("UUV Speed, Distance, and Connectivity")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_dataframe(df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def save_flow_trajectory_plot(env, trajectories: np.ndarray, output_path: str | Path, selected: np.ndarray | None = None) -> None:
    """Plot generated UUV-center trajectory proposals in the search area."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trajectories = np.asarray(trajectories, dtype=float)

    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    for trajectory in trajectories[:32]:
        ax.plot(trajectory[:, 0], trajectory[:, 1], color="#8ecae6", alpha=0.35, linewidth=1.0)
    if selected is not None:
        selected = np.asarray(selected, dtype=float)
        ax.plot(selected[:, 0], selected[:, 1], color="#d00000", linewidth=2.2, label="selected")
    if getattr(env, "state", None) is not None:
        ax.scatter(env.state.uuv_center[0], env.state.uuv_center[1], color="#023047", s=48, label="UUV center")
        ax.scatter(env.state.target_position[0], env.state.target_position[1], color="#ffb703", s=48, label="target")
        ax.scatter(env.state.usv_position[0], env.state.usv_position[1], color="#2a9d8f", s=38, label="USV")
    ax.set_xlim(0, float(getattr(env, "area_size", 400.0)))
    ax.set_ylim(0, float(getattr(env, "area_size", 400.0)))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Flow Matching Candidate UUV Trajectories")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_flow_ablation_plot(summary: pd.DataFrame, output_path: str | Path) -> None:
    """Save a compact metric panel for Flow Matching ablations."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("success_rate", "Success rate"),
        ("average_energy", "Average energy"),
        ("average_capture_time", "Capture time"),
        ("safety_violations", "Safety violations"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0))
    labels = summary["method"].tolist()
    colors = ["#457b9d", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"][: len(labels)]
    for ax, (column, title) in zip(axes.ravel(), metrics):
        values = summary[column].to_numpy(dtype=float) if column in summary else np.zeros(len(labels))
        ax.bar(labels, values, color=colors)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=18)
    fig.suptitle("Flow Matching Planner Ablation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_smoothness_plot(summary: pd.DataFrame, output_path: str | Path) -> None:
    """Plot average trajectory smoothness for planner variants."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    labels = summary["method"].tolist()
    values = summary["average_trajectory_smoothness"].to_numpy(dtype=float)
    ax.bar(labels, values, color="#6d597a")
    ax.set_ylabel("Mean squared acceleration")
    ax.set_title("Trajectory Smoothness")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
