# Project Map

This repository is organized around the main simulator, learning/planning
methods, safety/sensing/game extensions, and reproducible experiments.

## Core Simulator

- `src/envs/three_u_env.py` contains the UAV-USV-UUV target hunting
  environment.
- `src/utils/` contains reusable physics, energy, connectivity, plotting,
  seeding, and replay-buffer helpers.

## Controllers And Models

- `src/agents/` contains DQN-family agents and the ACO baseline planner.
- `src/models/` contains neural network modules used by the agents.
- `src/generative/` contains the Conditional Flow Matching trajectory proposal
  model and synthetic trajectory dataset builder.

## Safety, Sensing, And Game Layers

- `src/safety/` contains the Lyapunov-inspired action filter.
- `src/sensing/` contains Fisher Information Matrix and belief-state helpers.
- `src/game/` contains the one-step Stackelberg pursuit-evasion layer.

## Experiments

Experiments are grouped by purpose under `src/experiments/`:

- `baselines/`: direct ACO and DQN-family training/evaluation runs.
- `sensitivity/`: Fig. 2/Fig. 3 style height and speed sweeps.
- `reproduction/`: Table II style aggregate reproduction.
- `ablations/`: Lyapunov, FIM, Stackelberg, and Flow Matching ablation studies.

## Results

- `results/figures/` stores generated plots.
- `results/tables/` stores generated CSV and Markdown tables.
- `results/checkpoints/` stores trained model checkpoints.
- `results/datasets/` stores generated trajectory datasets.

Generated artifacts are ignored by Git except for `.gitkeep` placeholders.

