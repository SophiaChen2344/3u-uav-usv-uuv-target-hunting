# Reproduction of 3U UAV-USV-UUV Cooperative Target Hunting with DQN

This repository provides an educational Python/PyTorch reproduction of the
main ideas from a UAV-USV-UUV cooperative underwater target hunting paper. It
implements a compact 3U simulator, energy and connectivity models, DQN-family
controllers, an Ant Colony Optimization baseline, and experiment scripts for
Fig. 2-like, Fig. 3-like, and Table II-like comparisons.

The current version also includes noisy target sensing with Fisher Information
Matrix (FIM) diagnostics, a one-step Stackelberg pursuit-evasion game for
rational target motion, a Lyapunov-inspired safety filter, and a lightweight
Conditional Flow Matching trajectory proposal module. These additions are
educational approximations, not formal proofs or official paper code.

For a quick navigation guide, see [PROJECT_MAP.md](PROJECT_MAP.md).

## Paper Citation

Wei Wei et al., "3U: Joint Design of UAV-USV-UUV Networks for Cooperative
Target Hunting," IEEE Transactions on Vehicular Technology, 2023.

## Disclaimer

This repository is an educational reproduction based on the paper's public
mathematical description. It is not the official implementation. The original
PDF is not included due to copyright.

The numerical results produced by this code should be interpreted as
reproduction-style simulation results, not as exact paper results.

## Method Overview

- 3U heterogeneous network: A UAV, a USV, and multiple UUVs cooperate in one
  target hunting task.
- UAV aerial search: The UAV provides aerial monitoring from a configurable
  height.
- USV relay: The USV moves on the sea surface and acts as a communication relay
  between aerial and underwater agents.
- UUV target hunting: The UUV team is represented by a formation center that
  moves in eight discrete horizontal directions; this is formation-center
  planning, not independent multi-UUV control.
- Energy model: UUV energy includes propulsion-oriented motion energy and an
  optional acoustic communication term.
- Connectivity model: UAV-USV connectivity and underwater USV-UUV graph
  connectivity are approximated from distances.
- Sensing and belief model: UAV, USV, and UUV-center observations are noisy,
  and planners/agents use a fused target-position belief instead of the true
  target position when belief state is enabled.
- Fisher Information Matrix: A range-bearing FIM estimates how informative the
  current platform geometry is for 3D target-position estimation.
- Stackelberg pursuit-evasion game: The trajectory generator supplies the UUV
  action, and Stackelberg only predicts the target best-response escape action;
  it does not reselect or overwrite the UUV control action.
- Lyapunov-inspired safety filter: Candidate UUV actions are screened only for
  safety risks such as boundary violation, relay-chain breakage, and low
  remaining energy. It does not include a target-distance tracking term.
- Conditional Flow Matching: A small MLP vector field generates smooth
  short-horizon formation-center trajectories as the integrated planner's
  primary trajectory generator, conditioned on state, target belief, energy,
  connectivity, FIM, and predicted target response metrics.
- DQN / Double DQN / Dueling DQN: PyTorch agents learn UUV formation-center
  trajectory decisions.
- ACO baseline: A grid-based Ant Colony Optimization planner provides a
  classical path-planning comparison.

## Repository Layout

```text
3u-uav-usv-uuv-target-hunting/
|-- README.md
|-- LICENSE
|-- requirements.txt
|-- configs/
|   `-- default.yaml
|-- src/
|   |-- main.py
|   |-- envs/
|   |   `-- three_u_env.py
|   |-- agents/
|   |   |-- dqn_agent.py
|   |   |-- double_dqn_agent.py
|   |   |-- dueling_dqn_agent.py
|   |   `-- aco_baseline.py
|   |-- models/
|   |   `-- q_network.py
|   |-- utils/
|   |   |-- physics.py
|   |   |-- connectivity.py
|   |   |-- energy.py
|   |   |-- replay_buffer.py
|   |   |-- plotting.py
|   |   `-- seed.py
|   |-- sensing/
|   |   `-- fisher_information.py
|   |-- game/
|   |   `-- stackelberg.py
|   |-- generative/
|   |   |-- flow_matching.py
|   |   |-- trajectory_dataset.py
|   |   `-- train_flow_matching.py
|   |-- safety/
|   |   `-- lyapunov.py
|   `-- experiments/
|       |-- baselines/
|       |   |-- run_dqn.py
|       |   `-- run_aco.py
|       |-- sensitivity/
|       |   |-- compare_height.py
|       |   `-- compare_speed.py
|       |-- reproduction/
|       |   `-- reproduce_table2.py
|       `-- ablations/
|           |-- ablation_fim.py
|           |-- ablation_flow_matching.py
|           |-- ablation_lyapunov.py
|           `-- ablation_stackelberg.py
|-- results/
|   |-- figures/
|   |-- tables/
|   |-- checkpoints/
|   `-- datasets/
`-- tests/
    |-- test_env.py
    `-- test_flow_matching.py
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Run Examples

Run the main reproduction pipeline:

```bash
python src/main.py --config configs/default.yaml
```

Run planner modes from the main entry point:

```bash
python src/main.py --config configs/default.yaml --planner dqn
python src/main.py --config configs/default.yaml --planner dqn_lyapunov
python src/main.py --config configs/default.yaml --planner dqn_fim_stackelberg_lyapunov
python src/main.py --config configs/default.yaml --planner flow_matching
python src/main.py --config configs/default.yaml --planner full
```

Train or evaluate DQN-family agents:

```bash
python src/experiments/baselines/run_dqn.py
python src/experiments/baselines/run_dqn.py --agent double_dqn
python src/experiments/baselines/run_dqn.py --agent dueling_dqn
python src/experiments/baselines/run_dqn.py --no-lyapunov
```

Run the ACO baseline:

```bash
python src/experiments/baselines/run_aco.py
```

Run Fig. 2-like and Fig. 3-like sensitivity experiments:

```bash
python src/experiments/sensitivity/compare_height.py
python src/experiments/sensitivity/compare_speed.py
```

Run the Table II-like reproduction:

```bash
python src/experiments/reproduction/reproduce_table2.py
```

Run the Lyapunov safety ablation:

```bash
python src/experiments/ablations/ablation_lyapunov.py
```

Run the FIM and noisy-belief ablation:

```bash
python src/experiments/ablations/ablation_fim.py
```

Run the Stackelberg pursuit-evasion ablation:

```bash
python src/experiments/ablations/ablation_stackelberg.py
```

Train the Flow Matching trajectory generator and run its ablation:

```bash
python src/generative/train_flow_matching.py --dataset-size 512 --epochs 5 --regenerate-dataset
python src/experiments/ablations/ablation_flow_matching.py
```

For a short smoke run of the main pipeline:

```bash
python src/main.py --config configs/default.yaml --episodes 5
```

## Outputs

Generated figures are saved under:

```text
results/figures/
```

Common outputs include reward curves, energy curves, Fig. 2-like energy plots,
and Fig. 3-like distance/connectivity plots.

Generated tables are saved under:

```text
results/tables/
```

Common outputs include:

- `dqn_training.csv`
- `double_dqn_training.csv`
- `dueling_dqn_training.csv`
- `aco_results.csv`
- `fig2_height_raw_results.csv`
- `fig2_speed_raw_results.csv`
- `fig3_results.csv`
- `table2_reproduction.csv`
- `table2_reproduction.md`
- `ablation_fim.csv`
- `ablation_flow_matching.csv`
- `ablation_lyapunov.csv`
- `ablation_stackelberg.csv`

The FIM ablation also writes:

- `fim_logdet_curve.png`
- `belief_error_curve.png`

The Lyapunov ablation also writes:

- `lyapunov_curve.png`
- `lyapunov_ablation.png`

The Stackelberg ablation also writes:

- `stackelberg_distance_curve.png`
- `stackelberg_success_rate.png`

The Flow Matching ablation also writes:

- `flow_matching_trajectories.png`
- `flow_matching_ablation.png`
- `trajectory_smoothness.png`

Model checkpoints are saved under:

```text
results/checkpoints/
```

Synthetic trajectory datasets are saved under:

```text
results/datasets/
```

## Known Differences From The Paper

- Some details are approximated because the paper does not release official
  code.
- The UUV team is represented by a formation center. The simulator does not
  claim independent multi-UUV positions, energies, actions, or communication
  links unless that abstraction is explicitly extended.
- The USV movement and target escape behavior are simplified.
- The sensing model uses a compact Gaussian range-bearing approximation rather
  than a calibrated physical sensor stack.
- The UUV team belief state is a fused target-position estimate, not a full
  Bayesian multi-target tracker.
- The Stackelberg game is a one-step discrete best-response approximation; it
  is designed to test the pursuit-evasion idea without making DQN training
  prohibitively slow.
- The Flow Matching generator is trained on synthetic trajectories from DQN
  rollouts, ACO-style paths, heuristic pursuit, and safety-filtered simulator
  rollouts because no official expert trajectory dataset is available.
- Flow Matching is the default integrated trajectory generator. DQN-family
  agents remain available as baselines or coarse-action conditioners.
- Exact numerical results may differ from the paper.
- The ACO baseline is a standard grid-based approximation rather than an
  official baseline implementation.
- The Lyapunov module is a one-step risk filter for boundary, connectivity,
  and energy safety. It is not a formal proof of global stability.

## Future Work

- Multi-target hunting.
- More realistic UUV dynamics.
- More intelligent target escape strategy.
- Longer-horizon pursuit-evasion planning.
- Higher-capacity diffusion or flow policies trained on richer simulator data.
- Multi-agent reinforcement learning.
- Extended Kalman filtering or particle filtering for target belief updates.
- Learned UAV and USV policies instead of simple scripted movement.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
