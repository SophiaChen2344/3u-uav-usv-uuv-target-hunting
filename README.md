# Reproduction of 3U UAV-USV-UUV Cooperative Target Hunting with DQN

This repository provides an educational Python/PyTorch reproduction of the
main ideas from a UAV-USV-UUV cooperative underwater target hunting paper. It
implements a compact 3U simulator, energy and connectivity models, DQN-family
controllers, an Ant Colony Optimization baseline, and experiment scripts for
Fig. 2-like, Fig. 3-like, and Table II-like comparisons.

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
- UUV target hunting: The UUV team is represented by a group center that moves
  in eight discrete horizontal directions.
- Energy model: UUV energy includes propulsion-oriented motion energy and an
  optional acoustic communication term.
- Connectivity model: UAV-USV connectivity and underwater USV-UUV graph
  connectivity are approximated from distances.
- DQN / Double DQN / Dueling DQN: PyTorch agents learn UUV group-center
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
|   `-- experiments/
|       |-- run_dqn.py
|       |-- run_aco.py
|       |-- compare_height.py
|       |-- compare_speed.py
|       `-- reproduce_table2.py
|-- results/
|   |-- figures/
|   |-- tables/
|   `-- checkpoints/
`-- tests/
    `-- test_env.py
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

Train or evaluate DQN-family agents:

```bash
python src/experiments/run_dqn.py
python src/experiments/run_dqn.py --agent double_dqn
python src/experiments/run_dqn.py --agent dueling_dqn
```

Run the ACO baseline:

```bash
python src/experiments/run_aco.py
```

Run Fig. 2-like and Fig. 3-like sensitivity experiments:

```bash
python src/experiments/compare_height.py
python src/experiments/compare_speed.py
```

Run the Table II-like reproduction:

```bash
python src/experiments/reproduce_table2.py
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

Model checkpoints are saved under:

```text
results/checkpoints/
```

## Known Differences From The Paper

- Some details are approximated because the paper does not release official
  code.
- The UUV team is represented by a cluster center.
- The USV movement and target escape behavior are simplified.
- Exact numerical results may differ from the paper.
- The ACO baseline is a standard grid-based approximation rather than an
  official baseline implementation.

## Future Work

- Multi-target hunting.
- More realistic UUV dynamics.
- More intelligent target escape strategy.
- Multi-agent reinforcement learning.
- Learned UAV and USV policies instead of simple scripted movement.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
