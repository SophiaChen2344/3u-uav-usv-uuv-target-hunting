# Experiments

The experiment scripts are grouped by purpose so GitHub browsing stays readable.

## Baselines

- `baselines/run_dqn.py`: train and evaluate DQN, Double DQN, and Dueling DQN.
- `baselines/run_aco.py`: evaluate the Ant Colony Optimization baseline.

## Sensitivity Studies

- `sensitivity/compare_height.py`: Fig. 2/Fig. 3 style UAV-height sweep.
- `sensitivity/compare_speed.py`: Fig. 2/Fig. 3 style UUV-speed sweep.

## Paper-Style Reproduction

- `reproduction/reproduce_table2.py`: Table II style comparison across initial
  target distances.

## Ablations

- `ablations/ablation_lyapunov.py`: DQN with and without the Lyapunov filter.
- `ablations/ablation_fim.py`: true-target state, noisy belief, and FIM reward
  variants.
- `ablations/ablation_stackelberg.py`: scripted target escape versus
  Stackelberg best response.
- `ablations/ablation_flow_matching.py`: DQN variants versus Flow Matching
  trajectory proposal variants.

