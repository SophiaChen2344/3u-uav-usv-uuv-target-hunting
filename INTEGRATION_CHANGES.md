# Integration Changes and Effects

This note summarizes the recent code integration changes that align the
simulator with the intended Flow Matching, FIM, Stackelberg, and Lyapunov
responsibilities.

## What Changed

- Flow Matching is now treated as the primary integrated trajectory generator.
  The default planner mode is `full`, and Flow Matching conditions trajectory
  generation on the target belief mean, target belief covariance, FIM
  diagnostics, energy, connectivity, and predicted target response terms.
- Flow Matching training is now information-aware and game-guided. The
  condition vector's Stackelberg target escape response is rolled forward into
  a predicted target path, and the model's predicted UUV formation-center
  trajectory is combined with the current UAV and USV positions to compute
  differentiable heterogeneous Fisher-information gain. This term is optimized
  together with the base Flow Matching velocity loss plus speed, step-length,
  smoothness, boundary, relay-connectivity, and energy-reserve penalties.
- The differentiable FIM now uses smooth sigmoid observation-range gates, so
  sensor influence fades near UAV, USV, and UUV range limits instead of
  switching on or off discontinuously. The default ranges are 600 m for UAV,
  400 m for USV, and 130 m for the UUV formation center, with a 20 m smoothing
  width.
- The Flow Matching information-gain loss weight was increased so the FIM term
  is less likely to be drowned out by the base velocity-matching loss. The
  default `w_information_gain` is now `0.2`.
- Flow Matching training now includes Lyapunov-style safety regularizers for
  boundary margin, UAV-USV/USV-UUV relay continuity, and UUV energy reserve.
  These regularizers intentionally avoid target-distance terms, so they only
  shape feasible generated motion rather than becoming a tracking controller.
- Flow Matching inference now follows the learned generator directly. It
  samples one primary trajectory and converts that trajectory to the UUV
  formation-center action without using runtime FIM candidate filtering or
  online trajectory optimization.
- Stackelberg no longer reselects or overwrites the UUV control action. It now
  predicts only the target follower's best-response escape action for the
  action supplied by the trajectory generator or policy.
- The Lyapunov module is now a risk filter rather than a tracking controller.
  Its score excludes target-distance pursuit terms and only reacts to boundary
  risk, communication-chain risk, and low-energy risk.
- Target observation, belief update, FIM diagnostics, and Flow Matching's
  differentiable heterogeneous information-gain loss now share the same
  range-bearing noise model. This removes the earlier mismatch between
  Cartesian noisy observations and range-bearing FIM calculations.
- Target belief covariance is propagated with the fused observation and exposed
  as a condition input, so the generator can react to anisotropic target
  uncertainty instead of relying on target mean alone. The covariance is built
  from range variance and bearing variance, allowing far or poorly aligned
  sensing geometry to produce direction-specific uncertainty.
- The environment and Flow Matching scoring now call the same UUV
  formation-center step energy helper, so executed steps and generated
  trajectory candidates use the same energy assumptions.
- Communication ranges were reduced from always-connected values to ranges that
  can actually produce broken UAV-USV or USV-UUV links in the 400 m by 400 m
  simulation area.
- The simulator documentation now describes the UUV abstraction as
  formation-center planning. It does not claim independent multi-UUV positions,
  actions, energies, or communication links.

## Expected Effects

- Integrated runs should be easier to interpret because each module has a
  single responsibility: information-aware Flow Matching plans, FIM and belief
  covariance condition the planner, Stackelberg predicts target escape, and
  Lyapunov applies minimal safety correction.
- Stackelberg diagnostics may still report target response utility and
  predicted information terms, but the executed UUV action remains the proposed
  action unless the Lyapunov risk filter changes it for safety.
- Lyapunov interventions should now correspond to concrete safety risks such
  as approaching a boundary, breaking relay connectivity, or exhausting the
  energy reserve. It should not steer toward the target purely because of
  target distance.
- FIM values, belief errors, and target observations should be more consistent
  because they are derived from a common sensor model.
- Training should prefer trajectories that reduce target localization
  uncertainty along poorly estimated directions and the Stackelberg-predicted
  escape path, while still discouraging excessive speed, step length,
  nonsmooth motion, boundary risk, relay-chain breakage, and low reserve
  energy. The stronger FIM loss weight should make this information-seeking
  behavior more visible during training.
- Inference runtime should be more clearly attributable to the learned
  generator because FIM affects behavior through conditioning and training
  rather than through a separate candidate-ranking stage.
- Connectivity metrics and safety violations should be more meaningful because
  link loss can occur under the default ranges.
- Energy comparisons between real environment steps and Flow Matching
  candidate trajectories should be more consistent because both paths use the
  same helper function.

## Validation Notes

- Static diff validation with `git diff --check` passed; Git only reported
  expected CRLF line-ending warnings on Windows.
- Python 3.12.10 was installed through `winget`, dependencies from
  `requirements.txt` were installed, and `python -m pytest` passed with
  17 tests.
- A short Flow Matching training smoke test also passed with a 12-sample
  heuristic dataset, one training epoch, heterogeneous FIM information gain
  weight `0.2`, one generated trajectory, and direct planner inference.
