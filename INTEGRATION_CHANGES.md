# Integration Changes and Effects

This note summarizes the recent code integration changes that align the
simulator with the intended Flow Matching, FIM, Stackelberg, and Lyapunov
responsibilities.

## What Changed

- Flow Matching is now treated as the primary integrated trajectory generator.
  The default planner mode is `full`, and Flow Matching conditions trajectory
  generation on the target belief mean, target belief covariance, FIM
  diagnostics, energy, connectivity, and predicted target response terms.
- Flow Matching training is now information-aware. The model's predicted future
  trajectory is used to compute differentiable Fisher-information gain, which
  is optimized together with the base Flow Matching velocity loss plus speed,
  step-length, and smoothness penalties.
- Stackelberg no longer reselects or overwrites the UUV control action. It now
  predicts only the target follower's best-response escape action for the
  action supplied by the trajectory generator or policy.
- The Lyapunov module is now a risk filter rather than a tracking controller.
  Its score excludes target-distance pursuit terms and only reacts to boundary
  risk, communication-chain risk, and low-energy risk.
- Target observation, belief update, FIM diagnostics, and Flow Matching's
  differentiable information-gain loss now share the same range-bearing noise
  model. This removes the earlier mismatch between Cartesian noisy observations
  and range-bearing FIM calculations.
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
  uncertainty along poorly estimated directions, while still discouraging
  excessive speed, step length, and nonsmooth motion.
- Connectivity metrics and safety violations should be more meaningful because
  link loss can occur under the default ranges.
- Energy comparisons between real environment steps and Flow Matching
  candidate trajectories should be more consistent because both paths use the
  same helper function.

## Validation Notes

- Static diff validation with `git diff --check` passed; Git only reported
  expected CRLF line-ending warnings on Windows.
- Python tests were not run in this environment because the available
  `python.exe` is the WindowsApps placeholder and is not an executable Python
  runtime.
