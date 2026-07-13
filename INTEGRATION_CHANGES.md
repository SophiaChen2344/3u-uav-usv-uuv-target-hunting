# Integration Changes and Effects

This note summarizes the recent code integration changes that align the
simulator with the intended Flow Matching, FIM, Stackelberg, and Lyapunov
responsibilities.

## What Changed

- Flow Matching is now treated as the primary integrated trajectory generator.
  The default planner mode is `full`, and Flow Matching evaluates
  formation-center trajectory candidates using target belief, FIM diagnostics,
  energy, connectivity, and predicted target response terms.
- Stackelberg no longer reselects or overwrites the UUV control action. It now
  predicts only the target follower's best-response escape action for the
  action supplied by the trajectory generator or policy.
- The Lyapunov module is now a risk filter rather than a tracking controller.
  Its score excludes target-distance pursuit terms and only reacts to boundary
  risk, communication-chain risk, and low-energy risk.
- Target observation, belief update, and FIM diagnostics now share the same
  range-bearing noise model. This removes the earlier mismatch between
  Cartesian noisy observations and range-bearing FIM calculations.
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
  single responsibility: Flow Matching plans, FIM and belief condition the
  planner, Stackelberg predicts target escape, and Lyapunov applies minimal
  safety correction.
- Stackelberg diagnostics may still report target response utility and
  predicted information terms, but the executed UUV action remains the proposed
  action unless the Lyapunov risk filter changes it for safety.
- Lyapunov interventions should now correspond to concrete safety risks such
  as approaching a boundary, breaking relay connectivity, or exhausting the
  energy reserve. It should not steer toward the target purely because of
  target distance.
- FIM values, belief errors, and target observations should be more consistent
  because they are derived from a common sensor model.
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
