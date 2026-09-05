# SSL development: first controlled exploration experiment

The user authorized ownership of Rival's development toward SSL gameplay on
2026-09-05. This is a development objective, **not an achieved capability claim**.
Beating Nexto on a short scenario test would not establish SSL strength either.
The immediate bottleneck is reliable acquisition and returning after a miss.

## Preserved baseline and first evidence

The original fresh 30 Hz run was cleanly paused by the agent at update 597,
not stopped by a safety guard. Its runner labels every STOP-file exit
`stopped_by_user`; here the actual reason is the user-authorized experiment.
The original run directory retains its STOP file to prevent a concurrent restart.
The immutable comparison parent is `parent_u000597.pt`, SHA-256
`B0B35CDAF3B3551EC667776EB99C3822F863AAA1F17A0BA2F013B5F216BD87A5`.
No random restart, V5, or BC parent is used.

The bounded native CUDA trajectory diagnostic, with no optimizer step, found:

- First second: mean speed 1,015 uu/s; approaching the ball at >100 uu/s on
  80.4% of active focal samples; boost requested on 98.8%; no jump or handbrake.
- Seconds 1-3: mean speed 1,944 uu/s; receding from the ball at >100 uu/s on
  89.3% of active focal samples; mean nose-to-ball cosine -0.592.
- Seconds 5-10: mean speed 305 uu/s and average distance 4,666 uu.
- Sixteen of 64 cases touched in the first sixteen seconds. This diagnostic
  is not the full 30-second evaluation. Airborne/contact flags do not establish
  named mechanics. These data support a pass-by/miss and poor-recovery pattern,
  not a conclusion that Rival simply refuses to throttle or boost.

Raw trajectories, descriptive time bins and checkpoint hashes are in
`diagnostics/`. The earlier CPU attempt failed to compile a vehicle kernel;
CUDA now provides real completed trajectories without changing physics.

## Frozen comparison

Both arms start from identical parent model tensors, Adam moments/counters,
RNG state, freshly initialized scenario episodes and zero recurrent hidden state.
The control retains existing learned sigma. The other arm multiplies **effective
learned analog sigma by 0.5**, including its bounds. This changes neither the
deterministic action nor the actor mean, button logits, values or weights at
initialization. It is not action averaging or a different control cadence.
One shared forward implementation applies the same distribution in sampling,
PPO old/new likelihood, entropy and telemetry.

Rewards, potential weights, scenario distribution, model architecture, 32,768
worlds, 90-decision rollouts, 30 Hz actions, 120 Hz physics, two PPO passes and
learning rates remain unchanged. Both arms remain pure current self-play for
the comparison; Nexto is evaluation-only. No action penalties, behavior rewards,
mechanic detectors, preservation objectives or KL rejection are added.

Each arm receives exactly 30 additional PPO updates, with deterministic
evaluations and saved snapshots at +10/+20/+30. Every update has a rolling
checkpoint; numerical corruption rolls back and stops, never weakens a guard.
The arms run sequentially on one GPU. Their stochastic trajectories necessarily
diverge after changing the distribution; matching initial RNG does not claim
matched later physics states or multiple-seed statistical significance.

Prospective pilot selection: half-sigma must exceed control acquisition touch
coverage by at least 4/64 at both +20 and +30 and improve on parent coverage by
at least 4/64 at +30. At +30, finishing touch coverage may not trail control by
more than 4/64, nor finishing goals by more than three. Otherwise the result is
inconclusive/negative, not grounds for blindly extending that setting.
These repeatedly viewed evaluations are development/selection data, **not
untouched tests**. An apparent winner still needs independent cases before
promotion or longer use. No SSL acceptance claim uses these thresholds.

## Verification and operation

The original production source/authority is unchanged. The separate experiment
package binds all original and new source hashes. Before optimization it checks
32,768-world real rollouts and recurrent backward for both arms, old likelihood
recomputation, state/action alignment, value consistency, finite gradients, and
unchanged model/Adam step counters. Unit tests also prove exact deterministic
parity and half-sigma semantics. Publish authority/preflight before any optimizer
step; the runner enforces origin/main file equality.

```powershell
.\.venv\Scripts\python.exe benchmarks/run_rival2_ssl_exploration_comparison.py prepare
.\.venv\Scripts\python.exe benchmarks/run_rival2_ssl_exploration_comparison.py preflight
# Commit and push the prospective authority and passing preflight.
.\.venv\Scripts\python.exe -u benchmarks/run_rival2_ssl_exploration_comparison.py both
```

External state: `G:/dev/RivalSim-runs/ssl-development-exploration-v1`, with
per-arm `campaign_state.json`, `latest.json`, and curves. A common Windows file
lease forbids concurrent arms. A root STOP file stops at an accepted boundary.
Only `run --arm <same arm> --resume <verified checkpoint>` can recover an arm;
resume restarts simulator episodes as the original lane does. It does not
claim an interrupted native physics world was serialized.

Next development decisions follow measured acquisition, natural gameplay and
opponent outcomes. No existing reward contract is silently edited. This first
pilot deliberately isolates analog noise; it does not test different button
exploration, temporal noise correlation, reward design or an easier curriculum.
