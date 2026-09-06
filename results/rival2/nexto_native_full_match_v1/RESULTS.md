# Full matches confirm a serious simulated-Nexto integration fault

No training. Exact reference600 checkpoint and model state remain unchanged:
`8AC192B4154223DB7CE82B41C77DC475D63B6AFB260EF0B9B9AC4455F6C6F6B2`.
Protocol/code were committed at `b09e06b13887056b2ff0fbf2d5df5b56a2c43ff8` and
all 14 commit files remotely read back before launch. Both arms used the same
five standard layouts on both Rival sides, fixed seed, 120 Hz physics, 30 Hz
deterministic recurrent Rival, and 300 seconds of simulator regulation.

## Result

| Metric | Existing simulator Nexto | Native timing + corrected table |
|---|---:|---:|
| Rival match wins | 8 / 10 | 0 / 10 |
| Rival match losses | 2 / 10 | 10 / 10 |
| Unresolved ties | 0 | 0 |
| Rival goals | 169 | 34 |
| Nexto goals | 122 | 239 |
| Rival touches | 486 | 397 |
| Rival touches per world-minute | 9.72 | 7.94 |
| Worlds with no Rival touch | 0 | 0 |
| Rival kickoff first touches | 0 | 0 |
| Measured evaluation wall seconds | 266.82 | 297.82 |

The baseline exactly reproduces the previous 8/10, 169-122 result. Both arms'
first 2,400 ticks of **all applied Rival/Nexto controls** and all 600 car/ball
pose samples exactly match their respective earlier 20-second intervention
captures. Every arm runs the full 36,000 ticks. No overtime was needed or run.

### Per-match scores, Rival first

| Rival side / initial layout | Existing | Native timing + table |
|---|---:|---:|
| Blue / 0 | 17-12 | 2-21 |
| Blue / 1 | 18-11 | 6-27 |
| Blue / 2 | 11-14 | 3-26 |
| Blue / 3 | 18-11 | 6-25 |
| Blue / 4 | 10-13 | 5-25 |
| Orange / 0 | 15-12 | 6-21 |
| Orange / 1 | 21-12 | 0-24 |
| Orange / 2 | 20-12 | 0-24 |
| Orange / 3 | 19-12 | 0-23 |
| Orange / 4 | 20-13 | 6-23 |

## What this establishes

The old opponent integration substantially overstates this Rival checkpoint's
strength on the fixed simulator benchmark. It is not simply an uncertain reward
design or a need for billions more samples. Changing the opponent's known
integration discrepancies reverses the match result while Rival's weights,
actions contract, recurrent semantics, physics and match rules are unchanged.

The prior four-way 20-second experiment separately identified **neural
compute/emission timing** as the influential component: table-only correction
left all sampled poses and scores identical. This full comparison confirms a
large persistent effect of the combined correction; it does not independently
separate late-match table effects from timing beyond the shorter factorial test.

Rival remains active, but kickoff acquisition is poor even against the original
opponent: all recorded kickoff first touches go to Nexto. Winning under the old
integration did not establish kickoff mastery or robust native gameplay.

The actual native V2 results (0-24 Blue, 1-30 Orange) remain separate evidence.
The corrected-timing simulator is qualitatively much less optimistic, but the
worlds are not matched to those native trajectories and these scores do not
measure the exact fraction of the native transfer gap explained.

## What this does not establish

The diagnostic subclass is **not production-ready full native Nexto parity**.
It isolates pending neural action versus emitted controls and the confirmed
kickoff yaw literal. Existing simulator observation construction, deterministic
argmax, kickoff admission/script lifecycle and absence of native countdowns
remain unchanged. Native stochastic kickoff neural selection is not reproduced.
The subclass assumes all worlds are active, which is valid for this fixed
regulation experiment but not dynamic training assignments or mixed overtime.

No user-play-ready promotion, SSL capability, new checkpoint, PPO update, reward
change, physics change or native game occurred. No change to Rival weights is
proposed as a substitute for correcting the simulator opponent.

## Evidence and next action

Both `.npz` files contain all 36,000 control ticks and 9,000 sampled physical
poses per world. Per-arm JSON contains existing full-match raw telemetry,
scores, hashes, elapsed time and exact-prefix/model checks. `results.json`
summarizes both arms; `protocol.json` binds sources and limits. The shared GPU
lease was held throughout. The worker exited normally.

Ten focused tests passed: exact installed-controller clock oracle, pending
history/emission separation, kickoff override dispatch, original causal evidence,
full trace/source/checkpoint hashes, exact prefixes and no-learning/tie semantics.
See `tests.xml`.

Next implement a separately versioned, production-safe Nexto integration with
independent pending-action and emitted-control buffers and per-world cadence.
Validate activation/deactivation, reset boundaries and batching against the
actual installed controller semantics before wiring it into training. Retire
the copied kickoff-table reference and correct the literal against the actual
upstream source. Preserve old benchmark results as legacy-opponent evidence.
Do not simply insert the all-active diagnostic subclass into PPO or declare
native parity from this test. No training is running now.
