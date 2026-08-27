# Rival 2.0 Gameplay V1 bounded curriculum

Status: `COMPLETE_239_UPDATE_BOUNDARY`.

Source commit: `61307571d86508f3026402c4948f759f310ff36c`.

Source checkpoint SHA-256: `4FB7A3B134B25D595374E3968E2EDFA150A9CD6F8910B903BF892B59D7F8BC9A`.

Gameplay reward contract SHA-256: `48AAC000B97D2652507F677184A3FE4F0A3A86CED136B680C933EFF33CD9F072`.

The run resumed only the pinned acquisition-complete learned/training state into fresh `RIVAL2_EPISODE_V1` worlds. Historical `RIVAL2_REWARD_V1` remained immutable; only the new reward identity and fresh short-lifecycle state changed.

```text
Blue = historical_V1_blue
     + speed(Blue) - speed(Orange)
     + supersonic(Blue) - supersonic(Orange)
     + actual_boost_use(Blue) - actual_boost_use(Orange)
     + positive_boost_pickups(Blue) - positive_boost_pickups(Orange)
     + 0.75 * BlueSaves - 0.75 * OrangeSaves
Orange = -Blue
```

## Held-out curve

| label | iteration | samples | touches/min | goals/min | goal fraction | no-touch | hard-time | mean seconds | saves/min |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| source | 120 | 1006632960 | 39.601990 | 1.314987 | 0.669922 | 0.000488 | 0.329590 | 30.567 | 0.597590 |
| plus_060 | 180 | 1509949440 | 38.435961 | 2.102758 | 0.846680 | 0.000488 | 0.152832 | 24.159 | 0.768828 |
| plus_120 | 240 | 2013265920 | 37.993909 | 2.381919 | 0.886230 | 0.000244 | 0.113525 | 22.324 | 0.648302 |
| plus_180 | 300 | 2516582400 | 32.692641 | 2.577840 | 0.912598 | 0.000244 | 0.087158 | 21.241 | 0.532395 |
| plus_239 | 359 | 3011510272 | 36.286826 | 2.288618 | 0.872559 | 0.000488 | 0.126953 | 22.876 | 0.555185 |

## Checkpoints

| label | iteration | samples | SHA-256 | audit |
|---|---:|---:|---|---|
| plus_060 | 180 | 1509949440 | `939F6C66D72C394DAB2AC640EBC181D1337B163A1D390456F39D7FBF15DB7627` | `PASS_GREEN` |
| plus_120 | 240 | 2013265920 | `F98D557217477040EEFB63218C3D56D78587044B56E848A6E6DFF939DE691386` | `PASS_GREEN` |
| plus_180 | 300 | 2516582400 | `FEC1C289E7F7EB8D69876FB75C5325D56063A7A674A46F6FD20C5C270542511B` | `PASS_GREEN` |
| plus_239 | 359 | 3011510272 | `77BF257131FB71DDEAEAE49D668C5E25AB1D06EE26149AB0D0AE303573CA5F21` | `PASS_GREEN` |

## Safety boundary

Every update used a transactional post-step minibatch KL limit of `0.1` and a complete-rollout final-policy mean KL limit of `0.05`. A violation restores model, optimizer, gradients, and RNG state and ends the run; no automatic retuning is permitted.

Full per-update optimizer, actor-distribution, value/return/advantage, saturation, and rollout reward evidence, plus held-out reward-component, movement, save, and boost evidence, is machine-readable under `results/rival2/gameplay_v1/`.

## Exact reward scales

Historical V1 remains goal `+/-10`, progress
`0.5 * delta_ball_y / 5120`, unique touch onset `+/-0.05`, and unique
demolition onset `+/-0.10`. The added per-player terms are speed
`0.00010 * clamp(speed / 2300, 0, 1)`, authoritative supersonic `0.00020`,
physical boost use `0.00005`, positive-gain small/full pad events
`0.001/0.005`, and a legitimate save event `0.75`. All player-local additions
enter as Blue minus Orange before exact negation.

The source reward-scale audit passed. Across the held-out curve, competitive
incidental movement/resource shaping remained only `0.241%` to `0.361%` of
historical V1 mean-absolute reward. At the final checkpoint it was
`0.0000388403` per world-decision versus `0.0155430` for historical V1. Save
shaping was separately `0.000231327` mean absolute per world-decision. It did
not dominate the historical gameplay reward.

## PPO and action trend

| offset | iteration | mean KL | max minibatch KL | clip fraction | value loss | raw grad | post-clip grad | throttle saturation |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 121 | 0.011949 | 0.087496 | 0.184794 | 0.015353 | 3.374754 | 0.500000 | 0.051232 |
| 60 | 180 | 0.010244 | 0.019245 | 0.108143 | 0.158491 | 0.787421 | 0.499216 | 0.049419 |
| 120 | 240 | 0.011667 | 0.014594 | 0.125460 | 0.188580 | 0.975878 | 0.499999 | 0.060507 |
| 180 | 300 | 0.018025 | 0.018728 | 0.136468 | 0.212769 | 1.265033 | 0.500000 | 0.073684 |
| 239 | 359 | 0.013288 | 0.029204 | 0.162568 | 0.205186 | 1.483801 | 0.500000 | 0.091594 |

All 239 updates passed. The maximum completed-update mean KL was `0.027896`
at iteration 335 (limit `0.05`); the maximum post-step minibatch KL was
`0.087496` at iteration 121 (limit `0.10`). The old failure neighborhood was
crossed without recurrence: iterations 347, 348, and 349 had mean KL
`0.014346`, `0.013202`, and `0.014872` rather than the failed lineage's
approximately `58`, `3016`, and `10537`.

## Held-out movement/resource trend

Side values below are averaged only to keep this curriculum report compact;
the machine-readable evaluations preserve Blue and Orange separately.

| label | speed uu/s | supersonic | grounded | boost level | physical boost-use intervals | pad pickups/player-min | jump edges/player-min | flips/player-min | jump active | throttle saturation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| source | 925.734 | 0.024985 | 0.175116 | 10.723 | 0.115020 | 7.873 | 379.516 | 41.765 | 0.491363 | 0.042143 |
| +60 | 969.277 | 0.022428 | 0.331014 | 9.391 | 0.120664 | 8.396 | 252.043 | 34.922 | 0.384364 | 0.052309 |
| +120 | 1020.083 | 0.024945 | 0.395017 | 9.659 | 0.123184 | 8.965 | 183.169 | 31.821 | 0.339768 | 0.060328 |
| +180 | 1043.172 | 0.029218 | 0.428250 | 8.956 | 0.124584 | 9.463 | 142.365 | 29.298 | 0.321299 | 0.070888 |
| +239 | 1049.455 | 0.040266 | 0.474018 | 9.366 | 0.126889 | 9.930 | 112.233 | 27.107 | 0.285580 | 0.092264 |

No movement mechanic was directly rewarded. Speed, supersonic time, grounded
time, boost use, and pad acquisition increased, while jump edges and flips fell.

## Outcome interpretation

The final policy retained the acquisition gate (`0.0488%` no-touch, equal to
source) and improved goals from `1.315` to `2.289/min` and goal termination from
`66.99%` to `87.26%`. Touch rate declined from `39.60` to `36.29/min`
(`-8.37%`). The curve was non-monotonic: +180 peaked at `2.578 goals/min` and
`91.26%` goal termination but only `32.69 touches/min`; +239 recovered touches
while giving back some scoring. Saves ended at `0.555/min`, below the source
`0.598/min` and the +60 peak `0.769/min`. These are honest tradeoffs, not a
monotonic-success claim.

## Implementation and validation

The implementation changed the reward contract/kernel/environment state,
authoritative boost pickup readback, shared goal geometry, transactional PPO
guard and diagnostics, curriculum transition record, bounded runner, active
handoff, and focused tests. It did not change observations, actions, network
architecture, PPO hyperparameters, simulator mechanics, jump/flip mechanics, or
historical V1 semantics.

The final focused validation ran 26 tests covering Gameplay V1 events, exact KL
rollback, runner pinning/arithmetic, existing Rival 2.0 contracts, acquisition
and Scoring V1 reward regressions, and boost-pad lifecycle/visitation ordering.
All 26 passed. Targeted Ruff and `git diff --check` also passed.

One recoverable CUDA caching-allocator allocation warning occurred around update
326, and update 359 took `65.864s` rather than the usual roughly `5-6s`. Update
326 and every later update still passed; final checkpoint loading, finite-model,
counter, contract, evaluation, and SHA-256 audits are green. This runtime anomaly
is retained in `verification.json` rather than hidden.

## Final resumable artifact and recommendation

The final resumable checkpoint is
`checkpoints/rival2/gameplay_v1/rival2_gameplay_resume.pt`, iteration/policy
version 359, 3,011,510,272 cumulative samples, SHA-256
`77BF257131FB71DDEAEAE49D668C5E25AB1D06EE26149AB0D0AE303573CA5F21`.

Do not automatically continue this lineage. The next authorization should first
run a fixed short-lifecycle checkpoint-selection comparison among +120, +180,
and +239 (including direct defense/save outcomes), because scoring peaked at
+180 while final acquisition/touch rate recovered at +239. Select one complete
checkpoint branch prospectively, then define the next bounded curriculum; do
not mix metrics or silently choose the most favorable metric from each branch.
