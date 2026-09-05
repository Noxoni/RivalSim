# Count-weighted training telemetry through +30

These are stochastic curriculum/self-play measurements, not fixed-case
deterministic evaluation or evidence of competitive strength. Each block
contains58,982,400 trainable player decisions. All ratios below use sums of
their underlying counts rather than means of per-update ratios.

| Metric | Updates1-10 | Updates11-20 | Updates21-30 |
|---|---:|---:|---:|
| Touches / player-minute | 1.0851 | 1.5289 | 1.7468 |
| Physical goals / world-minute | 0.4730 | 0.5464 | 0.5588 |
| Ended player episodes with a touch | 17.87% | 29.78% | 33.33% |
| No-touch fraction of world resets | 85.01% | 85.83% | 86.35% |
| Conditional first-touch seconds | 1.5917 | 1.7443 | 1.7730 |
| Goalward fraction of touch events | 91.11% | 86.13% | 85.10% |
| Mean movement speed, uu/s | 747.13 | 737.63 | 724.02 |
| Jump requested fraction | 19.36% | 17.19% | 15.43% |
| Boost requested fraction | 25.11% | 23.90% | 24.14% |
| Handbrake requested fraction | 29.87% | 28.97% | 26.81% |
| Categorical entropy, nats | 3.2883 | 3.3590 | 3.4109 |

There are more contacts and somewhat more goals in sampled training experience,
but no-touch resets remain the dominant outcome and rise slightly as a fraction
of resets. Goalward contact fraction and movement speed fall. Touch timing
does not improve when correctly weighted by actual first-touch events. The
first block is particularly affected by synchronized initial/reset state.

A naive unweighted average of per-update conditional first-touch times gives
3.57s for the first block and1.84s for the third, falsely suggesting a large
improvement. The actual count-weighted means are1.59s and1.77s. The reporter
therefore recomputes every such conditional statistic from its numerator and
denominator; the focused test includes deliberately unequal episode counts.

No-touch reset age is world-owned and resets on either player's touch. Player
episode touch coverage has a different denominator and may overlap a later
no-touch reset. These counts must not be presented as complements. Likewise,
signed antisymmetric reward components cancel over the two trainable players;
zero signed goal-reward sum is expected and not evidence of missing goals.

All90 action-index counts reconcile exactly with emitted sample totals and
the three recorded button counts. Every update reports182 optimizer steps and
zero KL rejections. KL remains telemetry only. No training configuration,
reward, model, optimizer, or active append log was modified by this reporter.

`training_summary_030.json` retains all30 source rows, their original byte-prefix
hash, ten-update blocks and complete-prefix statistics. The live append file
is deliberately not staged while the worker is writing it.

The fixed development evaluations remain the behavioral comparison: at +20,
acquisition18/64 vs original parent16/64, finishing14 goals unchanged, and
Nexto0 goals for /64 against. These training trends do not supersede that result.

## Extension through +50

The same count-weighted method, without changing its definitions, yields:

| Metric | Updates31-40 | Updates41-50 |
|---|---:|---:|
| Touches / player-minute | 1.9731 | 2.1787 |
| Physical goals / world-minute | 0.6166 | 0.6964 |
| Ended player episodes with a touch | 36.74% | 40.08% |
| No-touch fraction of world resets | 84.81% | 82.66% |
| Conditional first-touch seconds | 1.7611 | 1.7615 |
| Goalward fraction of touch events | 84.64% | 85.51% |
| Mean movement speed, uu/s | 715.75 | 686.82 |
| Jump requested fraction | 14.89% | 11.76% |
| Boost requested fraction | 23.45% | 23.12% |
| Handbrake requested fraction | 26.69% | 24.45% |
| Categorical entropy, nats | 3.4709 | 3.4817 |

Training contacts and goal rates continue upward; no-touch share now declines,
although most completed world episodes still end by no-touch timeout. Speed is
lower, not higher. First-touch time remains flat after the initial block. The
fixed +50 evaluation independently measures35/64 acquisition cases with a touch
and17 finishing goals, while Nexto scoring remains0for/64against. This is better
basic acquisition, not evidence of useful aerials, possession, or match strength.

`training_summary_050.json` binds all50 source rows and their original byte
prefix hash. Its presence does not authorize changing the live pilot budget.
