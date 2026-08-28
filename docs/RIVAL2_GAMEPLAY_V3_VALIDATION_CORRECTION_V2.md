# Rival2 Gameplay V3 validation correction v2

Status: `GAMEPLAY_V3_VALIDATION_CORRECTION_V2_READY_FOR_REVIEW`

This is a validation/runtime-parity correction only. It does not authorize
Gameplay V3 PPO training. No PPO update, optimizer step, campaign checkpoint,
policy change, or opponent change was made.

## Repository identity

- Requested ancestor: `70ea2c73bbfd3a5b0415694eb37043fa276fa0c6`.
- Starting `main`: `296095d478693bd11def97963827763cd34fad0b`.
- Reviewed V1 correction: `5efa83f331855ae86a8076b7c0c1a9dc8fae88c4`.
- V2 runtime implementation: `acffb4b01de43fc556cccc27ca5868de5fd1bc92`.
- Source checkpoint SHA-256:
  `3B994E118A9498713DC6115D38F061958A900EA8F4D00CE568F916942E851D9A`.

## Exact correction

Production now keeps per-car GPU-resident opponent-contact onset state: the
legitimate onset tick and the ball position at that onset. At a Rival
directional-dodge contact candidate, an opponent contact within the frozen
three-tick association window and 26.472905 uu displacement marks the candidate
as contested whether that opponent contact came before, after, or on the same
physical tick. The state is affirmative exemption evidence only. The sole
negative outcome is still the existing `UNNECESSARY_FLIP_THROUGH_CONTACT`
fallback after all exemptions fail.

Controlled release no longer reuses the contest timeout's current distance.
Production freezes the control relation at the authoritative `has_flipped`
transition, including the valid zero-history case, then captures the first
post-contact outward-separation transition within a separately derived
five-physics-tick window. It records the release tick, distance, outward speed,
and age while allowing the candidate to remain pending. The resolver applies
the frozen release distance, outward-speed, and ball-transfer boundaries to
that captured sample. Controlled flick remains exemption-only and pays zero
positive reward.

## Physical corpus and measured contact order

The corpus contains 216 authoritative CUDA Soccar traces: three classifiers,
each with 24 positives, 24 hard near misses, and 24 ordinary controls. Per
classifier/class, 16 cases are derivation and 8 are untouched held-out.

Contest positives are balanced by physical topology:

| Measured/topological class | Derivation | Held-out | Production result |
|---|---:|---:|---|
| Opponent before Rival | 4 | 2 | `EXEMPT_CONTESTED_50` |
| Opponent after Rival | 4 | 2 | `EXEMPT_CONTESTED_50` |
| Closest representable simultaneous | 4 | 2 | `EXEMPT_CONTESTED_50` |
| Convergence-only challenge | 4 | 2 | `EXEMPT_CONTESTED_50` |

Every ordered case asserts its measured ticks before it is admitted. Delayed
unrelated opponent-before and opponent-after contacts are retained as negative
controls. No runtime symbol, branch, reward component, or outcome treats
opponent-first contact as negative.

## Prospectively derived boundaries

Values are positive edge / negative edge / physical margin / selected midpoint.

| Classifier | Boundary | Positive | Negative | Margin | Selected |
|---|---|---:|---:|---:|---:|
| Contest | association ticks max | 1 | 6 | 5 | 3 |
| Contest | association displacement max | 11.669798 | 41.276012 | 29.606215 | 26.472905 |
| Contest | opponent distance max | 296.267578 | 575.944885 | 279.677307 | 436.106232 |
| Contest | self closing min | 1099.186768 | 356.162964 | 743.023804 | 727.674866 |
| Contest | opponent closing min | 924.399048 | 256.540955 | 667.858093 | 590.470001 |
| Contest | time-to-ball delta max | 0.186091 | 0.287204 | 0.101113 | 0.236648 |
| Power | total closing min | 190.630966 | 0 | 190.630966 | 95.315483 |
| Power | rotational closing min | 190.630966 | 138.139969 | 52.490997 | 164.385468 |
| Power | rotational share min | 0.303494 | 0.206315 | 0.097179 | 0.254904 |
| Power | ball delta-v min | 405.662811 | 201.453308 | 204.209503 | 303.558060 |
| Controlled | history ticks min | 8 | 2 | 6 | 5 |
| Controlled | max distance | 200.024445 | 357.816498 | 157.792053 | 278.920471 |
| Controlled | max relative speed | 6.886312 | 1699.393677 | 1692.507364 | 853.139995 |
| Controlled | release window ticks max | 1 | 9 | 8 | 5 |
| Controlled | release distance min | 152.353317 | 0 | 152.353317 | 76.176659 |
| Controlled | release outward speed min | 222.432205 | 0 | 222.432205 | 111.216103 |
| Controlled | release ball delta-v min | 326.830811 | 0 | 326.830811 | 163.415405 |

Frozen threshold identity:
`03A1F2C015579CB28B031E618C3681C49CB99AFCFED60B636E5B9107A7B454BE`.

Historical Gameplay V1 and V2 hashes remain exactly unchanged. The corrected
Gameplay V3 contract hash is
`174D94E19B3F053E250147F98835C18CF65260A82E23B6E58F234F6E81E0D4E7`.

## Classifier and runtime gates

Untouched held-out results are 8 TP, 16 TN, 0 FP, and 0 FN for each of contest,
power contact, and controlled flick. The separately executed production replay
also passes all 216 traces, with production candidate/outcome/exemption/tick and
captured-feature evidence retained row by row. Production held-out FP/FN are
0/0. Float parity uses `max(1e-3 absolute, 2e-6 relative)`; the apparently
large 47.289844 maximum absolute time-to-ball error occurs only on roughly
1e8-valued float32 sentinel-like division results and is within the documented
relative tolerance. All ordinary-scale errors are at most 0.000244141.

The unchanged dash/reset source-exact regression passes all 12 cases. Focused
regressions pass: Ruff is clean and 35 tests pass with 15 Torch JIT deprecation
warnings. The AST scan finds no `trainer.update`, `train_iteration`, or
`optimizer.step` call in either validator.

## Exact-scale and checkpoint gates

- One decision at exactly 131,072 worlds: PASS; observations are
  `(131072, 2, 182)`, rewards finite/zero-sum, production evidence buffers
  absent, and measured hot-path host transfers zero. Elapsed: 2.586604 s.
- V3 logical state: 338,690,284 bytes (323.000225 MiB), including only the
  minimal additional contest/release state required by this correction.
- Horizon-32 rollout-only at exactly 131,072 worlds: PASS in 4.902421 s; model,
  optimizer, iteration, and policy version unchanged; no update called.
- Explicit checkpoint transition: PASS; model/optimizer/RNG/counters,
  historical pool, opponent assignment, and mixed-curriculum state preserved.
- Reward reconstruction: PASS over 4,096 decisions with <=1e-6 error, exact
  zero-sum arithmetic, exact zero touch component, and runtime threshold parity.

## No-learning shadow comparison

Both shadows use the same 256 episodes, frozen policy, frozen opponents, and
source state. Touches/min (20.073142), flip-active touches/min (12.634272), all
canonical mechanics counts, mechanics/progress ratio (0.013925), budget-hit
fraction (0.001953125), and impossible count (0) are identical.

Unnecessary flip contacts decline from 6.280237 to 6.073602 per minute;
unnecessary flip-touch fraction declines from 0.497079 to 0.480724; and
bad-flip/progress ratio declines from 0.028832 to 0.027342. Exemptions change
from contest/controlled/power/recognized `789/66/15/3` to `682/255/15/3`.
This is expected: the corrected three-tick contest boundary removes delayed
unrelated associations while the aligned release transition recognizes real
controlled flicks.

The bounded natural shadow evidence contains 42 opponent-before, 51
opponent-after, 8 same-tick/closest, and 21 convergence-only contest exemplars,
plus 30 controlled-release exemplars with exact self-contact/release ticks and
16 captured features.

The checkpoint remains byte-identical. Model, optimizer, iteration 479, policy
version 479, and sample counter 3,655,854,038 are unchanged. PPO update calls
are exactly zero.

## Evidence

Machine evidence is under
`results/rival2/gameplay_v3_validation_correction_v2/`. The final artifact
manifest records staged Git-blob IDs, normalized byte sizes, and SHA-256 values
for every reviewer artifact; a post-commit readback independently verifies the
committed files before the final push.

Final verdict: `GAMEPLAY_V3_VALIDATION_CORRECTION_V2_READY_FOR_REVIEW`.
