# Rival 120 Hz human-demonstration BC observation bridge V1

## Outcome and boundary

This lane adds a masked domain bridge that makes the reviewed human demonstration frames
structurally usable by a future behavior-cloning trainer. It does not behavior-clone, run PPO,
construct an optimizer, call backward, change a reward, modify a mechanic detector, or mutate a
model.

The exact-observation adapter remains a separate audit contract. Its source file is byte-identical
to commit `4d8064d6260a87be458f5cbf11f2f882ebe65c07`, with SHA-256
`1B6D01C223419C2C3A686CAB3F012F8D80CA9E77922E8E8AC3AB9C96D2B8DD61`. It continues to
emit no complete observation for these recordings. The BC bridge does not weaken, replace, or
reinterpret that verdict.

## Bridge contract

The bridge version is `RIVAL2_HUMAN_DEMO_BC_OBSERVATION_BRIDGE_V1`. Every sample contains:

- a finite `float32[182]` value vector in `RIVAL2_OBS_V2_120HZ` field order;
- a `uint8[182]` quality vector;
- a boolean availability vector derived from that quality vector;
- the unchanged exact `float32[8]` `RIVAL2_ACTION_V2_120HZ` target;
- source session, sequence, physics-frame, and previous-action provenance; and
- independent `bc_usable` and exact-adapter usability flags.

Quality codes are ordered only to enforce non-promotion:

| Code | Meaning | Global fields |
| ---: | --- | ---: |
| 3 | exact/direct | 16 |
| 2 | exactly derivable | 58 |
| 1 | approximate/semantically reconstructed | 34 |
| 0 | unavailable | 74 |

The field-level contract, source, and reconstruction text for all 182 fields is committed in
`results/rival2/human_demo_bc_bridge_v1/field_quality_contract.json`. Its contract SHA-256 is
`49A6D3C09A3DD5C88263850CF816804FBF3B2BAAED28154F1C46EADED6B1D9BC`.

Per-sample quality may only move downward from the global classification. It does so when a
normally supported source is absent—for example, all opponent-dependent fields in single-car
Freeplay, a wheel whose native index is not unique, or previous action at a true reset/rebind/
respawn/discontinuity boundary. No field can be promoted above its global quality.

## Reconstruction semantics

Exact/direct fields are native on-ground, jump/double-jump, supersonic, and uniquely indexed
wheel-contact values. Exactly derived fields use deterministic team canonicalization,
normalization, basis conversion, relative subtraction, jump-availability derivation, and native
one-frame lifecycle events.

Approximate car values use explicit Rocket League proxies:

- native boost divided by 100;
- jump-component active state and activity time;
- `has_flip`, jumped, and double-jumped flags for flip-used/dodge-availability proxies;
- active dodge/flip components and their activity timers;
- demolished flag and native respawn timer;
- native time-off-ground for air-time and post-jump air-time proxies; and
- active boost-component activity time.

Each timer is clipped after the frozen Rival normalization. These fields remain quality 1 even
when their numeric value agrees with RivalSim; cross-engine component semantics have not been
promoted to exact.

The preceding 120 Hz action is retained when the source predecessor is contiguous. It remains
classified approximate because the field is unavailable at lifecycle boundaries. Episode age and
no-touch age are deterministic lower-bound counters from the first observed span/reset/touch
origin; they do not claim pre-recording history.

The 34 canonical pad active/cooldown pairs are unavailable. Native pickup events contain stable
runtime pointers but no authoritative mapping from those pointers to the canonical Rival pad
indices. Pointer sorting or a fabricated all-active state would be false telemetry. The bridge
therefore writes neutral zero only with quality 0. The same rule applies to `time_since_boosted`,
`supersonic_time`, and `sticky_ticks` for both cars.

Neutral zero is a training-domain placeholder, never a measured or exact value.

## Human corpus result

The bridge reads the split manifest directly from the base commit and verifies every native source
file against its committed byte size and SHA-256 before scanning.

All 114,311 frozen frames are BC-usable:

| Cohort | Split | Trajectories | Frames | BC-usable |
| --- | --- | ---: | ---: | ---: |
| Mechanic positive | train | 77 attempts | 39,275 | 39,275 |
| Mechanic positive | validation | 17 attempts | 8,531 | 8,531 |
| Mechanic positive | test | 16 attempts | 8,199 | 8,199 |
| `nexto_1v1` gameplay | train | 18 regions | 47,347 | 47,347 |
| `nexto_1v1` gameplay | validation | 2 regions | 7,151 | 7,151 |
| `nexto_1v1` gameplay | test | 2 regions | 3,808 | 3,808 |

All 56,005 mechanic-positive frames and all 58,306 gameplay frames retain their exact action.
There are zero action mismatches. The 71 failed and 14 ambiguous mechanic attempts remain outside
the positive cohort.

The normal two-car gameplay profile is 16 direct, 58 derived, 34 approximate, and 74 unavailable
fields. The normal single-car Freeplay profile is 8 direct, 36 derived, 22 approximate, and 116
unavailable fields because opponent context is a masked nuisance rather than a fabricated car.
Small additional profiles are caused by real wheel-index and lifecycle boundaries and are recorded
exactly in `human_corpus_statistics.json`.

## Authoritative simulator calibration

The bounded ground-truth corpus is the accepted
`RIVAL2_RETENTION_120HZ_FROM_ITERATION_479_V1` artifact. It contains 512 unique, finite,
authoritative `float32[182]` observations selected deterministically from 4,194,304 trainable
120 Hz observations generated with the unmodified iteration-479 actor. It covers all three X/Y
field regions, eight heading octants, ordinary ground play, ball approach, near-ball interaction,
recovery, and airborne play.

Provenance is pinned to:

- corpus artifact SHA-256:
  `C4957B06847E7F61B5DC313ABAC58CD2FE8AB696561C7A4898C6ACFF219DACDC`;
- true-observation tensor SHA-256:
  `BED4DD26268A667251B944844544306734E0E44E54AB31689B4DE782FC0965FA`;
- clean 120 Hz bootstrap SHA-256:
  `ADAF8D015C340CAFAE857B7253FBBDE3A6C842C4EA0BB091B31F8B1C210ED350`; and
- bootstrap model tensor SHA-256:
  `1AA50DC45E9E0FDD0B24510A26781787742BBE8C8ED5FF6B77FD72BEC3EFA8C3`.

For every source observation the committed corpus retains the true vector, degraded vector,
quality mask, teacher actor, and student actor. The simulator supplies the numerical target of each
non-unavailable reconstruction, but its quality remains direct/derived/approximate as appropriate.
Unavailable fields are neutralized. This intentionally isolates missing-feature loss; zero error
for approximate fields is not evidence that Rocket League and RivalSim component semantics are
identical.

Measured normalized observation error:

| Group | Fields | MAE | RMSE | Nonzero fraction |
| --- | ---: | ---: | ---: | ---: |
| Overall | 182 | 0.191723 | 0.435996 | 0.196557 |
| Boost pads | 68 | 0.492512 | 0.699794 | 0.500000 |
| Self car | 39 | 0.017848 | 0.128165 | 0.022436 |
| Opponent car | 39 | 0.018119 | 0.129675 | 0.023037 |
| Ball/relative/previous/lifecycle | 36 | 0 | 0 | 0 |

All error lies in the 74 unavailable fields. Their aggregate MAE is 0.471534 and RMSE is
0.683758. Active pad flags dominate: many have MAE near 1 because the true simulator pads are
usually active while the unavailable placeholder is zero. Per-field and per-group results are in
`simulator_reconstruction_calibration.json`.

## Missing-feature distillation interface

The teacher and student are independent modules using the unchanged 3x512 SiLU, 182-input
actor/critic. Both initialize from the byte-identical bootstrap tensors and have bit-exact actor
and critic output parity on complete observations before any future learning. Reusing a single
module as both teacher and student is rejected so a future optimizer cannot move the frozen target.
No availability
channels are appended and no availability gate is required. The quality mask remains alongside the
input for auditing, sampling, and future loss analysis; the student input itself uses the same
masked-neutral domain as human demonstrations.

The objective computes analytic teacher-to-student KL for five pre-tanh Normal channels and three
Bernoulli channels:

```text
teacher: frozen bootstrap actor(true observation), detached
student: same actor(degraded observation), differentiable
loss: mean(sum(channel KL))
```

The interface is differentiable but creates no optimizer and calls neither backward nor step.
On the 512 paired states, the unadapted bootstrap has mean teacher/student KL 9.642109 and maximum
sample KL 100.989456. Steering (2.367311) and yaw (2.167488) are the largest mean channel KLs,
followed by pitch (1.448199) and throttle (1.432240). This is a strong measured case for a future,
separately authorized missing-feature invariance/distillation stage before human BC. It is not a
claim that such robustness has already been learned.

The checkpoint file, model tensors, model parameters, gradients, iteration, policy version, and
architecture remain unchanged.

## Rebuild and validation

From the repository root:

```powershell
.\.venv\Scripts\python.exe benchmarks\build_rival2_human_demo_bc_bridge_v1.py
.\.venv\Scripts\python.exe benchmarks\build_rival2_human_demo_bc_bridge_v1.py --verify-only
.\.venv\Scripts\pytest.exe -q --basetemp .tools\pytest_human_bc_bridge `
  tests\test_human_demo_bc_observation_bridge.py `
  tests\test_human_demo_training_adapter.py `
  tests\test_human_demo_recording.py
.\.venv\Scripts\ruff.exe check rivalsim\human_demo\bc_observation_bridge.py `
  benchmarks\build_rival2_human_demo_bc_bridge_v1.py `
  tests\test_human_demo_bc_observation_bridge.py
```

Detailed source hashes, field classifications, human quality distributions, paired arrays,
reconstruction error, policy KL, no-learning audit, test evidence, and deterministic artifact
hashes live under `results/rival2/human_demo_bc_bridge_v1/`.
