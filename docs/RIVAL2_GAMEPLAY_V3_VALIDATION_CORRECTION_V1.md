# Rival2 Gameplay V3 validation correction v1

Status: `GAMEPLAY_V3_VALIDATION_CORRECTION_READY_FOR_REVIEW`

This is a validation correction and review gate only. It does not authorize
Gameplay V3 PPO training, and no PPO update or optimizer step was run.

## Repository identity

- Requested starting commit: `7e6356fc2ebeffdec4d76bf458df840de71ead34`.
- Starting `main` after fast-forward: `7e6356fc2ebeffdec4d76bf458df840de71ead34`.
- Provisionally accepted package retained: `00a4865400291a5ff0a34925a966c0963f55d963`.
- Corrected production implementation: `296ca3d126f963b3a0a375554daf637230df2a31`.
- Final validation harness: `cb5a144e1a8733bde525bd5f1d23c96227bb3484`.
- Final evidence commit and remote identity are verified after this document and
  the artifact manifest are committed.

## Correction outcome

The prior hand-authored classifier rows are superseded. The runnable `static`
phase was removed from the original validator; its implementation remains named
as an archival synthetic reproducer with no CLI dispatch. The correction runner
uses deterministic `CompleteWorldSim` CUDA Soccar physics traces and the actual
contact/resource signals consumed by production.

Each classifier has 72 physical traces: 24 positive, 24 near-miss negative, and
24 ordinary/control negative. Each class was prospectively split into 16
derivation and 8 held-out cases, giving 48 derivation and 24 untouched held-out
traces per classifier. The held-out phase ran in a separate process and required
the frozen derivation artifact and SHA-256 before prediction.

Every JSONL row records its classifier, label, split, scenario ID and seed,
generator/initial-state/action identities, measured contact/completion ticks and
features, source commit, simulator hashes, physics identity, and contract hashes.
The physical corpus is `classifier_trace_corpus.jsonl`; the derivation-only
pre-freeze corpus is retained separately for audit.

## Independently derived thresholds

All thresholds use the required nearest-edge midpoint rule. Values below are
`positive edge / negative edge / margin / selected threshold`.

| Classifier | Boundary | Positive edge | Negative edge | Margin | Selected |
|---|---|---:|---:|---:|---:|
| Contest | contact window ticks (max) | 6 | 11 | 5 | 8 |
| Contest | association ball displacement (max) | 56.647736 | 116.552399 | 59.904663 | 86.600067 |
| Contest | opponent distance (max) | 286.928436 | 574.004211 | 287.075775 | 430.466324 |
| Contest | self closing speed (min) | 1079.498413 | 392.105011 | 687.393402 | 735.801712 |
| Contest | opponent closing speed (min) | 917.676147 | 256.537811 | 661.138336 | 587.106979 |
| Contest | time-to-ball delta (max) | 0.181071 | 0.291026 | 0.109955 | 0.236049 |
| Power | total closing speed (min) | 190.642746 | 0 | 190.642746 | 95.321373 |
| Power | rotational closing speed (min) | 190.642746 | 137.151276 | 53.491470 | 163.897011 |
| Power | rotational share (min) | 0.303038 | 0.206240 | 0.096798 | 0.254639 |
| Power | ball delta-v (min) | 405.852936 | 201.392181 | 204.460754 | 303.622559 |
| Controlled | history ticks (min) | 9 | 3 | 6 | 6 |
| Controlled | max distance (max) | 200.024338 | 282.311890 | 82.287552 | 241.168114 |
| Controlled | max relative speed (max) | 6.877872 | 850 | 843.122128 | 428.438936 |
| Controlled | release distance (min) | 154.636490 | 0 | 154.636490 | 77.318245 |
| Controlled | release ball delta-v (min) | 326.832520 | 0 | 326.832520 | 163.416260 |

The frozen-threshold identity is
`92E1002905AC4D8FD3B7E1CF89EBC0D80D58D30D89E88141E036940F022E1A78`.
Because these physical derivations changed production constants and added
minimal pre-dodge control snapshots, all affected full-scale gates were rerun.
The immutable Gameplay V3 contract hash is now
`AABCA03BEBCBFC2E74EE446452781F60AC43D8C3631151BDCB7E15D0FAE13508`.

## Untouched held-out result

| Classifier | TP | TN | FP | FN | Result |
|---|---:|---:|---:|---:|---|
| Contest / 50 | 8 | 16 | 0 | 0 | PASS |
| Dodge-powered contact | 8 | 16 | 0 | 0 | PASS |
| Controlled flick | 8 | 16 | 0 | 0 | PASS |

No held-out trace was used for threshold or topology selection. The controlled
flick classifier remains exemption-only and pays zero reward.

## Source-exact dash/reset gate

All 12 bounded cases passed against
`rivalsim.gameplay_v3.gameplay_v3_track_tick`: strict tangent gain >1, <=1
rejection, >42-air rejection, >24-landing rejection, two dashes paying twice
without a third double-dash payout, no fresh-jump prohibition, three-wheel ball
and car resets, two-wheel rejection, unchanged-resource rejection, chain
loss/reacquisition, and zero extra preflip payout.

## Focused regression gate

- Ruff: PASS.
- Focused Gameplay V2/V3, reward, mechanics-calibration, and opponent-curriculum
  tests: 35 passed, 15 Torch JIT deprecation warnings.
- Executable AST scan: no `trainer.update`, `train_iteration`, or
  `optimizer.step` call in either validation runner.
- The full simulator suite was intentionally not rerun, as required by the
  correction authority.

## Exact-scale gates

- One-decision gate: 131,072 worlds, observation `(131072, 2, 182)`, finite and
  zero-sum rewards, no production evidence buffers, no hot-path H2D/D2H,
  2.358402 seconds: PASS.
- Actual V3 logical production state: 319,815,916 bytes (305.000225 MiB).
- One-decision Torch peak: 854,461,440 allocated / 991,952,896 reserved bytes.
- Horizon-32 disposable rollout: 131,072 worlds, 4.749265 seconds, finite,
  no update, model/optimizer/iteration/policy unchanged: PASS.
- Explicit V2-to-V3 transition: strict ordinary load rejected; authorized
  transition preserved model, optimizer, RNG, counters, self-play/curriculum
  state, and source checkpoint bytes: PASS.
- Reward reconstruction: 4,096 decisions, zero max absolute error, exact zero
  touch component, and runtime/frozen-threshold parity: PASS.

## No-learning shadow comparison

Both shadows use 256 episodes and the same frozen source policy/opponents.
Touches/minute (20.073142), flip-active touches/minute (12.634272), mechanic
counts, mechanics/progress ratio (0.013925), budget-hit fraction (0.001953125),
and impossible count (0) are unchanged. Corrected unnecessary flip contacts
fall from 8.376109 to 6.280237 per minute, with unnecessary flip-touch fraction
falling from 0.662967 to 0.497079 and bad-flip/progress ratio from 0.037912 to
0.028832. Exemption counts move from contest/power/controlled/recognized
`572/2/0/3` to `789/15/66/3`.

The source checkpoint remains byte-identical at
`3B994E118A9498713DC6115D38F061958A900EA8F4D00CE568F916942E851D9A`.
Model, optimizer, iteration 479, policy version 479, and sample counter
3,655,854,038 are unchanged; PPO update calls are exactly zero.

## Mandatory seven-finding audit

All findings in `handoff/rival2-gameplay-v3-production-v1/POST_COMMIT_AUDIT.md`
are PASS. The machine-readable mapping in `post_commit_audit.json` records the
evidence for retained Gameplay accumulation, fail-closed unknown modes, all V3
dispatch sites, disposable validation state, reward-identity construction,
actual-array memory accounting, and unaffected graph-capture APIs.

## Evidence index

Machine evidence is under
`results/rival2/gameplay_v3_validation_correction_v1/`. The committed-blob
identity, byte size, and SHA-256 of every reviewer artifact are recorded in
`artifact_manifest.json` and independently checked after commit.

Final correction verdict: `GAMEPLAY_V3_VALIDATION_CORRECTION_READY_FOR_REVIEW`.
