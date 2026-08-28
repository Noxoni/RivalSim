# Codex Start Prompt — Gameplay V3 Validation Correction V1

Work in `Noxoni/RivalSim` on latest `origin/main`.

This is a **validation correction only**. Do not start Gameplay V3 PPO training.

## 1. Integrity

Before editing:

1. `git fetch origin`;
2. verify `00a4865400291a5ff0a34925a966c0963f55d963` is an ancestor of current HEAD;
3. record starting HEAD and clean/preserved worktree state;
4. preserve concurrent work and never force-push `main`.

## 2. Read authority

Read completely:

- `handoff/rival2-gameplay-v3-validation-correction-v1/README.md`
- the original `handoff/rival2-gameplay-v3-production-v1/` package
- `docs/RIVAL2_GAMEPLAY_V3_VALIDATION.md`
- current production V3 source and validation artifacts.

The correction README overrides the prior `GAMEPLAY_V3_READY_FOR_REVIEW` verdict where it identifies missing evidence.

## 3. Core problem

The current contest, dodge-powered-contact, and controlled-flick calibration rows are hand-authored scalar dictionaries in `benchmarks/run_rival2_gameplay_v3_validation.py::static_phase()`. They are not measured RivalSim traces. The thresholds are already constants in `rivalsim/gameplay_v3.py`, so the reported 0 FP / 0 FN results are circular.

Replace that evidence with **actual deterministic RivalSim physics traces**.

Do not type target feature values into calibration rows.

## 4. Required physical corpus

For each exemption classifier generate:

- 24 clear positive physical traces;
- 24 near-miss physical negatives;
- 24 ordinary/control physical negatives.

Use 16 derivation + 8 untouched held-out traces per class.

All classifier inputs must be measured from the same authoritative simulator/contact/resource signals consumed by production V3.

### Contest/50

Cover simultaneous/adjacent contacts, real convergence, distant/nonconverging/moving-away/behind-play/uncontested cases. Measure distance, closing speeds, time-to-ball delta, adjacent contact separation, and ball displacement. Derive all active numerical boundaries/window from derivation traces.

### Dodge-powered contact

Cover offensive shots, defensive clears, varied contact geometry, weak-real rotational power, weak ordinary flip touches, translation-dominated hits, already-fast-ball negligible-dodge contacts, and drive-through controls. Measure `v_contact = v_linear + omega x r`, translational vs rotational contribution, rotational share, ball delta-v/impulse, contact point/normal. Derive all active boundaries from derivation traces.

### Controlled flick exemption

This remains zero-reward exemption only. Cover physically simulated front/diagonal/side controlled releases and hard loose-ball/kickoff/brief-control/chase/no-release negatives. Measure actual control history, distance, relative speed, release distance, ball delta-v, and active directional dodge at release. Derive all active boundaries from derivation traces.

If any classifier cannot be cleanly separated without broadening into an unsafe rule, return `BLOCKED`.

## 5. Threshold derivation

Do not preserve existing constants by default.

For every threshold record derivation positive edge, negative edge, margin, selected threshold, and direction. Use midpoint of the narrowest clean separating margin when possible. If classes overlap, add a missing physical feature/state transition rather than fitting held-out data.

Freeze thresholds/topology before held-out evaluation.

The held-out split must not be inspected or tuned against before freeze.

## 6. Close the V3 dash/reset source-exact evidence gap

Add focused tests/evidence for the actual production V3 state machine:

- valid successful dash >1 uu/s tangent gain;
- <=1 uu/s near miss rejects;
- >42 air rejects;
- >24 landing rejects;
- two qualifying dashes within 90 ticks pay twice total, double-dash label no third payout;
- no fresh-jump prohibition;
- ball reset >=3 ball-supporting wheels + real untimed-resource transition;
- car reset >=3 other-car-supporting wheels;
- unchanged untimed resource rejects;
- chain reset requires consume/loss then distinct reacquisition;
- preflip subtype adds zero extra payout.

Keep this bounded. Do not turn it into a broad simulator benchmark.

## 7. Preserve already-green implementation/runtime work

Do not rewrite the V3 implementation merely because validation evidence was inadequate.

Provisionally preserve:

- V1/V2 compatibility;
- V3 reward arithmetic;
- native GPU post-physics integration;
- checkpoint transition architecture;
- 131,072-world configuration;
- mixed PPO safety state;
- 10-event mechanics budget;
- current mechanics reward set.

Do not optimize the ~302 MiB production V3 state in this task. Existing exact-scale rollout had substantial memory headroom; a refactor now would add unrelated risk.

## 8. Rerun rules

If measured calibration independently derives the current constants and production code stays byte-identical except validation/tests/evidence:

- rerun focused tests;
- rerun 256-episode no-learning shadow gate;
- do not repeat full 131k rollout merely for ceremony.

If production constants/topology/code change:

- update V3 identity/contract hash where required;
- rerun checkpoint transition;
- rerun reward reconstruction;
- rerun exact 131,072-world one-decision and horizon-32 rollout-only crash/memory gates;
- rerun 256-episode no-learning shadow;
- prove no PPO update.

## 9. Evidence

Publish immutable correction evidence under:

`results/rival2/gameplay_v3_validation_correction_v1/`

including at minimum:

- `classifier_trace_corpus.jsonl`
- `classifier_threshold_derivation.json`
- `classifier_heldout.json`
- `dash_reset_source_exact.json`
- `shadow_gate_summary.json`
- `shadow_event_evidence.json`
- `regression_tests.json`
- `artifact_manifest.json`

Every physical trace must carry reproducibility provenance: source commit, simulator/contract hashes, seed, scenario ID, initial-state/generator identity, action sequence identity, and measured contact/completion ticks.

## 10. Hard no-training boundary

Forbidden:

- `trainer.update(...)`
- `train_iteration(...)`
- any PPO update
- optimizer `.step()`
- saving a validation rollout as campaign continuation.

No Gameplay V3 PPO training in this task.

## 11. Return

Push normally to `main` and return:

- final SHA;
- files changed;
- physical corpus counts/provenance by classifier;
- derivation extrema/margins/thresholds;
- untouched held-out confusion matrices;
- whether any production constant/code changed;
- focused dash/reset test results;
- shadow metric comparison against the prior `00a486...` shadow;
- rerun crash/memory gates if production changed;
- confirmation no PPO update/training occurred;
- final verdict exactly:

`GAMEPLAY_V3_VALIDATION_CORRECTION_READY_FOR_REVIEW`

or

`BLOCKED: <specific reason>`

Stop there.
