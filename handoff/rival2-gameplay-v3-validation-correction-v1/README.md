# Rival 2.0 Gameplay V3 Validation Correction V1

Status: **validation correction only; NOT training authorization**

Starting implementation/evidence commit under review: `00a4865400291a5ff0a34925a966c0963f55d963`

## BLUF

The Gameplay V3 implementation is **not rejected**, and the exact-scale runtime/crash gates are materially green. The blocker is the new anti-flip exemption calibration evidence.

The committed `classifier_calibration.json` does not contain measured RivalSim trace-derived feature rows. The validation runner creates hand-authored scalar feature dictionaries and then applies thresholds that are already hard-coded in `rivalsim/gameplay_v3.py`. That makes the reported 0 FP / 0 FN result circular and does not satisfy the handoff requirement for physical calibration.

Do not start Gameplay V3 PPO training until this correction closes that evidence gap.

## What is already accepted provisionally

Do not rewrite these merely because this correction exists:

- immutable Gameplay V3 reward identity and zero-touch arithmetic;
- historical Gameplay V1/V2 hashes and behavior;
- explicit V3 reward dispatch and retained boost/pad/save accumulation;
- native post-physics GPU detector placement;
- integer 10-event mechanics budget / 0.005 payout;
- V2 standalone double-dash isolation;
- fresh V2 -> V3 checkpoint transition architecture;
- exact 131,072-world environment construction and four-tick step;
- exact 131,072-world horizon-32 rollout-only smoke with no PPO update;
- no-learning 256-episode shadow infrastructure;
- absence of production evidence buffers;
- current 302 MiB logical V3 state is **not** a correction target in this task because the full-scale rollout completed with substantial GPU headroom.

If new measured calibration changes production classifier constants or state topology, rerun the affected gates as required below.

## Reviewer finding 1 — blocker: synthetic exemption calibration

At `00a486...`, `benchmarks/run_rival2_gameplay_v3_validation.py::static_phase()` constructs examples such as:

- `contest_train_simultaneous` with literal `opponent_distance=310`, `self_closing_speed=260`, etc.;
- `power_train_offensive` with literal closing/rotation/delta-v values;
- controlled-flick cases with literal control ticks/distances/release values.

Those values are not extracted from authoritative simulator traces. The thresholds are already constants in `rivalsim/gameplay_v3.py`, and the same hand-authored rows are then used to report confusion matrices.

Therefore the existing claims:

- contest held-out FP=0/FN=0;
- power held-out FP=0/FN=0;
- controlled-flick held-out FP=0/FN=0;

are **not sufficient calibration evidence**.

## Required correction — real physical trace corpus

For each of the three exemption classifiers, produce a compact but real deterministic RivalSim corpus. Features must be **measured from authoritative simulated states/contact data**, never typed directly into a row.

Use `24 positive + 24 near-miss negative + 24 ordinary/control negative = 72 traces per classifier` unless a physical case is impossible to realize; if so, stop and explain rather than silently shrinking to a toy corpus.

Use a prospective split per class:

- 16 derivation traces;
- 8 held-out traces.

Held-out traces must not influence thresholds/state topology.

### Contest / 50 corpus

Physically simulate and label cases covering at minimum:

- simultaneous two-car 50;
- opponent contact just before self;
- opponent contact just after self;
- genuine converging challenge without immediate opponent contact;
- distant opponent;
- nearby opponent moving away;
- nearby but non-converging opponent;
- opponent behind the play;
- uncontested loose-ball flip-through.

Measure from the same authoritative tick data used by production:

- opponent-ball distance;
- self closing speed;
- opponent closing speed;
- time-to-ball difference;
- adjacent contact tick separation;
- ball displacement across the contact-association window.

Derive the association window/displacement and convergence boundaries from the derivation traces. Do not preserve `2/300/500/150/150/0.12` merely because they are current constants.

### Dodge-powered contact corpus

Physically simulate and label:

- offensive dodge-powered shot;
- defensive dodge-powered clear;
- several contact points/dodge axes;
- weak-but-real rotationally powered contact;
- weak ordinary flip touch;
- translation-dominated high-speed hit while flipping;
- already-fast ball with negligible dodge contribution;
- normal drive-through contact controls.

Measure from the real contact:

`v_contact = v_linear + omega x r`

and export:

- total closing speed;
- translational closing contribution;
- rotational closing contribution;
- rotational share;
- authoritative ball delta-v / impulse signal;
- contact point and normal identity used to compute them.

Derive thresholds from derivation traces. Do not preserve `300/100/0.18/175` by default.

### Controlled-flick exemption corpus

This remains exemption-only and pays zero reward.

Physically simulate and label real controlled release sequences, including multiple release axes, plus hard negatives:

- front/diagonal/side controlled flick releases;
- stable roof/rear/nose control origins where feasible;
- loose-ball flip-through;
- kickoff/50;
- brief near-car relation without actual control;
- chase contact;
- controlled-looking state with no dodge-generated release.

Measure the actual pre-dodge control history and terminal release:

- control duration;
- max car-ball distance;
- max relative speed;
- release distance;
- release ball delta-v;
- active directional dodge state at release.

Derive thresholds from physical derivation traces. Do not preserve `4/220/260/245/120` by default.

If legitimate controlled flicks cannot be cleanly separated from loose flip-throughs with a high-confidence physical classifier, return `BLOCKED` rather than activating an unsafe exemption or silently removing the user-requested protection.

## Threshold-selection rule

For every scalar boundary:

1. export all measured derivation values;
2. calculate nearest positive/negative extrema;
3. if a clean directed separation exists, use midpoint of the narrowest separating margin;
4. record positive edge, negative edge, threshold, and margin;
5. if classes overlap materially, add the missing physical feature/state topology;
6. do not loosen a threshold merely to fit both classes;
7. freeze classifier constants/topology before held-out evaluation.

For conjunction classifiers, every active numerical threshold must have trace provenance. A threshold may remain numerically unchanged from V3 only if the measured derivation evidence independently derives/supports that value.

## Reviewer finding 2 — source-exact V3 dash/reset evidence is too weak

The current `deterministic_cases.json` mostly records expected constants/claims. The focused V3 test module does not provide a direct source-exact state-machine test for every new V3 dash/reset rule required by the original acceptance package.

Add focused, bounded tests for the actual V3 detector state machine. Do not create another broad simulator benchmark.

Required at minimum:

- successful dash accepts a real qualifying sequence with tangent gain >1 uu/s;
- same timing/contact sequence with <=1 uu/s gain does not pay;
- >42 tick pre-flip air rejects;
- >24 tick landing rejects;
- two distinct successful dashes within 90 ticks can each pay while double-dash label adds zero third payout;
- no fresh-jump prohibition is introduced;
- ball reset requires >=3 ball-supporting wheels and actual transition to untimed aerial resource;
- car reset requires >=3 other-car-supporting wheels;
- unchanged untimed resource does not create reset;
- chain reset cannot pay again until prior resource is consumed/lost and a distinct reacquisition occurs;
- preflip label adds no extra payout.

Use authoritative arrays/physics/state transitions. A compact kernel/state harness is acceptable where it directly exercises the production V3 kernel semantics; use full physical traces where geometry/contact identity is itself under test.

## What not to change

Do **not** in this correction:

- start PPO training;
- alter policy/observation/action cadence;
- alter V1/V2 reward behavior;
- add new reward families;
- broaden anti-flip penalty scope;
- optimize/refactor the 302 MiB production state solely for memory aesthetics;
- rerun the full simulator acceptance suite;
- replace the existing mechanics 72-case calibration with new research.

## Rerun policy

If the real trace calibration derives **identical production constants and no runtime code changes are required**:

- replace classifier calibration evidence with real trace-derived artifacts;
- add focused dash/reset tests/evidence;
- rerun focused V3/V2/mechanics tests;
- rerun the 256-episode no-learning shadow gate;
- no need to repeat the 131,072-world memory/rollout smoke unless production code changed.

If any classifier constant/state topology or V3 production detector code changes:

- update the immutable V3 contract/hash if the contract semantics/numerical thresholds are encoded there;
- rerun checkpoint-transition validation;
- rerun reward reconstruction;
- rerun exact 131,072-world one-decision + horizon-32 rollout-only smoke;
- rerun the 256-episode shadow gate;
- prove no PPO update occurred.

## Required machine evidence

Publish under a new immutable correction directory, suggested:

`results/rival2/gameplay_v3_validation_correction_v1/`

At minimum:

- `classifier_trace_corpus.jsonl` — trace IDs, labels, split, scenario provenance, measured feature values;
- `classifier_threshold_derivation.json` — extrema, margins, thresholds, classifier version;
- `classifier_heldout.json` — untouched held-out predictions/confusion;
- `dash_reset_source_exact.json` — focused production state-machine cases;
- `shadow_gate_summary.json` — rerun no-learning shadow metrics;
- `shadow_event_evidence.json` — bounded event evidence;
- `regression_tests.json`;
- `artifact_manifest.json` bound to committed blobs.

Every trace must include enough provenance to reproduce it: source commit, simulator/contract hashes, seed, scenario ID, initial-state or generator identity, action-sequence identity, and measured completion/contact ticks.

## Final verdict

Return exactly one:

`GAMEPLAY_V3_VALIDATION_CORRECTION_READY_FOR_REVIEW`

or

`BLOCKED: <specific reason>`

This is still **not training authorization**. Stop after pushing the correction evidence/code and returning the reviewer package.
