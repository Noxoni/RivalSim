# Rival 2.0 Gameplay V3 Validation Correction V2

Status: **final production/runtime-parity correction; NOT training authorization**

Reviewed source/evidence commit: `5efa83f331855ae86a8076b7c0c1a9dc8fae88c4`

## BLUF

The V1 correction successfully replaced the synthetic classifier rows with real deterministic RivalSim physics traces, derived the thresholds prospectively, passed held-out classification, closed the dash/reset source-exact gate, and reran the exact-scale runtime validation.

Do **not** undo that work.

Two remaining mismatches prevent training authorization:

1. the contest corpus does not actually contain the claimed opponent-contact-before-self physical ordering, while production cannot remember a prior opponent ball contact;
2. controlled-flick `release_distance` is calibrated at contact +2 physics ticks but production evaluates current car-ball distance when the shared contest pending window resolves at up to 8 ticks.

The correction must align the physical corpus and production state machine exactly, then prove end-to-end runtime parity on the complete 216-trace classifier corpus.

## Finding 1 — contest contact association is not bidirectional

The committed physical corpus exposes a labeling/topology mismatch.

For example:

- `contest-positive-D01` is named `opponent_contact_just_before_self`, but measured contact ticks are self `18`, opponent `21`;
- `contest-positive-H17` has the same pattern: self `18`, opponent `21`.

The physical event therefore has the opponent contacting **after** Rival, not before.

Current production `gameplay_v3_track_tick` creates `pending_active` only when Rival's flip-touch candidate occurs. It can mark a later `other_reports` contact during the pending window, but it has no authoritative recent-opponent-contact tick/ball-position state to associate an opponent contact which genuinely happened shortly **before** Rival's candidate.

A true opponent-first 50 can therefore be penalized unless the independent convergence classifier also happens to pass.

### Required correction

Implement symmetric adjacent-contact association at 120 Hz.

A valid design should retain minimal GPU-resident recent legitimate ball-contact state per car/world, sufficient to answer at candidate creation:

- did the opponent have a legitimate ball-contact onset within the calibrated association window immediately before this contact?;
- is the ball displacement from that opponent contact within the calibrated association displacement boundary?;

Then retain the existing forward pending check for opponent contacts occurring after Rival's contact.

Do not use a broad proximity substitute.

Do not alter the historical Rival touch latch.

The association must remain same-event bounded and must reset cleanly at episode reset.

### Physical corpus correction

Regenerate/revise contest traces so scenario names are verified against **measured** contact order, not intended initial geometry.

Required positive physical order classes:

- same-tick or closest physically representable simultaneous contest;
- measured opponent-before-self: `opponent_tick < self_tick`;
- measured opponent-after-self: `opponent_tick > self_tick`;
- convergence-only challenge where adjacent opponent contact is absent.

If exact same-tick dual contact is not representable because of authoritative serial collision ordering, document the closest physically meaningful simultaneous-contest topology rather than falsely naming it same-tick.

For every ordered-contact positive, assert the measured ordering before accepting the trace into the corpus.

Retain hard delayed/unrelated contact negatives on both sides of the candidate where feasible.

Re-derive contact-association window/displacement if the corrected corpus changes the extrema. Do not preserve 8 ticks / 86.600067 uu by default.

## Finding 2 — controlled-flick release timing differs between calibration and production

The V1 correction physical calibration computes:

`release_tick = selected_contact_tick + 2`

and derives `CONTROL_RELEASE_DISTANCE_MIN` from that measurement.

Production does not evaluate that same quantity at +2. A flip-touch candidate remains pending for the shared contest window and `_resolve_flip_candidate` evaluates the **current** car-ball distance when the candidate resolves. `CONTEST_CONTACT_WINDOW_TICKS` is now 8.

Therefore `CONTROL_RELEASE_DISTANCE_MIN = 77.3182449341` is not calibrated for the quantity production actually uses.

### Required correction

Make controlled-flick release identity explicit and identical in calibration and runtime.

Preferred architecture:

- give controlled flick its own bounded release-evaluation state/window rather than accidentally inheriting the contest-association timeout;
- derive that release window from physical traces if a numerical timing boundary is needed;
- capture the same release evidence production will evaluate (distance and/or a stronger physical exit-from-control feature) at the exact same event/time definition;
- let the overall flip candidate remain pending long enough for contest association without changing the already-captured controlled-flick release evidence.

Alternative: if production deliberately evaluates release at the final contest-resolution tick, recalibrate every controlled-flick trace using that exact runtime timing and re-derive the threshold. Do not retain +2-derived constants against +8 runtime semantics.

The classifier must still prove:

- real controlled relation before the dodge;
- active directional dodge;
- legitimate dodge contact;
- meaningful ball transfer;
- physical release from the pre-dodge controlled relation.

It remains exemption-only and pays zero mechanics reward.

Do not make release identity trivially true merely because time passed after a contact. If absolute distance is insufficient to distinguish release, add an outward relative-motion/state-transition feature from the physical traces rather than broadening the exemption.

## Mandatory end-to-end production parity gate

The prior correction held-out gate applies an offline `_classify(row, thresholds)` function to extracted features. That is useful but insufficient to prove the production state machine computes the same classification.

After this correction, run every physical classifier trace through the actual production Gameplay V3 detector path using the recorded initial state/action sequence.

For all 216 traces compare:

- expected physical label;
- offline frozen classifier result;
- production candidate existence;
- production primary outcome;
- applicable exemption flags;
- contact/completion ticks;
- captured production feature values versus calibration-extracted features at the defined sampling ticks.

Required:

- production/offline feature parity within documented float tolerance;
- production outcome matches frozen physical label for the calibrated classifier under test;
- held-out FP=0/FN=0 remains true **through the production runtime state machine**, not only offline feature replay.

For contest traces, explicitly report counts by measured contact order:

- opponent-before-self;
- opponent-after-self;
- simultaneous/closest representable;
- convergence-only.

At least one derivation and one held-out positive must exist for both actual before-self and after-self ordering. Prefer more than one; preserve the 24-positive class total.

## Preserve already-green work

Do not change unless required by these two fixes:

- Gameplay V1/V2 behavior/hashes;
- Gameplay V3 reward arithmetic;
- `-0.01` bad-flip penalty magnitude;
- `+0.005` mechanics payout and 10-event budget;
- rewardable mechanics set;
- continuous mechanics thresholds;
- dash/reset state machine;
- 30 Hz policy / 120 Hz physics cadence;
- checkpoint/mixed-PPO architecture;
- opponent mix;
- V3 memory layout except the minimal arrays required for bidirectional contest/release state.

Do not optimize the V3 state broadly.

## Rerun requirements

Production code/state and likely contract semantics will change, so rerun:

1. corrected physical derivation + separately frozen held-out classifier corpus;
2. full 216-trace production-runtime parity gate;
3. focused Gameplay V2/V3/reward/mechanics/curriculum tests;
4. dash/reset source-exact gate (regression only; no redesign);
5. immutable V3 contract/hash update if classifier topology/timing is encoded in the contract;
6. checkpoint transition validation;
7. exact reward reconstruction;
8. exact `131,072`-world one-decision smoke;
9. exact `131,072`-world horizon-32 rollout-only smoke with no update;
10. 256-episode no-learning shadow gate;
11. committed artifact manifest/readback audit.

No PPO update is permitted.

## Shadow review requirements

Compare against the V1 correction shadow at `5efa83f...`.

Report:

- touches/min;
- flip-active touches/min;
- unnecessary flip contacts/min;
- unnecessary fraction of flip touches;
- contest exemptions;
- controlled-flick exemptions;
- power exemptions;
- recognized-mechanic exemptions;
- mechanics counts/reward scale;
- bad-flip/progress ratio;
- budget-hit fraction;
- impossible/jitter count.

Additionally export bounded evidence for:

- real opponent-before-self contest exemptions;
- real opponent-after-self contest exemptions;
- controlled-flick exemptions with the exact calibrated release evidence/tick;
- unnecessary flip contacts near each classifier boundary.

## Machine evidence

Publish under a new immutable directory:

`results/rival2/gameplay_v3_validation_correction_v2/`

At minimum include:

- `classifier_trace_corpus.jsonl`
- `classifier_threshold_derivation.json`
- `classifier_heldout.json`
- `production_runtime_parity.json`
- `contest_order_evidence.json`
- `controlled_release_evidence.json`
- `dash_reset_source_exact.json`
- `contract.json`
- `checkpoint_transition.json`
- `reward_reconstruction.json`
- `memory_smoke.json`
- `shadow_gate_summary.json`
- `shadow_event_evidence.json`
- `regression_tests.json`
- `no_training_assertion.json`
- `artifact_manifest.json`

Bind artifacts to committed blobs/content as before.

## Stop conditions

Return `BLOCKED` rather than training if:

- true opponent-before-self physical contests cannot be represented/protected;
- production and offline classifier results differ on the physical corpus;
- controlled-flick release identity cannot be aligned cleanly;
- the corrected held-out classifier produces FP/FN;
- V1/V2 hashes change;
- reward reconstruction fails;
- exact-scale smoke fails;
- checkpoint transition preservation fails.

## Final verdict

Return exactly one:

`GAMEPLAY_V3_VALIDATION_CORRECTION_V2_READY_FOR_REVIEW`

or

`BLOCKED: <specific reason>`

This remains **not training authorization**. Stop after pushing the correction package/evidence.