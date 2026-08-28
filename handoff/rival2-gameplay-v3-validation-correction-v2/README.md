# Rival 2.0 Gameplay V3 Validation Correction V2

Status: **final production/runtime-parity correction; NOT training authorization**

Reviewed source/evidence commit: `5efa83f331855ae86a8076b7c0c1a9dc8fae88c4`

## User clarification — authoritative

**Contact order is never a penalty condition.**

Gameplay V3 must never penalize Rival merely because the opponent touched the ball first.

In a normal controlled 50, allowing the opponent to put the ball into Rival and then taking the later/last contact can be the correct play. Therefore:

- opponent-first -> Rival contact, when both contacts belong to the same physical contest, is `EXEMPT_CONTESTED_50`;
- Rival-first -> opponent contact, when both contacts belong to the same physical contest, is also `EXEMPT_CONTESTED_50`;
- closest physically representable simultaneous contact is also `EXEMPT_CONTESTED_50`;
- contact order by itself has **zero negative reward meaning**;
- there is no first-touch/last-touch penalty;
- there is no reward for beating the opponent to the ball merely because Rival touched first.

The only negative event remains `UNNECESSARY_FLIP_THROUGH_CONTACT`: Rival performs an active directional dodge through a ball contact that is not a legitimate contest, dodge-powered contact, controlled flick, or recognized same-contact mechanic.

Recent-opponent-contact state is required only so production can recognize an opponent-first 50 and **suppress** that penalty. It must never create a penalty.

This clarification overrides any earlier wording that could be read as treating opponent-before-self as undesirable.

## BLUF

The V1 correction successfully replaced the synthetic classifier rows with real deterministic RivalSim physics traces, derived thresholds prospectively, passed held-out classification, closed the dash/reset source-exact gate, and reran exact-scale runtime validation.

Do **not** undo that work.

Two runtime/calibration mismatches remain:

1. contest association must recognize both physical contact orders so legitimate opponent-first 50s are protected;
2. controlled-flick `release_distance` is calibrated at contact +2 physics ticks while production evaluates current car-ball distance when the shared contest pending window resolves at up to 8 ticks.

The correction must align the physical corpus and production state machine exactly, then prove end-to-end runtime parity on the complete 216-trace classifier corpus.

## Finding 1 — contest association must protect either contact order

The committed V1 physical corpus exposes a scenario-label/topology mismatch.

For example:

- `contest-positive-D01` is named `opponent_contact_just_before_self`, but measured contact ticks are self `18`, opponent `21`;
- `contest-positive-H17` has the same measured order: self `18`, opponent `21`.

Those traces are opponent-after-self events despite their names.

Current production creates `pending_active` only when Rival's flip-touch candidate occurs. It can associate a later opponent contact during the pending window, but it cannot currently associate a legitimate opponent contact that happened shortly **before** Rival's candidate.

That is a problem because an opponent-first 50 is a legitimate contest and must be exempt from the anti-flip penalty.

### Required correction

Implement symmetric adjacent-contact **exemption** association at 120 Hz.

Retain minimal GPU-resident recent legitimate opponent ball-contact state sufficient to answer at Rival candidate creation:

- did the opponent have a legitimate ball-contact onset within the calibrated association window immediately before Rival's contact?;
- is the ball displacement from that opponent contact within the calibrated association displacement boundary?;

If yes, mark the candidate as contested. This is affirmative exemption evidence.

Retain the existing forward pending check for opponent contacts occurring after Rival's contact.

Rules:

- opponent-first associated contact -> `EXEMPT_CONTESTED_50`;
- opponent-after associated contact -> `EXEMPT_CONTESTED_50`;
- order must not affect reward sign or desirability;
- do not use broad opponent proximity as a substitute;
- do not alter the historical Rival touch latch;
- association must be same-event bounded and reset cleanly on episode reset.

### Physical corpus correction

Regenerate/revise contest traces so scenario names are verified against **measured** contact order, not intended initial geometry.

Required positive physical classes:

- same-tick or closest physically representable simultaneous contest;
- measured opponent-before-self: `opponent_tick < self_tick`;
- measured opponent-after-self: `opponent_tick > self_tick`;
- convergence-only challenge where adjacent opponent contact is absent.

For every ordered-contact positive, assert measured ordering before accepting the trace.

At least one derivation and one untouched held-out positive must exist for both actual opponent-before-self and opponent-after-self order. Prefer several while preserving the 24-positive class total.

If exact same-tick dual contact is not representable because of authoritative serial collision ordering, document the closest physically meaningful simultaneous topology instead of naming it same-tick.

Retain hard delayed/unrelated contact negatives on both sides of the candidate where feasible.

Re-derive the association window/displacement if corrected measured-order traces change the extrema. Do not preserve `8 ticks / 86.600067 uu` by default.

## Finding 2 — controlled-flick release timing differs between calibration and production

The V1 correction physical calibration computes:

`release_tick = selected_contact_tick + 2`

and derives `CONTROL_RELEASE_DISTANCE_MIN` from that measurement.

Production does not evaluate that same quantity at +2. A flip-touch candidate remains pending for the shared contest window and `_resolve_flip_candidate` evaluates the current car-ball distance when the candidate resolves. `CONTEST_CONTACT_WINDOW_TICKS` is currently 8.

Therefore the calibrated release threshold and runtime quantity are not yet the same measurement.

### Required correction

Make controlled-flick release identity explicit and identical in calibration and runtime.

Preferred architecture:

- give controlled flick its own bounded release-evaluation state/window rather than inheriting contest timeout semantics;
- derive any release timing boundary from physical traces;
- capture the same distance/outward-motion/release evidence production will evaluate at the exact same event/time definition;
- allow the overall candidate to remain pending for contest association without losing the already-captured controlled-flick release evidence.

Alternative: deliberately evaluate controlled release at final contest resolution and recalibrate every controlled-flick trace using that exact timing. Do not retain +2-derived constants against +8 runtime semantics.

The classifier must still prove:

- real controlled relation before dodge;
- active directional dodge;
- legitimate dodge contact;
- meaningful ball transfer;
- physical release from the pre-dodge controlled relation.

Controlled flick remains exemption-only and pays zero positive mechanics reward.

If absolute distance is insufficient, add an outward relative-motion/state-transition feature from real traces rather than broadening the exemption.

## Mandatory end-to-end production parity gate

Offline `_classify(row, thresholds)` confusion matrices are insufficient by themselves.

After correction, replay every physical classifier scenario through the actual production Gameplay V3 detector/state machine using its recorded initial state/action sequence.

For all 216 traces compare:

- expected physical label;
- frozen offline result;
- production candidate existence;
- production primary outcome;
- applicable exemption flags;
- contact/completion/order ticks;
- captured production feature values versus calibration-extracted features at the defined sampling ticks.

Required:

- production/offline feature parity within documented float tolerance;
- production outcome matches the frozen physical label for the classifier under test;
- held-out FP=0/FN=0 through the production runtime state machine, not only offline replay.

For contest positives explicitly report measured counts for:

- opponent-before-self;
- opponent-after-self;
- simultaneous/closest representable;
- convergence-only.

Again: **both valid contact orders are exemptions. Neither is a penalty condition.**

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
- V3 memory layout except minimal arrays required for contest/release state.

Do not add first-touch reward, last-touch reward, or any contact-order shaping.
Do not optimize the V3 state broadly.

## Rerun requirements

Production code/state and likely contract semantics will change, so rerun:

1. corrected physical derivation + separately frozen held-out classifier corpus;
2. full 216-trace production-runtime parity gate;
3. focused Gameplay V2/V3/reward/mechanics/curriculum tests;
4. dash/reset source-exact gate as regression only;
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
- controlled-flick exemptions with exact calibrated release evidence/tick;
- unnecessary flip contacts near each classifier boundary.

Verify there is **no counter, reward component, penalty, or feature whose meaning is "opponent touched first" as a negative outcome.**

## Machine evidence

Publish under:

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

- legitimate opponent-first physical contests are not protected as contest exemptions;
- legitimate opponent-after physical contests are not protected as contest exemptions;
- any contact-order-specific negative reward/penalty is introduced;
- production and offline classifier results differ on the physical corpus;
- controlled-flick release identity cannot be aligned cleanly;
- corrected held-out classifier produces FP/FN;
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
