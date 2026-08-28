# Codex Start Prompt — Gameplay V3 Validation Correction V2

Work in `Noxoni/RivalSim` on latest `origin/main`.

This is a **final validation/runtime-parity correction**. Do not start Gameplay V3 PPO training.

## 1. Integrity

Before editing:

1. `git fetch origin`;
2. verify reviewed correction commit `5efa83f331855ae86a8076b7c0c1a9dc8fae88c4` is an ancestor of current HEAD;
3. record starting HEAD/worktree;
4. preserve concurrent work;
5. never force-push `main`.

## 2. Read authority

Read completely:

- `handoff/rival2-gameplay-v3-validation-correction-v2/README.md`
- `handoff/rival2-gameplay-v3-validation-correction-v1/README.md`
- original `handoff/rival2-gameplay-v3-production-v1/` package
- current V3 production source
- V1 correction physical corpus/evidence at `results/rival2/gameplay_v3_validation_correction_v1/`.

Correction V2 overrides the prior READY_FOR_REVIEW verdict only for the two runtime/calibration mismatches it identifies.

## 3. Fix only the two blockers

### A. Bidirectional contest association

The V1 physical corpus case named `opponent_contact_just_before_self` actually measured Rival/self first and opponent later (for example self tick 18, opponent tick 21). Current production also only remembers future opponent contacts after a Rival candidate exists.

Add minimal authoritative recent-opponent-contact state so a real opponent contact occurring shortly **before** Rival's flip-touch can suppress the penalty when it belongs to the same calibrated contest event.

Regenerate contest physical traces with measured-order assertions. Include real before-self and after-self positives in both derivation and held-out splits.

Do not use scenario names as truth; measured contact ticks are authority.

### B. Controlled-flick release timing parity

V1 calibration measured release distance at contact +2 ticks, but production currently uses current distance when the shared contest pending window resolves at up to 8 ticks.

Align calibration and runtime exactly.

Prefer a separate calibrated controlled-release state/window or capture the physical release transition independently while the candidate remains pending for contest association.

Do not preserve a +2-derived release threshold against +8 runtime semantics.

If absolute distance is insufficient, add the missing physical release feature/state topology rather than broadening the exemption.

Controlled flick remains exemption-only with zero positive reward.

## 4. Mandatory 216-trace production runtime parity

Do not stop at offline `_classify()` confusion matrices.

Replay every recorded physical classifier scenario through the actual production Gameplay V3 detector/state machine and compare:

- physical label;
- frozen offline result;
- production candidate;
- primary outcome;
- exemption flags;
- contact/order ticks;
- runtime-captured feature values at the calibrated sampling times.

Require production feature parity within documented float tolerance and zero held-out FP/FN through the actual runtime path.

Explicitly report contest positive counts by measured order:

- opponent-before-self;
- opponent-after-self;
- simultaneous/closest physically representable;
- convergence-only.

Both before-self and after-self must have derivation and untouched held-out positives.

## 5. Preserve everything else

Do not redesign:

- reward arithmetic;
- penalty magnitude;
- mechanics reward amount/budget/families;
- dash/reset semantics;
- V1/V2;
- policy/PPO/obs/action cadence;
- opponent curriculum;
- general V3 state layout beyond minimal state required by this correction.

No memory-optimization project.

## 6. Rerun required gates

Because production state/code will change, rerun all affected release gates exactly as required in the V2 README:

- physical derivation and separate held-out;
- production runtime parity on all 216 traces;
- focused regressions;
- dash/reset regression gate;
- V3 contract/hash as needed;
- checkpoint transition;
- reward reconstruction;
- exact 131072-world one-decision smoke;
- exact 131072-world horizon-32 rollout-only smoke;
- 256-episode no-learning shadow;
- artifact manifest/readback audit.

No PPO update, no optimizer step, no campaign checkpoint.

## 7. Return

Push normally to `main` and return:

1. final SHA;
2. files changed;
3. exact contest previous/future association implementation;
4. measured-order corpus counts and examples;
5. controlled-release timing/state definition;
6. any rederived thresholds and margins;
7. untouched held-out confusion;
8. 216-trace production-runtime parity result;
9. focused regression/dash-reset results;
10. V3 contract hash;
11. checkpoint/reward reconstruction results;
12. exact-scale smoke results;
13. corrected 256-episode shadow comparison;
14. confirmation no PPO update/training occurred;
15. final verdict exactly:

`GAMEPLAY_V3_VALIDATION_CORRECTION_V2_READY_FOR_REVIEW`

or

`BLOCKED: <specific reason>`

Stop there. **Do not start Gameplay V3 training.**