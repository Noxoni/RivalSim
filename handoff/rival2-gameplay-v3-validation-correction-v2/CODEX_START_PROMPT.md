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

The V2 README contains an authoritative user clarification about contest contact order. Follow it exactly.

## 3. Contest rule — do not misinterpret this

**Opponent-first contact is NOT a penalty condition.**

If the opponent hits the ball first and Rival's later contact belongs to the same physical 50/challenge, Rival's contact must be classified as `EXEMPT_CONTESTED_50` and receive no bad-flip penalty.

Likewise, Rival-first followed by opponent contact in the same physical contest is also exempt.

Contact order has zero negative reward meaning. Do not create:

- a first-touch penalty;
- a last-touch penalty;
- a reward for beating the opponent to first contact;
- any shaping that treats opponent-first as worse.

The recent-opponent-contact memory requested below exists only to recognize legitimate opponent-first 50s and **suppress** the penalty.

## 4. Fix only the two blockers

### A. Symmetric contest exemption association

The V1 physical corpus case named `opponent_contact_just_before_self` actually measured Rival/self first and opponent later in the examples reviewed. Current production also only remembers future opponent contacts after a Rival candidate exists.

Add minimal authoritative recent-opponent-contact state so a real opponent contact occurring shortly **before** Rival's flip-touch is recognized as affirmative `EXEMPT_CONTESTED_50` evidence when it belongs to the same calibrated contest event.

Retain forward association for opponent contacts occurring shortly after Rival.

Regenerate contest physical traces with measured-order assertions. Include real opponent-before-self and opponent-after-self positives in both derivation and held-out splits.

Do not use scenario names as truth; measured contact ticks are authority.

Do not use broad opponent proximity as a substitute.

### B. Controlled-flick release timing parity

V1 calibration measured release distance at contact +2 ticks, while production currently uses current distance when the shared contest pending window resolves at up to 8 ticks.

Align calibration and runtime exactly.

Prefer a separate calibrated controlled-release state/window or capture the physical release transition independently while the candidate remains pending for contest association.

Do not preserve a +2-derived release threshold against +8 runtime semantics.

If absolute distance is insufficient, add the missing physical release feature/state topology rather than broadening the exemption.

Controlled flick remains exemption-only with zero positive reward.

## 5. Mandatory 216-trace production runtime parity

Do not stop at offline `_classify()` confusion matrices.

Replay every recorded physical classifier scenario through the actual production Gameplay V3 detector/state machine and compare:

- physical label;
- frozen offline result;
- production candidate;
- primary outcome;
- exemption flags;
- contact/order ticks;
- runtime-captured feature values at calibrated sampling times.

Require production feature parity within documented float tolerance and zero held-out FP/FN through the actual runtime path.

Explicitly report contest positive counts by measured order:

- opponent-before-self;
- opponent-after-self;
- simultaneous/closest physically representable;
- convergence-only.

Both before-self and after-self must have derivation and untouched held-out positives, and **both must resolve as contest exemptions**.

Also prove there is no reward component, telemetry outcome, or classifier branch that treats opponent-first as a negative event.

## 6. Preserve everything else

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
No contact-order shaping.

## 7. Rerun required gates

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

## 8. Return

Push normally to `main` and return:

1. final SHA;
2. files changed;
3. exact symmetric contest exemption implementation;
4. measured-order corpus counts and examples;
5. proof opponent-first and opponent-after both exempt with no order penalty;
6. controlled-release timing/state definition;
7. any rederived thresholds and margins;
8. untouched held-out confusion;
9. 216-trace production-runtime parity result;
10. focused regression/dash-reset results;
11. V3 contract hash;
12. checkpoint/reward reconstruction results;
13. exact-scale smoke results;
14. corrected 256-episode shadow comparison;
15. confirmation no PPO update/training occurred;
16. final verdict exactly:

`GAMEPLAY_V3_VALIDATION_CORRECTION_V2_READY_FOR_REVIEW`

or

`BLOCKED: <specific reason>`

Stop there. **Do not start Gameplay V3 training.**
