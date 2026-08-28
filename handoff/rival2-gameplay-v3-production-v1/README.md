# Rival 2.0 Gameplay V3 Production Handoff

Status: **implementation + validation authority; NOT training authorization**

Package version: `rival2-gameplay-v3-production-v1`

Audited baseline commit: `0228a833e90d2db2715f8b79b65f6cbdc59fefbc`

## BLUF

Implement **one** immutable reward transition to `RIVAL2_REWARD_GAMEPLAY_V3` that combines:

1. Gameplay V1's healthy game shaping;
2. removal of the unconditional `+0.05` unique-touch payout;
3. the approved low-value mechanics reward system (`+0.005/event`, hard `0.05` per-player episode budget);
4. a narrowly classified `UNNECESSARY_FLIP_THROUGH_CONTACT` penalty (`-0.01` to the offending player before zero-sum composition), with calibrated exemptions for genuine contests, dodge-powered contacts, controlled flicks, and same-contact recognized mechanics.

Do **not** start PPO training in this task. The task ends after implementation, focused calibration/tests, production-scale smoke, and no-learning shadow reward reconstruction.

## Why this package exists

The change is large enough that a superficially correct implementation can still crash or corrupt training through kernel ABI mismatches, bad reset ordering, duplicated rewards, oversized GPU state, or checkpoint/reward-version incompatibility.

The current code was audited before this handoff was written. Important facts:

- `Rival2WorldSim._launch_tick()` executes authoritative physics first, then the Gameplay V2 strict-double-dash tracker, then `rival2_accumulate_tick()`.
- `rival2_accumulate_tick()` owns the current unique-touch latch/count and computes the 30 Hz reward on tick four.
- Gameplay V2's standalone strict-double-dash reward is embedded directly in that reward path.
- The calibrated mechanics system is currently a **read-only post-physics observer** attached by wrapping `world._launch_tick`; it is not production reward code.
- The calibration observer allocates diagnostic/evidence storage that must not be copied wholesale into the 131,072-world training path.
- `Rival2Trainer.load_checkpoint()` is contract-strict. Reward changes require an explicit curriculum-transition path.
- The mixed-opponent trainer checkpoint contains additional curriculum RNG, adapter temporal state, adaptive PPO state, split optimizer state, and retention corpus state that must not be silently lost.
- Published mixed-opponent training uses `131,072` worlds and the short `RIVAL2_EPISODE_V1` lifecycle.

## Known latest accepted source candidate

The latest accepted checkpoint recorded in the audited repository is:

- label: `plus_120`
- iteration: `479`
- policy version: `479`
- agent decision samples: `3,655,854,038`
- reward: `RIVAL2_REWARD_GAMEPLAY_V2`
- SHA-256: `3B994E118A9498713DC6115D38F061958A900EA8F4D00CE568F916942E851D9A`
- recorded path: `G:/dev/RivalSim-runs/opponent-curriculum-v1-safe-20260827-b2af03d/checkpoints/rival2_opponent_curriculum_plus_120_resume.pt`

This is a **candidate**, not permission to guess. Before validation, Codex must check whether a newer accepted local checkpoint exists. If a newer accepted checkpoint exists, use it only after proving its identity/audit status. If the intended source cannot be established safely, return `BLOCKED` rather than falling back silently to Gameplay V1 +239.

## Authority order

For this task, use this order:

1. this package;
2. `docs/RIVAL2_MECHANICS_CALIBRATION_TARGETED_CORRECTION_V1.md` and committed calibration artifacts;
3. `docs/RIVAL2_MECHANICS_CALIBRATION_IMPLEMENTATION_V0_1.md`;
4. `docs/RIVAL2_MECHANICS_REWARD_CONTRACT_V0_1.md`;
5. `docs/RIVAL2_MECHANICS_DETECTOR_PHYSICS_V0_1.md`;
6. `docs/RIVAL2_MECHANICS_FLICK_VARIANTS_V0_1.md`;
7. `docs/RIVAL2_MECHANICS_CATALOG_V0_1.md`;
8. current source behavior for historical V1/V2 compatibility.

If a contradiction remains after reading those sources, stop and report it. Do not resolve it by intuition.

## Frozen reward-ready mechanic set for Gameplay V3

Gameplay V3 initially rewards only the following explicitly approved mechanics:

### Continuous calibrated

- speedflip;
- half-flip;
- Musty;
- Breezi;
- redirect;
- pinch;
- pogo.

### Source-exact / topology-backed

- successful wavedash/dash completion using the calibrated success definition;
- zapdash as a subtype/sequence whose terminal successful dash is the payout event;
- Rival double-dash as a sequence label over two separately successful dash completions, with **no third bonus payout**;
- ball flip-reset acquisition;
- distinct chain reset reacquisition;
- pre-flip reset as a subtype of the reset acquisition, not an extra payout;
- car reset acquisition.

Do **not** opportunistically add every catalog entry during this task.

Specifically leave out of V3 mechanics reward for now:

- possession milestones;
- ground carry/dribble;
- bounce dribble;
- generic front/side flick rewards;
- air-dribble milestones;
- pop reset reward beyond any separately proven reset acquisition;
- double tap / rebound reward;
- bare stall;
- generic recovery;
- any `OBSERVE FIRST` or telemetry/tactical-only catalog item.

The controlled-flick logic built in this task is **exemption-only**. It does not create a flick reward.

## Non-negotiable historical compatibility

Gameplay V1 and Gameplay V2 must remain behaviorally and hash compatible.

Known hashes to preserve:

- `RIVAL2_REWARD_GAMEPLAY_V1`: `48AAC000B97D2652507F677184A3FE4F0A3A86CED136B680C933EFF33CD9F072`
- `RIVAL2_REWARD_GAMEPLAY_V2`: `4073E29C1013458D5784435061FE47C639525BE37E8CD519783889C69BA87D41`
- `RIVAL2_OBS_V1`: `10669E7D240D553BEA601F8AD7AEF9F9029310E55EA6DC4072E866F34BB218AF`
- `RIVAL2_ACTION_V1`: `145AF5B49E1A0C85209022A6DE862F05EF996EB89B90B124072A59BC9936727B`
- `RIVAL2_EPISODE_V1`: `E22B6014C6D975D700D1129B9F554D6F35E4CA5003F1C1BD09C7D394D4F9347E`

Do not rewrite V1/V2 arithmetic to share a new generalized branch if doing so can alter operation order or float behavior. A small amount of duplicated historical code is preferable to silently changing frozen contracts.

## Required package files

Read every file before implementation:

- `README.md` — scope and authority;
- `IMPLEMENTATION_SPEC.md` — exact V3 architecture and reward semantics;
- `TRAINING_SAFETY_GATES.md` — checkpoint, GPU, kernel, reset, and crash-prevention gates;
- `ACCEPTANCE_CRITERIA.md` — tests/evidence required before READY;
- `CODEX_START_PROMPT.md` — execution entrypoint.

## Final allowed verdicts

The implementation task must end with exactly one of:

- `GAMEPLAY_V3_READY_FOR_REVIEW`
- `BLOCKED: <specific reason>`

`READY_FOR_REVIEW` is not training authorization. No PPO update may run under Gameplay V3 until the returned evidence has been reviewed explicitly.
