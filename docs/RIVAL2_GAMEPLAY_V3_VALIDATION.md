# Rival2 Gameplay V3 production validation

Status: `SUPERSEDED_BY_VALIDATION_CORRECTION_V1`

> The classifier-calibration claims in this original package were found to be
> circular because they were based on hand-authored feature dictionaries. They
> are not current review evidence. The replacement physical-trace calibration,
> rerun runtime gates, and corrected verdict are indexed in
> `docs/RIVAL2_GAMEPLAY_V3_VALIDATION_CORRECTION_V1.md`. This original document
> remains only as the immutable index for the provisionally accepted production
> package at `00a4865400291a5ff0a34925a966c0963f55d963`.

This document is the compact reviewer index for the implementation and evidence authorized by `handoff/rival2-gameplay-v3-production-v1/`. It is a review gate only. Gameplay V3 PPO training was not started.

## Required reviewer package

### 1. Final commit SHA

The final evidence commit is the commit containing this document and `results/rival2/gameplay_v3_validation/artifact_manifest.json`. The immutable implementation commit audited below is `e160ee18be6db64f6c8b0fc609599cd48708f9b1`. The final pushed SHA is reported in the Codex return package after committed-blob verification.

### 2. Starting and ending HEAD

- Starting `main`/`origin/main`: `a38e1b5d88f94b733d631feb1eac955459a45d78`.
- Required commit ancestry: PASS; the starting HEAD equals and therefore includes the required commit.
- Implementation commit: `e160ee18be6db64f6c8b0fc609599cd48708f9b1`.
- Ending HEAD: the final evidence commit containing this document; reported after push.

### 3. Files changed

- Production: `rivalsim/gameplay_v3.py`, `rivalsim/kernels/rival2.py`, `rivalsim/mechanics_calibration.py`, `rivalsim/rival2_contracts.py`, `rivalsim/rival2_env.py`, and `rivalsim/rival2_opponent_curriculum.py`.
- Validation: `tests/test_rival2_gameplay_v3.py` and `benchmarks/run_rival2_gameplay_v3_validation.py`.
- Review: this document.
- Machine evidence: all JSON files under `results/rival2/gameplay_v3_validation/`, including the manifest and mandatory post-commit audit closure.

### 4. V3 contract hash

`RIVAL2_REWARD_GAMEPLAY_V3` is immutable with canonical SHA-256 `77541461F470420451B9E17421FC0C6BA3E84F8B5BB453E4DCB5AF70871A5051`. Evidence: `contract.json`.

### 5. Exact reward arithmetic

For each 30 Hz decision:

`Blue = Goal + Progress + 0Touch + Demo + Speed + Supersonic + BoostUse + BoostPickup + Save + 0.005*(BluePaidMechanics - OrangePaidMechanics) - 0.01*(BlueBadFlipContacts - OrangeBadFlipContacts)`

`Orange = -Blue`

Retained coefficients are goal `10.0`, progress `0.5 * delta_ball_y / 5120`, demo `0.1`, speed `0.0001 * clamped_speed_fraction`, supersonic `0.0002`, boost-use `0.00005`, small pad `0.001`, full pad `0.005`, and save `0.75`. Touch is exactly zero, and the Gameplay V2 strict-double-dash component is exactly zero. Reconstruction covered 4,096 decisions with `0.0` maximum Blue and Orange error at tolerance `1e-6`. Evidence: `contract.json` and `reward_reconstruction.json`.

### 6. Historical hash proof

- Gameplay V1: `48AAC000B97D2652507F677184A3FE4F0A3A86CED136B680C933EFF33CD9F072`.
- Gameplay V2: `4073E29C1013458D5784435061FE47C639525BE37E8CD519783889C69BA87D41`.
- Observation V1: `10669E7D240D553BEA601F8AD7AEF9F9029310E55EA6DC4072E866F34BB218AF`.
- Action V1: `145AF5B49E1A0C85209022A6DE862F05EF996EB89B90B124072A59BC9936727B`.
- Episode V1: `E22B6014C6D975D700D1129B9F554D6F35E4CA5003F1C1BD09C7D394D4F9347E`.

All match the authority exactly. Existing V2 tests passed unchanged. Evidence: `contract.json`.

### 7. Production mechanics list

Canonical `+0.005` completions are speedflip, half-flip, Musty, Breezi, redirect, pinch, pogo, successful dash/wavedash, ball-reset acquisition, and car-reset acquisition. Zapdash and Rival double-dash are labels/sequences over successful dash events and pay no third/subtype reward. Chain reset and preflip reset are labels over reset acquisition and pay no extra subtype reward.

The continuous detector is a direct production launch of the final calibrated GPU kernel, not `MechanicsShadowObserver.attach()`. Each of speedflip, half-flip, Musty, Breezi, redirect, pinch, and pogo matched 72 deterministic traces, including 24 held-out traces per family with held-out FP=0 and FN=0. Evidence: `detector_parity.json`.

### 8. Disabled and telemetry-only mechanics

No reward exists for possession, ground carry/dribble, controlled flick, air-dribble milestones, pop reset beyond reset acquisition, double tap/rebound, bare stall, recovery, generic jump/flip/aerial, or any observe-first catalog item. Controlled flick is exemption-only and contributes exactly zero positive reward. There is no generic jump penalty, generic flip penalty, or mechanic-failure penalty. Evidence: `contract.json` and `deterministic_cases.json`.

### 9. V3 dash and reset semantics

- Successful dash requires tangent-speed gain strictly greater than `1.0 uu/s`; air/landing windows are `42/24` physics ticks.
- Zap windows are `12/30` ticks; Rival double-dash is two distinct successful dashes within `90` ticks and adds no third payout. No fresh-jump prohibition was added.
- Ball reset uses ball-support face identity, at least three supporting wheels, and a real transition to untimed aerial resource availability. Car reset uses other-car support identity.
- Chain reset requires loss/consumption, separation, and a later distinct reacquisition. Preflip reset is a zero-extra-payout subtype. Evidence: `contract.json` and `deterministic_cases.json`.

### 10. Bad-flip candidate definition

A candidate is a new legitimate car-ball contact onset during an active directional dodge: `is_flipping != 0`, `has_flipped != 0`, and non-zero directional `flip_rel_torque`. It has a bounded two-physics-tick pending window for same-contact recognized/exemption resolution. Drive-through contacts, ordinary jump/aerial contacts, stale `has_flipped`, active flips without contact, and temporally unrelated flips do not qualify. Contact onset, pending/resolution latches, separation/recontact handling, decision-boundary persistence, and episode-reset clearing provide exactly-once behavior. Evidence: `contract.json`, `deterministic_cases.json`, and focused tests.

### 11. Contest/50 exemption

The frozen association window is 2 physics ticks with ball displacement at most `300 uu`. The opponent must be within `500 uu`; both self and opponent closing speed must be at least `150 uu/s`; time-to-ball difference must be at most `0.12 s`. The corpus covers simultaneous and narrowly adjacent 50s, convergence, distant/non-converging/moving-away opponents, and an uncontested loose ball. Prospective held-out confusion: TP=1, TN=2, FP=0, FN=0. Training confusion: TP=2, TN=4, FP=0, FN=0. Evidence: `classifier_calibration.json`.

### 12. Power-contact exemption

Contact-point velocity is `v_linear + omega x r`. Frozen thresholds are total closing speed at least `300 uu/s`, rotational closing contribution at least `100 uu/s`, rotational share at least `0.18`, and ball delta-v at least `175 uu/s`. The corpus covers offensive shot, defensive clear, weak-real dodge power, weak ordinary flip, translation-dominated speed, and already-fast/insignificant-delta cases. Prospective held-out confusion: TP=1, TN=2, FP=0, FN=0. Training confusion: TP=3, TN=3, FP=0, FN=0. Evidence: `classifier_calibration.json`.

### 13. Controlled-flick exemption

Frozen control requires at least 4 control ticks, maximum ball-car distance `220 uu`, maximum relative speed `260 uu/s`, then release distance at least `245 uu` and ball delta-v at least `120 uu/s` on the directional-dodge contact. Front, diagonal, and side releases plus loose-ball, kickoff/50, brief-control, no-release, and chase negatives are represented. Prospective held-out confusion: TP=1, TN=2, FP=0, FN=0. Training confusion: TP=3, TN=3, FP=0, FN=0. Positive reward is exactly `0.0`. Evidence: `classifier_calibration.json`.

### 14. Recognized-mechanic association

The same-contact allowlist is Musty, Breezi, and authorized preflip reset. Association is to the candidate's bounded physical-contact sequence; unrelated nearby mechanics do not suppress. Redirect, pinch, pogo, dash, speedflip, and half-flip are not blanket flags. Primary precedence is recognized mechanic, controlled flick, contested 50, power contact, then unnecessary flip-through; all applicable exemption flags remain in telemetry while only one primary outcome is counted. Evidence: `contract.json`, `deterministic_cases.json`, and `shadow_event_evidence.json`.

### 15. Deterministic and focused test results

- Ruff on all changed Python files: PASS.
- `pytest -q tests/test_rival2_gameplay_v3.py`: 13 passed.
- `pytest -q tests/test_rival2_gameplay_v2.py`: 4 passed.
- `pytest -q tests/test_rival2_mechanics_calibration.py`: 10 passed with exact local CMFs/CUDA.
- `pytest -q tests/test_rival2_gameplay_reward.py`: 4 passed with exact local CMFs/CUDA.
- `pytest -q tests/test_rival2_scoring_reward.py::test_checkpoint_transition_preserves_compatible_state_and_freshens_match`: 1 passed with exact local CMFs/CUDA.
- `git diff --check`: PASS.

No broad simulator suite was substituted for the handoff's focused scope.

### 16. Kernel ABI, reset, and lifetime tests

CUDA launch compilation and argument order/type smoke passed. The V1/V2/V3 construction/step matrix reported aliased tensor views and zero host-transfer counters. V3 detector placement is `physics -> V3 detector -> retained accumulation -> V3 composition`. Interval state clears at decision boundaries; persistent detector/budget history survives a decision boundary and clears only on an actual episode reset. Reset history is initialized and V3 bridge views alias Warp arrays. Evidence: `kernel_abi_smoke.json`, `deterministic_cases.json`, and focused tests.

### 17. V3 state memory footprint

At exactly 131,072 worlds, the production V3 inventory contains 107 actual Warp arrays totaling `316,670,188` logical bytes (`302.000225 MiB`). Calibration evidence buffers are absent at production capacity zero; only two one-element ABI sentinels totaling 8 bytes remain. V1/V2 do not include V3 bytes.

The full V3 environment's observed device-used delta was `2,740,977,664` bytes (`2,614 MiB`) from pre-allocation to constructed environment; after the first decision the observed delta was `7,866,417,152` bytes (`7,502 MiB`). These larger numbers include the complete simulator, observations/bridge, Warp allocations, module loading, and allocator reservations, not only the 302 MiB logical V3 state. Warp's process-local memory pool retained allocations after object destruction; external device usage returned to the idle baseline after process exit. Evidence: `memory_smoke.json`.

### 18. Exact-scale one-decision smoke

On NVIDIA GeForce RTX 5090, an exact 131,072-world V3 environment allocated and completed one 4-tick decision in `2.470335 s`. Reward was finite and exactly zero-sum, observations were `(131072,2,182)`, production evidence buffers were absent, and hot-path H2D/D2H counters were zero. No OOM, illegal access, Warp launch failure, or unrecoverable allocator warning occurred. Evidence: `memory_smoke.json`.

### 19. Exact-scale horizon-32 rollout-only smoke

A disposable 131,072-world mixed-opponent trainer loaded the explicit transition and completed `collect_rollout()` at horizon 32 in `5.178852 s`. Rewards were finite. No update was called; model, optimizer, iteration 479, and policy version 479 stayed exact. The disposable sample counter moved from `3,655,854,038` to `3,661,315,734`, was not saved, and is not campaign state. Peak Torch allocated/reserved memory was `12,070,424,064`/`13,958,643,712` bytes. Evidence: `memory_smoke.json`.

### 20. Source checkpoint

Selected source: `G:/dev/RivalSim-runs/opponent-curriculum-v1-safe-20260827-b2af03d/checkpoints/rival2_opponent_curriculum_plus_120_resume.pt`, Gameplay V2 iteration/policy 479, samples `3,655,854,038`, SHA-256 `3B994E118A9498713DC6115D38F061958A900EA8F4D00CE568F916942E851D9A`. No newer accepted local checkpoint was identified. Evidence: `checkpoint_transition.json`.

### 21. Checkpoint transition preservation

Ordinary strict V2-to-V3 load rejected with an incompatible-contract error; the authorized explicit transition into a fresh V3 environment passed. Model and optimizer digests are byte-exact. Iteration, policy version, sample counter, CPU/CUDA/policy/opponent/curriculum/Wisp observation generators, opponent assignment/family/side and realized counts, historical pool, curriculum configuration, mixed adaptive PPO state, and retention state are exact. New simulator, kickoff/lifecycle, V3 detector, Nexto cadence/action/kickoff, Wisp delay/ETA/slot state are fresh and consistent with restored assignments. The source file hash remained exact. Evidence: `checkpoint_transition.json`.

### 22. 256-episode no-learning shadow gate

The exact source policy completed 256 episodes over `135.504444` car-minutes with no update. Policy/model/optimizer and counters remained exact and the source checkpoint rehashed exactly. Rates per car-minute were: touches `20.073142`, flip-active touches `12.634272`, unnecessary flip-through contacts `8.376109`, and theoretical paid mechanics `6.258097`. Exemptions were contest 572, controlled flick 0, power 2, and recognized mechanic 3. Detected/paid mechanics were speedflip 12/12, half-flip 1/1, Musty 3/3, Breezi 0/0, redirect 509/509, pinch 21/21, pogo 181/181, successful dash 112/109, ball reset 10/10, and car reset 2/2. Player-episode budget-hit fraction was `0.001953125`; impossible count was 0. Bounded raw evidence exported 256 mechanic and 256 classifier records, all finite. Evidence: `shadow_gate_summary.json` and `shadow_event_evidence.json`.

### 23. Mechanics and bad-flip reward scale

Canonical mechanics pay `+0.005`, at most 10 paid events per player/episode (`0.05` integer-accounted budget); event 11+ is suppressed. Bad flip pays `-0.01` to the offender before zero-sum composition. In shadow inference, mean absolute mechanics reward/active-world-decision was `0.00003370123`, bad-flip penalty `0.00009175591`, progress `0.002420217`, and total reward `0.022543174`. Mechanics/progress ratio was `0.01392488`; bad-flip/progress ratio was `0.03791226`. Evidence: `contract.json` and `shadow_gate_summary.json`.

### 24. No PPO update or training

Confirmed. The validation runner has no trainer update, PPO update, optimizer step, or checkpoint save call. Both full-shape rollout and shadow evidence report `no_update_called=true`, unchanged model/optimizer/iteration/policy, and `ppo_update_calls=0`. No Gameplay V3 training process was started.

### 25. No historical checkpoint changed

Confirmed. The source checkpoint SHA-256 was verified before and after transition, full-shape rollout, and shadow validation and remained `3B994E118A9498713DC6115D38F061958A900EA8F4D00CE568F916942E851D9A`. No validation checkpoint was written.

### 26. Final verdict

`GAMEPLAY_V3_READY_FOR_REVIEW`

## Mandatory POST_COMMIT_AUDIT findings

1. PASS — retained Gameplay boost-use/pad/save accumulation is explicitly shared with V3; identical V1/V3 deterministic event cases pass while V3 touch is zero. Evidence: `rivalsim/kernels/rival2.py`, `tests/test_rival2_gameplay_v3.py`, and `reward_reconstruction.json`.
2. PASS — historical Gameplay V1/V2 and V3 have explicit branches; unsupported reward modes fail before stepping. Evidence: `rivalsim/kernels/rival2.py`, `rivalsim/rival2_env.py`, and `kernel_abi_smoke.json`.
3. PASS — contract/hash, environment constructor/setter, optional state/bridge, kernel mode, validator, and explicit checkpoint-transition dispatch sites are all closed. Evidence: production files, `contract.json`, and `checkpoint_transition.json`.
4. PASS — rollout and shadow trainers are disposable; no update/save occurs, source vs disposable counters are separate, and the source checkpoint is byte-identical. Evidence: `memory_smoke.json`, `checkpoint_transition.json`, and `shadow_gate_summary.json`.
5. PASS — V1/V2/V3 construction/step matrix passes; only V3 allocates/binds V3 state and all observations remain `(N,2,182)`. Evidence: `kernel_abi_smoke.json`.
6. PASS — logical V3 memory is derived from the 107-array production inventory; production has no calibration evidence buffers and measured allocator deltas are separately explained. Evidence: `memory_smoke.json`.
7. PASS — capture/invalidation code is untouched; current direct launch order is validated. Evidence: `rivalsim/rival2_env.py` and `kernel_abi_smoke.json`.

Machine-readable closure: `results/rival2/gameplay_v3_validation/post_commit_audit.json`.
