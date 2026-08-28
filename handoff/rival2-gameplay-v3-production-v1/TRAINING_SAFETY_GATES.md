# Gameplay V3 Training Safety Gates

This file is the crash/corruption prevention authority for the implementation task.

A green unit test suite is not enough. Gameplay V3 may be marked READY_FOR_REVIEW only after every applicable gate below passes.

## Gate 0 — source/worktree integrity

Before editing:

1. fetch `origin/main`;
2. verify `0228a833e90d2db2715f8b79b65f6cbdc59fefbc` is an ancestor of current HEAD;
3. confirm the worktree is clean or explicitly preserve unrelated local changes;
4. do not reset/rebase/drop concurrent work;
5. record starting HEAD.

Before push:

1. fetch `origin/main` again;
2. if remote moved, reconcile normally and re-run affected validation;
3. never force-push `main`.

## Gate 1 — historical contract immutability

Before any V3 test proceeds, assert the historical identities are unchanged:

- Gameplay V1 hash: `48AAC000B97D2652507F677184A3FE4F0A3A86CED136B680C933EFF33CD9F072`
- Gameplay V2 hash: `4073E29C1013458D5784435061FE47C639525BE37E8CD519783889C69BA87D41`
- Observation V1 hash: `10669E7D240D553BEA601F8AD7AEF9F9029310E55EA6DC4072E866F34BB218AF`
- Action V1 hash: `145AF5B49E1A0C85209022A6DE862F05EF996EB89B90B124072A59BC9936727B`
- Episode V1 hash: `E22B6014C6D975D700D1129B9F554D6F35E4CA5003F1C1BD09C7D394D4F9347E`

Run existing Gameplay V2 dash tests unchanged.

Do not refactor historical arithmetic merely for code reuse if float/order equivalence is not proven exactly.

## Gate 2 — Warp kernel ABI completeness

The current reward path uses long positional Warp kernel signatures. Adding arrays to one side without updating every launch site can fail at compile/runtime.

For every changed/new kernel:

- enumerate its formal input/output arrays;
- enumerate every `wp.launch` callsite;
- assert count/order/dtype/shape compatibility;
- compile and launch on CPU where supported and CUDA where required;
- do not leave optional Python-level arguments that alter Warp signature based on reward mode.

Prefer new V3-specific kernels/state objects over continuously extending the historical `rival2_accumulate_tick` signature with dozens of detector arrays.

If V3 reward composition is separated into a dedicated final-interval kernel, preserve the historical reward kernel path byte/behaviorally and make execution order explicit.

## Gate 3 — post-physics ordering

Prove with a focused test/instrumented trace that V3 detection observes the same tick's authoritative:

- `hit_this_tick` contact flags;
- car/ball pre-contact data;
- contact points/normals/impulses;
- wheel contacts;
- flip/resource state;
- chassis contact state;
- post-physics car/ball velocities.

V3 detection must run after `super()._launch_tick()` and before final 30 Hz V3 composition.

Do not attach the production system by replacing `world._launch_tick` at runtime.

## Gate 4 — reset and interval lifetime

Explicitly test:

- interval event counters clear at `begin_decision()`;
- persistent detector state does not clear at `begin_decision()`;
- pending flip-contact classification can cross one decision boundary without loss/duplication;
- true episode reset clears all V3 persistent state;
- mechanics paid-event budget returns to 0 only on true episode reset;
- no Musty/Breezi sweep history bridges reset;
- no pending 50/power/flick candidate bridges reset;
- no same-contact exemption/event is emitted after reset;
- first post-reset tick starts from initialized previous-state values rather than zero/uninitialized garbage.

## Gate 5 — mechanics budget exactness

Use integer paid-event accounting.

Test exactly:

- events 1..10 each pay `0.005`;
- event 11+ pays `0` and increments budget suppression;
- total raw mechanics paid in one player episode is exactly represented as 10 paid events / `0.05` contract value;
- opposing player has independent budget;
- budget cannot be reset by a 30 Hz interval boundary;
- budget cannot underflow/overflow;
- zero-sum competitive composition is exact.

## Gate 6 — V2 double-dash isolation

Gameplay V2:

- old strict-double-dash tracker still runs;
- old V2 `+0.005` pair reward still behaves exactly as historical tests require.

Gameplay V3:

- old V2 standalone pair term is never added;
- successful V3 dash #1 may pay once;
- successful V3 dash #2 may pay once;
- `double_dash` label adds no third payout;
- V3 dash success requires the calibrated useful tangent-speed result, not just timing/contact.

## Gate 7 — bad flip classifier fail-closed behavior

Before a penalty can be active, all exemption classifiers must pass their prospective held-out tests.

Required exactly-once candidate tests:

- flip-active loose-ball contact -> one penalty;
- persistent manifold -> still one candidate/outcome;
- new recontact after real separation may create a new candidate;
- drive-through no flip -> no candidate;
- jump-only/aerial no directional dodge -> no candidate;
- `has_flipped` true but `is_flipping` false -> no candidate;
- active flip with no ball contact -> no candidate.

Required anti-overexemption tests:

- distant opponent does not exempt;
- nearby but non-converging opponent does not exempt;
- translation-dominated hit while flipping does not power-exempt;
- normal ball speed increase alone does not power-exempt;
- loose ball merely near roof/nose does not controlled-flick-exempt;
- unrelated mechanic event cannot recognized-mechanic-exempt.

Required anti-false-penalty tests:

- simultaneous/adjacent genuine 50 -> exempt;
- calibrated converging challenge -> exempt;
- dodge-powered shot -> exempt;
- dodge-powered defensive clear -> exempt;
- controlled flick -> exempt;
- Musty/Breezi same-contact terminal event -> exempt;
- recognized preflip/reset continuation, if implemented as same-contact mechanic -> exempt.

If any required legitimate flick/challenge/power case cannot be cleanly protected, V3 is `BLOCKED`.

## Gate 8 — mechanics detector parity

Production detector output must match the final read-only calibration observer on the relevant deterministic traces for:

- speedflip;
- half-flip;
- Musty;
- Breezi;
- redirect;
- pinch;
- pogo.

Do not compare only aggregate counts. Compare event identity/completion tick/subtype on a bounded deterministic trace set.

For new V3 source-exact dash/reset detectors, compare against the documented/source-exact invariants and existing researched examples.

## Gate 9 — bridge/view integrity

If new V3 telemetry arrays are exposed through `Rival2TensorBridge`:

- bind only arrays that exist for a V3 world;
- do not make V1/V2 bridge construction depend on V3 state;
- every Warp/Torch view must alias the same device pointer;
- shapes must be deterministic for the selected world count;
- observations remain exactly `(world, 2, 182)`;
- no V3 telemetry is appended to observation.

Run the existing alias report or equivalent pointer-alias assertions.

## Gate 10 — no hot-path host transfer

At ordinary V3 stepping and rollout collection:

- no `.numpy()`;
- no `.cpu()`;
- no per-world Python classifier;
- no full-state D2H/H2D transfer;
- no JSON/event-object construction.

Use `env.reset_transfer_counters()` before a representative step/rollout and assert hot-path transfer counters stay at the expected zero/unchanged values.

Host exports are allowed only in explicit calibration/evaluation evidence paths after synchronization.

## Gate 11 — GPU memory / production-scale allocation

Published mixed-opponent training uses `131,072` worlds. This exact scale is the memory authority.

Before READY_FOR_REVIEW:

1. report the added logical V3 state bytes at 131,072 worlds;
2. report CUDA allocated/reserved memory before and after V3 environment/trainer construction;
3. confirm calibration-only event evidence buffers are absent from production state;
4. construct the V3 environment at 131,072 worlds on the intended training GPU;
5. execute at least one complete 30 Hz decision (4 physics ticks) with V3 detection/reward composition active and policy frozen;
6. destroy it cleanly and verify no persistent allocator failure.

Then run one **training-shape rollout collection smoke** using the real 131,072-world mixed-opponent trainer configuration and rollout horizon 32, but do **not** call any optimizer/PPO update.

Purpose: catch OOMs, Warp launch/ABI failures, zero-copy view failures, and state-lifetime errors under the exact rollout shape before a real campaign.

This is a single smoke, not a benchmark campaign. If memory pressure/OOM/allocator corruption occurs, return `BLOCKED` and reduce production state safely; do not reduce world count silently and call it green.

## Gate 12 — checkpoint source identity

Known published latest accepted checkpoint candidate:

```text
label: plus_120
reward: RIVAL2_REWARD_GAMEPLAY_V2
iteration: 479
policy_version: 479
agent_decision_samples: 3655854038
sha256: 3B994E118A9498713DC6115D38F061958A900EA8F4D00CE568F916942E851D9A
recorded path:
G:/dev/RivalSim-runs/opponent-curriculum-v1-safe-20260827-b2af03d/checkpoints/rival2_opponent_curriculum_plus_120_resume.pt
```

Before any source-checkpoint shadow validation:

- inspect repository/local run evidence for a newer accepted boundary;
- if no newer accepted boundary exists, verify the plus_120 file exists and its SHA-256 matches exactly;
- if a newer accepted boundary exists, verify its audit and exact identity before choosing it;
- never silently fall back to Gameplay V1 +239;
- never overwrite the selected source checkpoint.

## Gate 13 — reward transition checkpoint semantics

A V2 checkpoint must not load into V3 via ordinary strict `load_checkpoint()`.

Implement/test the explicit fresh-environment reward transition.

The transition must prove preservation of learned/training state:

- model tensors exact;
- optimizer tensors/groups exact;
- policy/iteration/sample counters exact;
- CPU/CUDA RNG exact where intended;
- policy/opponent/curriculum generator state exact;
- historical pool exact;
- opponent family/side assignments exact;
- adaptive mixed PPO config/optimizer state exact;
- retention corpus exact.

Because simulator world state is fresh, reset external opponent temporal adapters to a fresh episode state consistent with the restored family/side assignments. Do not restore stale adapter action-delay/observation history against a new kickoff world.

Record this reinitialization explicitly in the transition record.

## Gate 14 — active-process safety

Codex may inspect local processes/workdirs, but must not abruptly kill an unknown live training process.

If an active Rival training process is found:

- identify its script, workdir, reward identity, and latest durable accepted checkpoint;
- do not edit its run workdir;
- do not overwrite its checkpoints;
- stop/exit it only through a documented safe boundary mechanism if one exists and can be proven;
- otherwise return the process as an implementation blocker for campaign transition and continue code work only if doing so cannot disturb that process.

No new V3 PPO process starts in this task.

## Gate 15 — no-learning V3 shadow reconstruction

After implementation/calibration and before READY_FOR_REVIEW, run approximately 256 short episodes from the exact selected source policy with policy/opponents frozen.

Compute the full V3 reward as shadow reconstruction. Do not call PPO update.

Report:

- touches/min;
- flip-active touches/min;
- unnecessary flip contacts/min;
- flip-touch/all-touch fraction;
- unnecessary/flip-touch fraction;
- contest exemption count/rate;
- power exemption count/rate;
- controlled flick exemption count/rate;
- recognized mechanic exemption count/rate;
- mechanic event rate by canonical event/subtype;
- theoretical paid mechanic rate after 10-event budget;
- budget-hit episode fraction;
- mechanics mean absolute reward/decision;
- bad-flip mean absolute penalty/decision;
- progress mean absolute reward/decision;
- mechanics/progress ratio;
- bad-flip/progress ratio;
- impossible/jitter classifications;
- duplicate suppressions/rearms;
- bounded representative evidence traces for every classifier outcome.

Do not declare PASS merely because counts are low. Inspect sampled event evidence for physical correctness.

## Gate 16 — reward reconstruction

For every shadow decision verify:

```text
BlueTotal == Goal + Progress + 0Touch + Demo
             + Speed + Supersonic + BoostUse + BoostPickup + Save
             + Mechanics + BadFlip
OrangeTotal == -BlueTotal
```

Use a strict tolerance appropriate to the exact device float arithmetic and separately prove the component arrays reconstruct the actual emitted reward.

No hidden reward term is allowed.

## Gate 17 — final no-training assertion

Before commit/push evidence, assert and report:

- policy/model tensors unchanged from the selected source during V3 validation;
- optimizer state unchanged;
- policy version unchanged;
- iteration unchanged;
- agent decision sample counter not represented as training progress by any validation harness;
- no PPO update function was called;
- no historical checkpoint was modified.

Only after all gates pass may the implementation verdict be `GAMEPLAY_V3_READY_FOR_REVIEW`.
