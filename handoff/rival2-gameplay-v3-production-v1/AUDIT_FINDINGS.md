# Pre-Implementation Audit Findings

This file records the code facts that were checked before the handoff was authored. Codex should re-verify them against its starting HEAD and report any drift.

## 1. Current reward structure

At audited baseline `0228a833e90d2db2715f8b79b65f6cbdc59fefbc`:

- `rivalsim/rival2_contracts.py` defines Gameplay V1/V2; there is no V3.
- Gameplay V1 is based on historical V1 and therefore inherits unconditional unique-touch reward `0.05`.
- Gameplay V2 is Gameplay V1 plus the standalone strict-double-dash component `0.005`.
- Published Gameplay V2 canonical hash is `4073E29C1013458D5784435061FE47C639525BE37E8CD519783889C69BA87D41`.

## 2. Current reward kernel facts

`rivalsim/kernels/rival2.py` currently:

- defines reward modes `BASE=0`, `ACQUISITION=1`, `GOAL_ONLY=2`, `GAMEPLAY=3`, `GAMEPLAY_V2=4`;
- clears per-decision touch/demo/reward/gameplay counters in `rival2_begin_decision()`;
- detects unique touch onset in `rival2_accumulate_tick()` from `hit_this_tick` plus `touch_contact_latched`;
- computes reward on the fourth physics tick;
- Gameplay V1/V2 arithmetic includes `0.05 * (BlueTouchCount - OrangeTouchCount)`;
- Gameplay V2 conditionally adds the standalone strict-double-dash counter;
- historical Orange reward is final negation of Blue in those competitive paths.

Implication: V3 must not mutate the historical touch latch before `rival2_accumulate_tick()`, and must have an explicit V3 arithmetic path with touch component zero.

## 3. Current tick ordering

`Rival2WorldSim._launch_tick()` currently:

1. calls `super()._launch_tick()`;
2. in Gameplay V2, runs `rival2_track_strict_double_dash`;
3. runs `rival2_accumulate_tick`.

The mechanics calibration spec requires the production detector at exactly the post-physics/pre-30-Hz-accumulation boundary.

Implication: native V3 detector launch belongs between steps 1 and 3. The read-only observer monkeypatch must not become the training architecture.

## 4. Current V2 dash tracker is not V3-successful-dash authority

`rival2_track_strict_double_dash()` checks:

- flip onset from airborne/zero-wheel state;
- <=42 tick air window;
- <=24 tick flip-to-landing window;
- <=90 tick pair separation;
- intervening landing/contact state.

It does **not** measure pre/post surface-tangent speed gain.

The calibrated V3 successful wavedash contract requires `delta_tangent_speed > 1.0 uu/s`.

Implication: preserve the V2 tracker unchanged for V2. Build a separate V3 successful-dash detector.

## 5. Current mechanics detector architecture

`rivalsim/mechanics_calibration.py` is explicitly read-only/reward-free and attaches by wrapping `world._launch_tick`.

Its final targeted detector semantics include:

- corrected signed Musty sweep;
- continuous-control Breezi setup + corrected terminal Musty;
- transverse-context Redirect;
- speedflip;
- half-flip;
- pinch;
- pogo;
- possession/ground-carry remain NOT_READY.

The observer also allocates per-car diagnostic evidence buffers and a number of cumulative counters.

At 131,072 worlds, an audited logical-size estimate for the current observer layout is approximately:

- ~180 MiB base observer state;
- ~112 MiB evidence arrays at 16 evidence records/car;
- ~292 MiB total.

This is not a measured CUDA allocator value; it is a logical layout estimate using 4-byte scalar/int, 12-byte vec3, and 16-byte quaternion storage. Codex must report the actual production logical and CUDA allocator impact.

Implication: production V3 state must be compact and evidence storage must remain evaluation-only.

## 6. Current environment / bridge behavior

`Rival2Env._step_impl()` currently:

1. saves decision observation;
2. calls `world.begin_decision()`;
3. emits action;
4. runs exactly four physics ticks;
5. clones transition observation and reward;
6. clones done/reset masks;
7. applies selective physical/episode resets;
8. builds post-reset observation.

`Rival2TensorBridge` binds persistent zero-copy Warp/Torch views at environment construction.

Implications:

- optional V3 arrays must exist before bridge construction if they need bridge views;
- reward must be complete before `transition_observation/reward` are cloned;
- V3 reset state must clear during `apply_interval_resets()`;
- observation schema must remain 182 dims.

## 7. Current checkpoint contract behavior

`Rival2Trainer.load_checkpoint()` rejects mismatched contract hashes/reward versions.

`load_checkpoint_curriculum_transition()` exists specifically for an authorized reward/lifecycle transition into a fresh environment. It restores learned/optimizer/RNG/counter/opponent-pool state and records changed semantics.

`transition_reward_curriculum()` also exists for an in-place reward-only transition that preserves live world state.

Implication: V3 production campaign should use the fresh-environment checkpoint-transition model, not the live in-place switch.

## 8. Mixed-opponent checkpoint state is richer than base trainer state

`Rival2OpponentCurriculumTrainer.checkpoint_payload()` additionally stores:

- opponent curriculum config/RNG/family/side assignment;
- realized family counts;
- adaptive mixed-PPO configuration;
- optimizer migration proof;
- retention observations + summary;
- split optimizer semantics;
- Nexto temporal state;
- Wisp temporal state.

The subclass restore path reconstructs split optimizer state and restores all of those fields.

Implication: V3 transition must not load only model/optimizer and accidentally discard mixed-curriculum safety state.

Because a reward-transition restart creates a fresh simulator world, opponent adapter temporal history from the old world must not be paired blindly with that fresh kickoff. Keep family/side assignments and curriculum RNG, but reinitialize adapter temporal state to the fresh episode using existing adapter reset/activate semantics. Record this as deliberate reinitialized state.

## 9. Published mixed-opponent scale/source

Published opponent-curriculum configuration:

- worlds: `131072`;
- short episodes, not five-minute training matches;
- PPO rollout horizon: `32`;
- entropy coefficient: `0`;
- hard KL guard unchanged;
- split mixed-PPO policy/critic optimizer safety active.

Published checkpoint boundaries all PASS_GREEN. Latest recorded:

- `plus_120`;
- iteration/policy `479`;
- decision samples `3655854038`;
- SHA `3B994E118A9498713DC6115D38F061958A900EA8F4D00CE568F916942E851D9A`.

Implication: production-size crash smoke must use 131072 worlds, and source selection must verify plus_120 or a newer explicitly accepted checkpoint.

## 10. Scope correction: explicit V3 mechanics set

The catalog contains more LOW REWARD CANDIDATE ideas than are currently ready to deploy safely. This handoff intentionally does not translate every catalog line into code.

Gameplay V3 initial payout set is limited to:

- calibrated continuous: speedflip, half-flip, Musty, Breezi, Redirect, Pinch, Pogo;
- source-exact/topology: successful dash, Zapdash subtype, double-dash label, ball reset, chain reset, preflip subtype, car reset.

This avoids dragging unresolved possession/ground-control/aerial-control/generic-flick dependencies into an already large reward transition.

## 11. Scope correction: controlled flick

The user requirement is to avoid punishing a legitimate controlled flick. Possession itself is not reward-ready.

Therefore the V3 controlled-flick logic is a **high-confidence exemption classifier only**. It must be calibrated for false-penalty avoidance and contributes no positive reward.

## 12. Scope correction: recognized mechanic exemption

Do not treat any mechanics event near a flip touch as proof the flip touch was legitimate.

Use same-contact association and an explicit allowlist. Musty/Breezi terminal scoop are clear examples. A Redirect nearby is not automatically an exemption; if the dodge materially powers the redirect, the power-contact exemption can cover it.

## 13. Main crash risks identified

Codex must explicitly close these risks:

1. Warp kernel argument/signature mismatch.
2. V3 detector running on pre-physics or stale contact state.
3. `begin_decision()` clearing a pending event before resolution.
4. episode reset leaking history/budget/pending contacts.
5. duplicate V2 double-dash + V3 mechanics reward.
6. V3 contact detector corrupting historical touch latch.
7. production inclusion of calibration evidence buffers causing unnecessary VRAM pressure.
8. V3 bridge binding arrays that do not exist for V1/V2.
9. strict checkpoint load failing because reward hash changed.
10. mixed-curriculum optimizer/retention/adapter state loss during transition.
11. stale external-opponent temporal state restored against a fresh world.
12. adding too many catalog mechanics at once.
13. exemption classifier so broad that it neutralizes the anti-flip correction.
14. penalty classifier so broad that legitimate challenges/power/flicks are punished.
15. starting PPO before shadow reward reconstruction is reviewed.

Every one of these has a corresponding gate in `TRAINING_SAFETY_GATES.md`.
