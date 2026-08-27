# Rival 2.0 Opponent Curriculum V1

This handoff authorizes one bounded Rival 2.0 continuation from the healthy
Gameplay V1 +239 checkpoint. The purpose is to expose Rival to competent fixed
opponents with different play styles while retaining self-play, and to reinforce
one mechanic Rival independently discovered: the strict successful double-dash
sequence already classified by the existing dash telemetry.

This package does **not** authorize five-minute training matches, Nexto/Wisp
learning, simulator changes, imitation learning, v0.6 work, or continuation past
the +120 boundary.

## 1. Authoritative parent

Required repository parent before implementation:

`1a4437fe92fa7ab66efd0e4100d74bb90302ea46`

Required Rival checkpoint:

`checkpoints/rival2/gameplay_v1/rival2_gameplay_resume.pt`

Required checkpoint SHA-256:

`77BF257131FB71DDEAEAE49D668C5E25AB1D06EE26149AB0D0AE303573CA5F21`

Expected source state:

- iteration / policy version: `359 / 359`;
- cumulative Rival agent decisions: `3,011,510,272`;
- reward: `RIVAL2_REWARD_GAMEPLAY_V1`;
- reward SHA-256: `48AAC000B97D2652507F677184A3FE4F0A3A86CED136B680C933EFF33CD9F072`;
- episode: `RIVAL2_EPISODE_V1`;
- short-episode SHA-256: `E22B6014C6D975D700D1129B9F554D6F35E4CA5003F1C1BD09C7D394D4F9347E`.

The prior Nexto checkpoint-selection evaluation selected +239 as the continuation
parent. Do not resume +180 and do not use any collapsed Scoring V1 checkpoint.

## 2. Objective

Rival already solves acquisition: against frozen Nexto, deterministic +239 took
the first legitimate touch in every episode but still lost every episode. The
next curriculum should therefore expose Rival to better post-contact decisions,
defense, possession pressure, and high-ball/aerial states rather than add more
acquisition shaping.

Use four opponent families:

- **35% frozen Nexto**: fixed competent 1v1 pressure;
- **35% frozen Wisp v2-75B**: additional possession/high-ball/aerial exposure;
- **20% current-policy Rival self-play**;
- **10% frozen historical Rival pool**.

Percentages are per newly reset training episode. The selected opponent family
and Rival side remain fixed for that episode and are sampled again only on reset.

For external/historical opponent episodes, balance current Rival between Blue and
Orange prospectively and independently of outcomes. For current-current
self-play, both sides use the current policy as before.

Use one dedicated seeded opponent-curriculum RNG, persist its state in
checkpoints, and make the mapping reproducible. A seed of `2026082703` is the
authorized default unless the implementation already has a better explicit
campaign seed convention; if a different deterministic derivation is used,
record it exactly.

## 3. Training-mask semantics

Only actions produced by the **current Rival policy** are trainable PPO samples.

- Nexto actions: frozen, never trainable.
- Wisp actions: frozen, never trainable.
- historical Rival actions: frozen, never trainable.
- current Rival vs current Rival: both current-policy sides remain trainable.

Do not count frozen-opponent decisions as Rival training samples. Because the
mix changes the number of trainable agents per world, do not assume the old fixed
`8,388,608` trainable-sample increment per PPO update. Record the exact current-
policy trainable sample count each update and the exact cumulative count.

Preserve the existing historical pool from the source checkpoint. At the
scheduled +30/+60/+90/+120 boundaries, the normal snapshot mechanism may add the
new Rival snapshot to the bounded pool after the checkpoint is made; record any
evictions exactly.

## 4. Lifecycle

Keep the exact original short lifecycle:

`RIVAL2_EPISODE_V1`

- standard kickoff initialization;
- first goal terminates the episode;
- 15 seconds without any touch truncates/reset;
- 45-second hard episode limit;
- Rival policy at 30 Hz;
- physics at 120 Hz.

Do not use five-minute matches.

Create fresh simulator/episode state for this curriculum transition. Preserve
model, optimizer, existing policy/PPO/self-play configuration, CPU/CUDA RNG,
policy RNG, historical-pool state, counters, and all other compatible resumable
training state. Record the exact curriculum transition. The new opponent-family
assignment state and its dedicated RNG are new curriculum state and must become
checkpointed state.

## 5. Frozen Nexto

Reuse the already verified RivalSim Nexto integration.

Pinned Nexto upstream commit:

`2e6ed7d6ed2b352e8ff529d4a12a0c9c70c28cca`

Pinned Nexto model SHA-256:

`BF5343B5EEACAC6BF7CDB75DAC4A5C14BA0F94D820EAE75F00A211B6119D69FA`

Preserve the exact behavior validated by the existing Gameplay V1 vs Nexto
suite, including its intended 15 Hz neural inference and 120 Hz stock kickoff
controller behavior. Do not modify or train Nexto.

## 6. Frozen Wisp v2-75B

Wisp is not currently integrated in RivalSim, so a small source-faithful adapter
port is authorized before training. The exact source pin and fidelity gate are in
`WISP_SOURCE.md`.

The training hot path must be batched/GPU-resident enough to support its assigned
worlds. Do not run one Python/CPU Wisp instance per world. Moving the frozen
Torch model to CUDA for batched inference is allowed only if the deterministic
behavior is shown equivalent by the targeted fidelity gate.

Do not create a second physics simulation for Wisp. Build Wisp's observation,
action mask, action parser, action delay/cadence, and any required prediction
inputs from authoritative RivalSim state while preserving pinned upstream
semantics.

Training may not start until the targeted Wisp fidelity gate passes and exact
vendored model SHA-256 identities are recorded.

## 7. Reward transition

Do not edit `RIVAL2_REWARD_GAMEPLAY_V1`.

Create a new immutable reward identity:

`RIVAL2_REWARD_GAMEPLAY_V2`

Gameplay V2 is **exactly Gameplay V1 plus one additional competitive event**:

`successful_strict_double_dash = +0.005`

For Blue:

```text
BlueReward = GameplayV1Blue
           + 0.005 * BlueSuccessfulStrictDoubleDashEvents
           - 0.005 * OrangeSuccessfulStrictDoubleDashEvents

OrangeReward = -BlueReward
```

All existing Gameplay V1 components and operation semantics remain unchanged:

- historical V1 goal/progress/touch/demo reward;
- speed shaping;
- supersonic shaping;
- physical boost-use shaping;
- positive-resource boost-pad shaping;
- legitimate-save shaping.

There is still no approach reward, first-touch bonus, no-touch reward/penalty,
proximity reward, generic jump/flip reward, wavedash reward, wall-dash reward,
zapdash reward, aerial reward, or named-mechanic reward other than the one strict
successful double-dash event above.

## 8. Strict double-dash event

The reward must use the already researched and published strict state-transition
semantics from:

`results/rival2/gameplay_v1_nexto/dash_mechanics_contract.json`

Expected contract content SHA-256:

`F8FCF018A1D013D84DEF0F16D5033CE86F56B3F82CB95C1022F32722D1FB7510`

The offline classifier currently lives in `rivalsim/nexto_short_eval.py` and its
raw event traces include the Rival-discovered sequence from the +180 stochastic
Nexto evaluation.

Do **not** call a large host-side trace classifier from the training hot path.
Implement a compact GPU-compatible online state machine whose successful event
is semantically equivalent to the strict double-dash classifier: two measured
successful wavedash outcomes in rapid succession with intervening wheel contact,
including the existing actual dodge/contact requirements.

Pay `+0.005` only when the second successful wavedash completes the strict
sequence. Pay once per completed sequence. Partial attempts, individual dodges,
individual wavedashes, jump inputs, and flip inputs receive nothing.

Before launch, parity-test the online training event against the existing offline
classifier using the retained real Rival double-dash trace plus focused negative
controls. Do not broaden this into a mechanics benchmark.

Continue to record all other dash-mechanic candidates as observational telemetry
in held-out evaluations, with no reward attached.

## 9. PPO safety

Keep the healthy Gameplay V1 PPO settings and the existing transactional KL
rollback exactly. Do not retune PPO for this phase.

Mandatory rejection/stop conditions remain:

- post-step minibatch KL `> 0.10`, or
- completed-update mean KL `> 0.05`.

A rejected update must restore model, optimizer, and relevant RNG state exactly,
publish the failure evidence, and stop. Do not automatically lower the learning
rate or continue.

Keep the actor/value/action-saturation diagnostics added for Gameplay V1.

## 10. Wisp/Nexto opponent telemetry

Training and held-out evidence must separate opponent families. At minimum,
record for Nexto and Wisp independently:

- Rival wins / opponent wins / no-goal episodes;
- goals for and against;
- Rival/opponent touch counts and first-touch share;
- saves and conceded goals;
- no-touch/hard-timeout fractions;
- mean episode duration;
- Rival mean speed, supersonic, boost use/pickups, grounded/airborne, jump and
  flip telemetry;
- Rival and opponent airborne-touch fraction;
- ball-center height distribution at each side's touch onsets (at least count,
  mean, median, p90, p99, maximum);
- strict double-dash count and its reward contribution;
- observational wavedash/zapdash/double-dash/wall/curve candidate counts using
  the existing researched telemetry definitions.

The purpose of Wisp exposure is to create genuine high-ball/aerial pressure, not
to add an aerial reward. Report whether Wisp actually produces a materially
higher airborne/high-ball touch distribution than the other opponent families in
this corpus.

## 11. Bounded run

Train exactly **120 additional PPO updates** from iteration 359 unless the KL
safety guard stops earlier.

Expected checkpoint iterations if every update succeeds:

- +30 -> iteration 389;
- +60 -> iteration 419;
- +90 -> iteration 449;
- +120 -> iteration 479.

Save a full resumable checkpoint at each boundary and stop after the +120
checkpoint/evaluation. Do not start an overnight continuation.

Report exact current-policy trainable agent samples rather than assuming a fixed
sample increment.

## 12. Compact held-out evaluation

At source and each scheduled boundary, evaluate Rival separately against frozen
Nexto and frozen Wisp using paired seeds/layouts across checkpoints.

For each opponent:

1. **Canonical deterministic check:** all five standard kickoff layouts with
   Rival on each side (`10` episodes total per opponent). This is an exact
   behavior check, not a statistical sample.
2. **Stochastic Rival robustness:** `128` Rival-Blue + `128` Rival-Orange short
   episodes against deterministic frozen opponent (`256` total per opponent).

Do not multiply deterministic copies of identical kickoff states and present
them as independent samples. Do not expand the evaluation without a concrete
failure requiring it.

The existing +239 Nexto result may be reused as source baseline where protocol
and identity match; do not rerun expensive evidence merely for ceremony. A new
source Wisp baseline is required after its adapter fidelity gate passes.

## 13. What success means

There is no single pass/fail gameplay threshold for this exploratory phase.
Report whether the trajectory shows:

- improving stochastic and/or deterministic performance against Nexto;
- improving performance against Wisp;
- preserved acquisition/no-touch behavior;
- increased ability to survive/respond after losing first contact;
- exposure to and response to high-ball/aerial states;
- stable PPO/KL/action distributions;
- emergence/frequency of the discovered strict double dash without reward
  farming;
- no collapse of self-play-capable general behavior.

The double-dash reward is deliberately tiny. If it becomes a meaningful fraction
of total gameplay reward, report that as a curriculum problem rather than
increasing it.

## 14. Hard boundaries

Do not:

- resume from anything except Gameplay V1 +239;
- train Nexto or Wisp;
- use imitation/supervised learning from either opponent;
- alter Rival observation or action contracts;
- alter network architecture;
- alter simulator physics;
- alter PPO hyperparameters;
- use five-minute training matches;
- add approach shaping;
- reward aerials or any other named mechanic;
- reward wavedash/zapdash/wall-dash independently;
- run RocketSim acceptance/parity suites;
- rerun the old 16,384-duel benchmark;
- begin v0.6;
- continue past +120.

Targeted adapter/reward/state-machine tests and the compact evaluations above are
authorized.

## 15. Required return package

At stop, commit and push a compact evidence package containing:

1. exact source checkpoint/hash and transition record;
2. `RIVAL2_REWARD_GAMEPLAY_V2` equation/hash;
3. Wisp source/model provenance and fidelity result;
4. exact opponent-assignment implementation and realized mix;
5. per-update current-policy trainable sample counts;
6. PPO/KL/value/action-saturation curve;
7. checkpoints/hashes at +30/+60/+90/+120;
8. Nexto evaluation curve by side;
9. Wisp evaluation curve by side;
10. high-ball/aerial touch telemetry by opponent;
11. dash-mechanic telemetry and strict double-dash reward totals;
12. acquisition-regression indicators;
13. final +120 checkpoint path/SHA-256;
14. recommendation for the next curriculum step.

Stop there and return the evidence before any further training.
