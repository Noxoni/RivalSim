# Wisp v2-75B source pin and adapter gate

This document freezes the Wisp source authority for Rival 2.0 Opponent
Curriculum V1. Wisp is a frozen opponent only; no Wisp learning or weight
modification is authorized.

## Upstream identity

Repository:

`NicEastvillage/RLBot-Wisp-v2-py`

Pinned commit:

`58d4ab18fd0c92529b5ae6582ecf1713a6b1887a`

Pinned agent identity from `src/config.py` / `src/wisp.bot.toml`:

`eastvillage/wisp/v2-75B`

Display name:

`Wisp v2-75B`

The pinned bot metadata describes it as a multi-mode ML bot focused on
possession, teamwork, and aerials. This curriculum uses it because its state
and action distribution should expose Rival to high-ball/aerial situations that
Rival's current self-play does not generate often enough.

Do not silently advance the upstream commit.

## Pinned model artifacts

At the pinned commit, the model files are:

- `src/models/POLICY.lt`
  - Git blob SHA: `5bcff2f1df61a55b0133a05f8d40e2c97e51259a`
  - size: `7,701,849` bytes
- `src/models/SHARED_HEAD.lt`
  - Git blob SHA: `0d2e9e57e5774138a6f656e79b87abd65a8e283b`
  - size: `6,052,377` bytes

Before training, retrieve the exact pinned files, compute their byte SHA-256
values, and freeze those values in RivalSim provenance plus the launch gate.
Git blob identity is not a substitute for the required byte SHA-256 evidence.

Verify and preserve the upstream license/notice material for every vendored file
and update `THIRD_PARTY_NOTICES.md` as appropriate.

## Source behavior that must be preserved

The pinned `src/config.py` specifies:

- `TICK_SKIP = 8`;
- `ACTION_DELAY = TICK_SKIP - 1` (`7` ticks);
- deterministic inference enabled;
- policy model `POLICY.lt`;
- shared-head model `SHARED_HEAD.lt` when present;
- ReLU model metadata.

The pinned `src/bot.py` uses the Wisp observation builder, action mask,
`XMirroredActionParser`, `ModelSet`, RocketSim-state adaptation, previous-action
history, score differential, and its tick/action-delay logic.

Inspect all pinned dependencies required to reproduce the policy semantics,
including at minimum:

- `src/config.py`;
- `src/bot.py`;
- `src/obs_builder.py`;
- `src/action_parser.py`;
- `src/backend/model.py`;
- `src/backend/rocketsim_adapter.py` and the state classes it depends on;
- any ball-prediction or previous-action inputs actually consumed by the
  observation path;
- the exact model artifacts.

Do not assume Wisp uses Nexto's kickoff controller or cadence. Derive Wisp's
kickoff/control behavior from the pinned Wisp source and preserve it.

## RivalSim integration requirements

Create a clearly isolated frozen-opponent integration, preferably under a path
such as `third_party/wisp75b/`, with a provenance record analogous to the
existing Nexto integration.

The RivalSim adapter must:

1. consume authoritative RivalSim world state;
2. build the exact Wisp observation and action mask semantics;
3. run the exact pinned model weights deterministically;
4. reproduce Wisp's discrete-action parsing and controller output;
5. preserve Wisp's 8-tick decision cadence and 7-tick action-delay semantics;
6. preserve any previous-action/history state required by Wisp;
7. reset Wisp adapter state correctly at RivalSim episode resets;
8. support Wisp on either Blue or Orange;
9. avoid a second physics simulation;
10. be batched/vectorized for the assigned training worlds.

A CUDA/batched execution port is allowed because the upstream CPU-per-bot path
cannot service tens of thousands of worlds. The CUDA port must be behaviorally
validated against the pinned reference before it is used for training.

Do not use a Python loop over one Wisp object per world in the training hot path.
Do not transfer full world state CPU<->GPU every physics tick.

## Targeted fidelity gate

Training may not start until a compact fixed-state fidelity check passes.

Build a small representative corpus from authoritative RivalSim states covering,
at minimum:

- all five kickoff layouts and both Wisp sides;
- ordinary grounded/open-play states;
- airborne/high-ball states;
- wall/curve states;
- nontrivial boost and previous-action states.

A corpus on the order of a few hundred states is sufficient. Do not turn this
into a broad simulator benchmark.

For every state, compare the pinned upstream/reference computation with the
batched RivalSim adapter at the meaningful boundaries available from the source:

- observation tensor;
- action mask;
- model output/action index;
- parsed eight-channel controller action;
- cadence/action-delay state transition.

Require exact discrete action/controller agreement for the fixed corpus unless
an unavoidable CPU-vs-CUDA floating-point tie at a decision boundary is actually
observed. If that occurs, document the precise state/logits, prove the numerical
cause, and use the narrowest defensible tolerance. Do not silently accept broad
mismatch.

Also verify deterministic repeatability, Blue/Orange handling, and reset/history
state.

If Wisp fidelity cannot be established, stop **before training** and return the
blocking evidence. Do not replace Wisp with an approximation and proceed.

## Training use

Once the gate passes, Wisp is a frozen deterministic opponent assigned to 35% of
new short training episodes. Its actions never enter the Rival PPO train mask.

Track Wisp separately from Nexto and Rival opponents in telemetry. In particular,
report Wisp's and Rival's airborne-touch fraction plus the ball-height
distribution at touch onset so the intended aerial/high-ball exposure is
measurable rather than assumed.
