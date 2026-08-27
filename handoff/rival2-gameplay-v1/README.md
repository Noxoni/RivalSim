# Rival 2.0 short-lifecycle Gameplay V1 curriculum

This package records the user-authorized curriculum transition issued on
2026-08-27. It supersedes the older root reciprocal-validation and failed
full-match Scoring V1 instructions only for this bounded run. The entire Scoring
V1 lineage remains evidence and is not an authorized training source.

## Immutable start

- Source commit: `61307571d86508f3026402c4948f759f310ff36c`.
- Source checkpoint:
  `checkpoints/rival2/acquisition_v1/rival2_acquisition_resume.pt`.
- Source SHA-256:
  `4FB7A3B134B25D595374E3968E2EDFA150A9CD6F8910B903BF892B59D7F8BC9A`.
- Source iteration and policy version: 120.
- Source agent decision samples: 1,006,632,960.
- Source reward: `RIVAL2_REWARD_ACQUISITION_V1`.
- Episode contract: `RIVAL2_EPISODE_V1`.

Preserve model weights, optimizer state, policy/PPO/self-play configuration, CPU
and CUDA RNG, policy and opponent generator RNG, counters, opponent assignments,
historical policy pool, and assignment semantics exactly. Initialize fresh world
and short-episode state. Never load a Scoring V1 or later collapsed checkpoint.

## Fixed lifecycle and training boundary

- 131,072 resident standard-Soccar 1v1 worlds.
- Standard kickoff initialization.
- First goal terminates the episode.
- Fifteen seconds with no touch truncates and resets.
- Forty-five-second hard episode limit.
- 30 Hz policy decisions, 120 Hz physics, four physics ticks per action.
- Horizon 32 and all checkpoint PPO/self-play settings unchanged.
- Exactly 239 additional PPO updates / 2,004,877,312 additional agent decisions.
- Resumable checkpoints and held-out evaluations at +60, +120, +180, and +239.
- Stop after +239. No five-minute matches, extra full-game set, timed or overnight
  continuation, Nexto training, or v0.6 work.

## `RIVAL2_REWARD_GAMEPLAY_V1`

Historical `RIVAL2_REWARD_V1` is immutable. Compute one Blue competitive reward
and set Orange to its exact negation:

```text
BlueReward = historical_V1_blue_reward
           + speed_reward(Blue)        - speed_reward(Orange)
           + supersonic_reward(Blue)   - supersonic_reward(Orange)
           + boost_use_reward(Blue)    - boost_use_reward(Orange)
           + boost_pickup_reward(Blue) - boost_pickup_reward(Orange)
           + 0.75 * BlueSaveEvents     - 0.75 * OrangeSaveEvents

OrangeReward = -BlueReward
```

Historical V1 is goal +/-10, `0.5 * delta_ball_y / 5120`, legitimate unique
touch onset +/-0.05, and unique demolition onset +/-0.10. The additions are:

- speed: `0.00010 * clamp(actual_linear_speed / 2300, 0, 1)` per player and
  decision;
- supersonic: +0.00020 only from authoritative simulator supersonic state;
- boost use: +0.00005 only if boost thrust was physically active during the
  decision interval;
- small pad: +0.001 per authoritative pickup with positive resource gain;
- large pad: +0.005 per authoritative pickup with positive resource gain;
- save: +0.75 once when a legitimate unique touch changes a pre-touch straight-
  line own-goal projection inside the existing Soccar goal mouth and within two
  seconds into a post-touch non-threat, provided no goal was scored on that tick.

There is no approach/proximity/no-touch reward, first-touch bonus, or direct
reward/cost for controller appearance, jump, flip, dodge, aerial, powerslide,
recovery, or any named mechanic.

## Transactional PPO displacement boundary

The checkpoint PPO identity remains unchanged: entropy 0, learning rate 0.0003,
clip 0.2, value coefficient 0.5, max gradient norm 0.5, horizon 32, and existing
epochs/minibatch size. Before each optimizer update preserve model, optimizer,
gradients, and all relevant RNG state in memory. Reject, restore, record, and stop
if either:

- a post-step minibatch approximate KL exceeds 0.10 or is non-finite; or
- the final policy's complete-rollout mean approximate KL exceeds 0.05 or is
  non-finite.

Do not retune or continue after rejection. Log approximate KL, clip fraction,
policy/value loss, raw/post-clip gradient norm, actor mean/log-std/button
statistics, analog saturation, and value/return/raw-advantage scale each update.

## Held-out evidence

At source and each scheduled checkpoint, run one stochastic current-policy
self-play short episode in each of 4,096 fixed held-out worlds. Preserve touch,
goal and termination rates; duration; controller distributions; speed,
supersonic, ground/air, boost consumption/use/pickups, jump rising edges, actual
flip onsets and saves; and every signed/absolute reward component. The source
reward-scale gate must demonstrate `goal > save > normal V1 gameplay shaping >
incidental movement/resource unit shaping` without changing the authorized
coefficients. A no-touch fraction above 1% is an acquisition regression indicator,
not a one-evaluation automatic stop.

Publish the final resumable checkpoint, SHA-256, transition and launch gates,
reward-scale evidence, per-update training curve, all five evaluations, and a
compact report to `origin/main`, then stop.
