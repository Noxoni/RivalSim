# Rival 2.0 acquisition restart authority

This package authorizes one fresh Rival 2.0 ball-acquisition campaign after the
mechanics-correctness checkpoint. It supersedes the abandoned full-match
acquisition attempt only; it does not rewrite or invalidate its retained evidence.

## Fixed training lifecycle

- 131,072 GPU-native RivalSim worlds.
- Fresh policy, optimizer, RNG, counters, and historical opponent state.
- Standard kickoff initialization.
- Original `RIVAL2_EPISODE_V1`: a goal ends the episode, 15 seconds without any
  touch truncates and resets the episode, and 45 seconds is the hard limit.
- 30 Hz policy decisions and 120 Hz physics.
- No five-minute matches in training or held-out acquisition evaluation.
- No Nexto training and no simulator, observation, action, PPO, or mechanics change.

## `RIVAL2_REWARD_ACQUISITION_V1`

At each 30 Hz decision, each player's reward is the sum of:

- Goal: +10 to the scorer and -10 to the opponent.
- Canonical ball progress: the existing zero-sum
  `0.5 * delta_ball_y / 5120` term.
- True 3D approach: `(distance_before - distance_after) / 4096`, independently
  per player and calculated from the final pre-reset transition state.
- First legitimate touch by that player during the episode: +1.0 once. It stacks
  with the ordinary touch reward.
- Each legitimate unique touch by that player: +0.20. Sustained contact remains
  latched and is not repeatedly rewarded; a genuine recontact is another event.
- Existing unique-demolition reward: +0.10/-0.10, unchanged and zero-sum.
- At the 15-second no-touch boundary: -0.5 to each player who has not touched at
  any earlier point in that episode. A player who already touched is not penalized.

There is no direct reward for jump, boost, speed, flip, wavedash, aerial, or any
other mechanic.

## Held-out gate

Every 30 PPO updates, evaluate one stochastic current-policy self-play episode in
each of 4,096 held-out worlds using the same short lifecycle. Acquisition is
complete only after the no-touch truncation fraction is at most 1% in two
consecutive evaluations. There is no sample cap. Preserve the evaluation curve,
checkpoint hashes, controller distributions, first-touch latency, touch and goal
rates, boost use/level/starvation, speed/supersonic time, grounded/airborne time,
car-ball distance, jump rising edges, and actual flip onsets.

Stop at acquisition completion. A later reward transition requires separate
authorization. Do not automatically run an additional timed continuation or a
five-minute match set.
