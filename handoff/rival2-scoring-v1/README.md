# Rival 2.0 scoring curriculum V1

This handoff records the user-authorized curriculum transition issued on
2026-08-27. It supersedes the older root no-training boundary only for this
bounded run and does not authorize Nexto training, simulator changes, v0.6, or
any continuation after the scheduled +239 evaluation.

## Immutable start

- Source commit: `61307571d86508f3026402c4948f759f310ff36c`.
- Source checkpoint:
  `checkpoints/rival2/acquisition_v1/rival2_acquisition_resume.pt`.
- Source SHA-256:
  `4FB7A3B134B25D595374E3968E2EDFA150A9CD6F8910B903BF892B59D7F8BC9A`.
- Source reward: `RIVAL2_REWARD_ACQUISITION_V1`.
- Source episode: `RIVAL2_EPISODE_V1`.

Model, optimizer, policy/CPU/CUDA RNG, counters, self-play configuration,
opponent assignment, and the frozen historical pool are preserved exactly.
Because the episode contract changes, simulator and match state start freshly
under `RIVAL2_EPISODE_FULL_MATCH_V1`.

## Training contract

- 131,072 resident standard-Soccar 1v1 worlds.
- Five-minute regulation, ordinary goals and kickoff resets, persistent score,
  and sudden-death overtime when regulation is tied.
- No no-touch or hard-time truncation.
- 30 Hz policy, 120 Hz physics, horizon 32.
- Existing entropy-zero PPO and current/historical self-play configuration.
- PPO update boundaries never reset a match.
- Exactly 239 additional PPO updates / 2,004,877,312 additional agent decisions.
- Checkpoints and compact evaluations at offsets +60, +120, +180, and +239.
- Stop after the +239 evaluation; no extra match set or timed continuation.

## `RIVAL2_REWARD_SCORING_V1`

For each agent and each 30 Hz decision interval:

```text
reward = goal
       + 0.5 * signed_delta_ball_y_toward_opponent_goal / 5120
       + 0.10 * (car_ball_distance_before - car_ball_distance_after) / 4096
       + 0.02 * unique_own_touch_onsets
       + 0.10 * (own_demo_onsets - opponent_demo_onsets)
       - 0.002 * jump_button_rising_edges
       - 0.01 * actual_directional_flip_or_dodge_onsets
```

`goal` is +10 for scoring and -10 for conceding. There is no first-touch
bonus, no no-touch failure penalty, no airborne/grounded occupancy cost, no
jump-hold cost, and no direct reward for named mechanics, speed, boost,
supersonic state, or recovery. Approach and mechanic terms use the final
pre-reset state, so kickoff reset motion never leaks into reward.

## Evaluation boundary

At the source checkpoint and each scheduled scoring checkpoint, run a compact
held-out stochastic full-match self-play evaluation with movement/controller
telemetry. At each scoring checkpoint also run 256 matches with the current
policy as Blue and 256 as Orange against the immutable acquisition checkpoint.
Report sides separately and never train against that reference.

The counterfactual fraction of kickoff segments with no touch within 15 seconds
is telemetry only. Values above 1% are an acquisition-regression warning and do
not alter or truncate matches.
