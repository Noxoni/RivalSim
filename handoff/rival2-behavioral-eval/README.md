# Rival 2.0 Behavioral Trajectory Evaluation

This handoff authorizes one **evaluation-only** analysis of the completed 45.3B-sample Rival 2.0 checkpoint. Do not train, change rewards, build the viewer, or begin v0.6.

Required parent commit:

`fe057b12253d4e416a30c187853b03f2ec8f4d26`

Required checkpoint:

`checkpoints/rival2/overnight/rival2_overnight_final_6h_resume.pt`

Expected SHA-256:

`4DC158DC2A9D16B79FB5FE7D868E3B50928AB113B55DFCC753F3734F8D87372E`

Checkpoint reward must remain `RIVAL2_REWARD_V1`. This run is descriptive only. In particular, **do not remove or change the existing ball-progress reward in response to the results.** Publish the evidence and return for review.

## Evaluation

Run one large held-out **ordinary stochastic current-policy self-play** evaluation with the final checkpoint controlling both cars. Use:

- 16,384 worlds;
- all five standard kickoff layouts distributed deterministically;
- held-out seed `920260826` unless an independent telemetry seed is required, in which case record it explicitly;
- first completed episode per world;
- normal 30 Hz policy / 120 Hz physics;
- no historical opponent in this evaluation: final checkpoint versus itself on both sides.

The evaluation may instrument internal state/events at 120 Hz for telemetry, but it must not change simulator dynamics, controller inputs, observation/reward semantics, episode semantics, or policy outputs.

Canonicalize every touch from the **toucher's perspective**, where +Y is always the opponent goal direction. Do not label backward or lateral touches as bad; record what happened.

## Per-touch trajectory telemetry

For every unique accepted ball-touch entry, capture enough information to reconstruct and classify the resulting sequence:

- world, episode tick/time, toucher side;
- ball position immediately before/at the accepted touch;
- ball linear velocity immediately before the touch and immediately after the collision-resolved touch tick;
- speed before/after;
- canonical post-touch velocity components and heading angle relative to the opponent-goal center vector;
- immediate longitudinal velocity delta caused by the touch;
- instantaneous straight-line goal-plane intercept from the immediate post-touch velocity, if defined, including X/Z and whether that projection lies inside the authoritative goal mouth; **label this only as an instantaneous projection, not as a missed/made shot**;
- ball position/displacement at 0.25 s, 0.5 s, 1.0 s, and 2.0 s after the touch if the sequence is still alive and no later player touch has occurred;
- canonical net Y displacement from the touch until the next player touch, goal, or episode end;
- maximum forward Y excursion and maximum backward Y excursion before the next player touch/goal;
- whether the ball contacts ground, side wall, backboard/goal structure, or ceiling before the next player touch/goal, using source-backed arena/contact information where available;
- identity of next player touch: same player, opponent, or none before episode end;
- time to next touch;
- whether the same player gets one or more consecutive touches (primitive possession chain length);
- whether a goal occurs within 1 s / 3 s / 5 s after the touch and which side scores;
- whether this touch is the final player touch before a goal and time from touch to goal.

Publish aggregate distributions for at minimum:

- immediate post-touch direction: forward / lateral-neutral / backward using a declared symmetric threshold, plus the raw continuous distribution so the threshold is not the authority;
- actual net Y displacement before next touch;
- same-player next-touch rate versus opponent next-touch rate;
- possession-chain length distribution;
- wall/backboard/ceiling/ground continuation rates;
- touch-to-goal rates at 1/3/5 seconds;
- final-touch-before-goal rate;
- touch location by defensive / middle / attacking field thirds.

## Goal-entry / shot-placement telemetry

For every scored goal:

- scoring side;
- last toucher and time since last touch;
- ball center position and velocity at the scoring event;
- interpolate the trajectory to the authoritative goal-line/scoring plane where possible and record the X/Z crossing point;
- record absolute X/Z and normalized goal-mouth X/Z using source-backed goal geometry; if normalization cannot be sourced cleanly, publish absolute coordinates and the exact geometry used rather than inventing bounds;
- entry speed and velocity vector/angle;
- number of player touches in the scoring possession/sequence;
- arena-contact sequence since the final touch (direct, ground, side-wall, backboard/goal-structure, ceiling, or combinations).

Publish:

- 2D goal-mouth X/Z histogram/heatmap data;
- horizontal and vertical goal-entry distributions;
- center/corner and low/mid/high descriptive bins, with exact bin boundaries recorded;
- entry-speed distribution;
- direct-goal versus wall/backboard/other-contact scoring sequence rates.

Do **not** infer that a wall/backboard trajectory is inaccurate. Do not call a non-scoring touch a missed shot merely because its immediate straight-line projection is outside the goal mouth.

## Compact evidence

Publish under `results/rival2/behavioral_telemetry/` and a human-readable `docs/RIVAL2_BEHAVIORAL_TELEMETRY.md`:

- `summary.json`;
- `touch_trajectory_summary.json`;
- `goal_entry_summary.json`;
- `goal_mouth_histogram.json`;
- `possession_summary.json`;
- evaluator configuration/seed/checkpoint identity;
- compact raw event telemetry as compressed NPZ if practical.

If full raw touch telemetry would exceed roughly 50 MiB committed size, keep exact aggregates authoritative and commit a deterministic representative raw sample plus the full raw artifact SHA-256/path in the report rather than bloating the repository.

Report explicit counts so every percentage has a denominator.

## Boundaries

- no training updates;
- no reward/model/PPO/observation/action/episode/self-play changes;
- no reward recommendation implemented automatically;
- no viewer;
- no RocketSim/RLBot/v0.6 work;
- no preflight/regression/parity/pytest/Ruff/compileall ceremony.

Implement only the telemetry/evaluator needed to answer the behavioral questions, run it once on the final 45B checkpoint, publish the result, and stop.