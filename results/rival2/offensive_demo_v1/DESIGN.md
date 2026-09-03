# Productive offensive-demolition scaffold V1

Date: 2026-09-03

Status: scenario and physical-outcome telemetry only. No production reward,
policy, optimizer, opponent mix, or running campaign is changed by this work.

## Baseline

The promoted V23 ten-match Nexto artifact
`results/rival2/codex_autonomous_v23/full_match_evaluation.json` reports exactly
zero Rival demolitions overall, from both team perspectives, and from every
starting layout. Nexto also recorded zero in that bounded evaluation. V23 has
therefore not demonstrated the requested demolition behavior.

## Why the historical capability-V2 route is insufficient

The historical route proved that the simulator emits an authoritative
`lifecycle.opponent_demoed_event`, but it did not prove useful offensive intent:

- the main event payment occurred immediately on any demo in the scenario;
- "offensive context" meant only that the absolute ball position was in one
  half of the field;
- any self touch within five seconds counted as a follow-up, regardless of
  whether it recovered possession or moved the ball goalward;
- the tracker multiplied ball Y by an attacker-side direction even though
  `Rival2TensorBridge.observation()` already team-normalizes every perspective.
  This made the old offensive-half test asymmetric for Orange.

The historical code remains unchanged for audit. V1 introduces a separate
corrected scaffold rather than silently changing an old result.

## New physical routes

`rivalsim/rival2_offensive_demo_v1.py` supplies two side-symmetric routes:

1. `recover_possession`: the defender is ahead of Rival and between Rival and a
   reachable loose ball;
2. `open_goal`: the moving ball is ahead of Rival and the defender is goal-side
   of the ball.

The defender begins 250--550 uu off the direct ball lane. Rival starts pointed
goalward at a physically supersonic speed, so a demolition requires a deliberate
off-axis route and a later recovery toward the play. This avoids training a
trivial straight-line collision that happens to contain a ball somewhere else.

## Outcome semantics

The tracker distinguishes and counts these facts independently:

- any authoritative opponent demolition;
- a demolition while the defender and ball satisfy the route's physical
  offensive context;
- a subsequent Rival ball touch within a bounded three-second window;
- at least 300 uu of subsequent goalward ball progress, requiring a Rival touch
  on the recover-possession route;
- a subsequent goal;
- expiration of the advantage without any conversion.

All observations are interpreted in their existing team-normalized coordinates,
where positive Y is goalward for both sides. A held event signal is latched once,
and post-demo outcomes must occur on a later tick. These are physical telemetry
semantics, not named-mechanic inference.

No reward coefficients have been selected. A prospective training authority
must first run the native scenarios and measure V23's baseline collision,
conversion, and false-positive rates after the GPU is available.
