# RivalSim Version Boundary

**Current completed milestone:** v0.5.0 — Rival 2.0 GPU-native training: `PASS_GREEN`

**Latest completed executions:** final-45B behavioral telemetry and GPU-native public-Nexto full-match benchmark — `COMPLETE`

**Active authorized work:** reciprocal RocketSim simulator cross-validation and RocketSim kickoff-free open-play benchmark

## Stable trained checkpoint

Final-45B Rival checkpoint:

`checkpoints/rival2/overnight/rival2_overnight_final_6h_resume.pt`

SHA-256:

`4DC158DC2A9D16B79FB5FE7D868E3B50928AB113B55DFCC753F3734F8D87372E`

Policy version / cumulative agent-decision samples:

`5403 / 45,323,649,024`

Reward remains frozen `RIVAL2_REWARD_V1`.

## Completed evidence

Behavioral telemetry is complete under `docs/RIVAL2_BEHAVIORAL_TELEMETRY.md` and `results/rival2/behavioral_telemetry/`.

The GPU-native public Nexto port and RivalSim full-match benchmark are complete under `docs/RIVAL2_NEXTO_RESULTS.md` and `results/rival2/nexto/`.

Pinned Nexto:

`Rolv-Arild/Necto@2e6ed7d6ed2b352e8ff529d4a12a0c9c70c28cca`

model SHA-256:

`BF5343B5EEACAC6BF7CDB75DAC4A5C14BA0F94D820EAE75F00A211B6119D69FA`

The RivalSim full-match benchmark showed Rival winning essentially every match, but direct kickoff goals accounted for most Rival scoring. It also showed a material Blue/Orange scoring asymmetry. Those findings motivate reciprocal validation rather than immediate training changes.

## Active reciprocal validation

The controlling handoff is:

`handoff/rival2-rocketsim-crosscheck/README.md`

Pinned RocketSim reference physics:

`ZealanL/RocketSim@c2baacb8f4b441dd8505e63c2aeb5a1679b60b02`

The active milestone must:

1. build a RocketSim-state -> frozen `RIVAL2_OBS_V1` adapter and prove targeted observation/action parity;
2. run the same Rival-vs-Nexto full-match protocol in RocketSim while Nexto uses its upstream/source observation/model/action semantics;
3. compare RocketSim matchup distributions directly with the already-published RivalSim benchmark;
4. run a kickoff-free open-play Rival-vs-Nexto benchmark inside RocketSim;
5. publish all reciprocal/cross-simulator evidence and stop.

The comparison is intended to validate RivalSim behaviorally. It does not require identical trajectories; it asks whether matchup ordering, scoring rates, kickoff behavior, side asymmetry and open-play results remain broadly consistent across the two simulators.

## Deferred RivalSim-only open-play handoff

`handoff/rival2-nexto-open-play/README.md` remains in the repository as a prior plan but is **not active** during this milestone. Do not execute it unless explicitly reauthorized later.

## Future curriculum note — not active

The user intends to train later against fake-kickoff behavior, including opponents that backflip/retreat to boost and intentionally concede first contact so Rival's kickoff hit is received by the defender. This remains deferred until simulator/open-play validation is understood.

## Boundaries

No training is authorized. Do not change Rival rewards, PPO, policy architecture, `RIVAL2_OBS_V1`, `RIVAL2_ACTION_V1`, controller semantics, or either simulator's physics in response to the benchmark. Do not begin fake-kickoff curriculum work, build the viewer, or begin v0.6. General release/lint/regression ceremony remains out of scope; perform only targeted checks required to trust the adapter and reciprocal benchmark.
