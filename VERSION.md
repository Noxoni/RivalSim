# RivalSim Version Boundary

**Current completed milestone:** v0.5.0 — Rival 2.0 GPU-native training: `PASS_GREEN`

**Workflow/handoff version:** RocketSim reciprocal validation `v2`

**Latest completed executions:** final-45B behavioral telemetry, GPU-native public-Nexto full-match benchmark, and RivalSim kickoff-free Rival-vs-Nexto open-play benchmark — `COMPLETE`

**Latest completed execution commit:** `9807da8b3c404beb63a5426959132de549332128`

**Active authorized work:** build/validate a RocketSim -> `RIVAL2_OBS_V1` frozen-Rival adapter, then run normal 5-minute Rival-vs-Nexto RocketSim matches with kickoffs plus the RocketSim kickoff-free cross-check.

## Stable trained checkpoint

Final-45B Rival checkpoint:

`checkpoints/rival2/overnight/rival2_overnight_final_6h_resume.pt`

SHA-256:

`4DC158DC2A9D16B79FB5FE7D868E3B50928AB113B55DFCC753F3734F8D87372E`

Policy version / cumulative agent-decision samples:

`5403 / 45,323,649,024`

Reward remains frozen `RIVAL2_REWARD_V1`.

## Completed RivalSim evidence

Behavioral telemetry is complete under `docs/RIVAL2_BEHAVIORAL_TELEMETRY.md` and `results/rival2/behavioral_telemetry/`.

The GPU-native public Nexto port and RivalSim normal full-match benchmark are complete under `docs/RIVAL2_NEXTO_RESULTS.md` and `results/rival2/nexto/`.

The kickoff-free RivalSim open-play benchmark is complete under `docs/RIVAL2_NEXTO_OPEN_PLAY_RESULTS.md` and `results/rival2/nexto_open_play/` at commit `9807da8b3c404beb63a5426959132de549332128`.

Pinned Nexto:

`Rolv-Arild/Necto@2e6ed7d6ed2b352e8ff529d4a12a0c9c70c28cca`

model SHA-256:

`BF5343B5EEACAC6BF7CDB75DAC4A5C14BA0F94D820EAE75F00A211B6119D69FA`

### Completed kickoff-free reference result

Across 16,384 first-goal open-play duels, Rival won 8,786, Nexto won 7,255, and 343 drew. Overall decisive Rival win rate was `54.772%`.

The side-separated result is now a primary cross-validation target:

- Rival Blue: `3,753-4,241`, 198 draws, `46.948%` decisive win rate;
- Rival Orange: `5,033-3,014`, 145 draws, `62.545%` decisive win rate;
- Orange-minus-Blue decisive win-rate difference: approximately `15.597` percentage points.

The controls did not show a comparable physical-state imbalance: original/mirrored states were `55.158% / 54.384%`, and Rival inheriting the original Blue/Orange physical car was `54.678% / 54.866%`. The cause of the remaining team-side asymmetry is unresolved.

## Active reciprocal validation

The controlling handoff is:

`handoff/rival2-rocketsim-crosscheck/README.md`

Pinned RocketSim reference physics:

`ZealanL/RocketSim@c2baacb8f4b441dd8505e63c2aeb5a1679b60b02`

The active milestone must:

1. build a RocketSim-state -> frozen `RIVAL2_OBS_V1` adapter without changing the observation contract;
2. prove adapter parity and team/mirror symmetry on broad reference states before official gameplay;
3. run **normal 5:00 RocketSim 1v1 matches with ordinary kickoffs, goal resets and overtime** using frozen Rival and native/source Nexto;
4. compare those full-match results against the published RivalSim full-match benchmark, including kickoff behavior and Blue/Orange asymmetry;
5. reproduce the kickoff-free open-play comparison inside RocketSim and report the side split explicitly;
6. publish reciprocal/cross-simulator evidence and stop.

The adapter is the immediate implementation objective. The official gameplay benchmarks must not proceed through a failed or unexplained adapter parity/symmetry gate.

## Completed prior handoff

`handoff/rival2-nexto-open-play/README.md` is historical/completed. Do not execute it again. Its result is the `9807da8...` evidence package referenced above.

## Future curriculum note — not active

Fake-kickoff training, including opponents that backflip/retreat to boost and intentionally concede first contact, remains deferred until simulator/open-play validation is understood.

## Boundaries

No training is authorized. Do not change Rival rewards, PPO, policy architecture, `RIVAL2_OBS_V1`, `RIVAL2_ACTION_V1`, controller semantics, or either simulator's physics in response to the benchmark. Do not begin fake-kickoff curriculum work, build the viewer, or begin v0.6. Perform only targeted checks needed to trust the adapter and reciprocal benchmark.
