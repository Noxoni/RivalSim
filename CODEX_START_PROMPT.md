# Active Codex Handoff — Rival vs Nexto Kickoff-Free Open Play

The behavioral telemetry, GPU-native public-Nexto port, and full-match Rival-vs-Nexto benchmark are complete in the current lineage. Do not repeat them.

The kickoff-free RivalSim open-play benchmark remains the **active authorized work and must run first**. The later RocketSim reciprocal-validation/Rival-adapter handoff has been written and queued, but it does **not** replace, defer, interrupt, or supersede this active open-play test.

Start from current `origin/main` and read:

`handoff/rival2-nexto-open-play/README.md`

in full. Treat that file as the controlling requirement for the current execution.

## Current mission

Reuse the frozen final-45B Rival policy and the already-validated pinned public Nexto GPU port, harvest the specified 4,096 physically continuous ordinary open-play states, construct the required four-way role/side/mirror pairing, then run exactly 16,384 deterministic first-goal open-play duels with a 60-second maximum and no kickoff or goal reset anywhere in the duel.

Publish overall, side-separated, source-separated, role-separated and paired-family outcome evidence plus open-play touch/possession/trajectory telemetry.

Frozen Rival checkpoint SHA-256:

`4DC158DC2A9D16B79FB5FE7D868E3B50928AB113B55DFCC753F3734F8D87372E`

Pinned Nexto upstream:

`Rolv-Arild/Necto@2e6ed7d6ed2b352e8ff529d4a12a0c9c70c28cca`

Do not train Rival against Nexto yet. Do not change Rival rewards, PPO, architecture, observation/action contracts, controller behavior, or simulator physics. Do not begin fake-kickoff training; retain that as future curriculum work only. Do not build the viewer or begin v0.6. Avoid unrelated release/lint/regression ceremony; only the targeted integrity checks specified by the open-play handoff are authorized.

When complete, commit and push implementation/evidence to `origin/main`, return the final commit SHA and the kickoff-free open-play results, then stop for review.

## Queued next milestone — do not execute during this run

After the RivalSim kickoff-free open-play benchmark is complete and reviewed, the next intended milestone is the reciprocal RocketSim simulator-validation experiment defined in:

`handoff/rival2-rocketsim-crosscheck/README.md`

That later milestone will build a RocketSim -> `RIVAL2_OBS_V1` adapter for the same frozen Rival checkpoint, keep public Nexto in the RocketSim/RLGym-style environment it was built around, reproduce the Rival-vs-Nexto matchup there, and compare performance between RivalSim and RocketSim as a simulator-validation cross-check.

Do **not** begin that RocketSim adapter/cross-check until the current RivalSim open-play benchmark has finished and the user has reviewed its result.
