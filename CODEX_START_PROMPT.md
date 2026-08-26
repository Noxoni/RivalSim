# Active Codex Handoff — RocketSim Reciprocal Validation

The final-45B behavioral telemetry and GPU-native public-Nexto RivalSim benchmark are complete. Do not repeat them.

The prior RivalSim-only kickoff-free open-play handoff is **deferred**. The user wants the reciprocal simulator validation first.

Start from current `origin/main` and read:

`handoff/rival2-rocketsim-crosscheck/README.md`

in full. Treat that file as the controlling requirement.

## Mission

Keep pinned public Nexto in the RocketSim/RLGym-style environment it was built around, build only the adapter required to run frozen final-45B Rival from RocketSim state, then compare the frozen Rival-vs-Nexto matchup across RocketSim and the already-published RivalSim result.

Frozen Rival checkpoint SHA-256:

`4DC158DC2A9D16B79FB5FE7D868E3B50928AB113B55DFCC753F3734F8D87372E`

Pinned Nexto:

`Rolv-Arild/Necto@2e6ed7d6ed2b352e8ff529d4a12a0c9c70c28cca`

Expected Nexto model SHA-256:

`BF5343B5EEACAC6BF7CDB75DAC4A5C14BA0F94D820EAE75F00A211B6119D69FA`

Pinned RocketSim reference:

`ZealanL/RocketSim@c2baacb8f4b441dd8505e63c2aeb5a1679b60b02`

Execute the complete authorized sequence without returning for approval between phases:

1. build and validate a RocketSim -> `RIVAL2_OBS_V1` frozen-Rival adapter;
2. reproduce the prior full-match Rival-vs-Nexto protocol in pinned RocketSim;
3. publish a direct RivalSim-vs-RocketSim performance comparison;
4. run the specified kickoff-free open-play Rival-vs-Nexto benchmark in RocketSim;
5. commit/push all implementation and evidence, then stop.

This is evaluation only. Do not train Rival or Nexto. Do not tune either simulator or policy in response to the results. Do not change Rival rewards, PPO, architecture, observation/action contracts, or physics. Do not begin fake-kickoff curriculum work, build the viewer, or begin v0.6. Avoid unrelated release/lint/regression ceremony; perform only the targeted integrity and adapter-parity checks required by the handoff.
