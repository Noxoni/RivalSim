# RivalSim Version Boundary

**Current completed milestone:** v0.5.0 — Rival 2.0 GPU-native training: `PASS_GREEN`

**Latest completed executions:** final-45B behavioral telemetry and GPU-native public-Nexto full-match benchmark — `COMPLETE`

**Active authorized work:** kickoff-free Rival-vs-Nexto open-play evaluation only

## Stable trained checkpoint

Final-45B Rival checkpoint:

`checkpoints/rival2/overnight/rival2_overnight_final_6h_resume.pt`

SHA-256:

`4DC158DC2A9D16B79FB5FE7D868E3B50928AB113B55DFCC753F3734F8D87372E`

Policy version / cumulative agent-decision samples:

`5403 / 45,323,649,024`

Reward remains the frozen base `RIVAL2_REWARD_V1`.

## Completed evidence

Behavioral telemetry is complete under `docs/RIVAL2_BEHAVIORAL_TELEMETRY.md` and `results/rival2/behavioral_telemetry/`.

The GPU-native public Nexto port and full-match benchmark are complete under `docs/RIVAL2_NEXTO_RESULTS.md` and `results/rival2/nexto/`.

Pinned Nexto:

`Rolv-Arild/Necto@2e6ed7d6ed2b352e8ff529d4a12a0c9c70c28cca`

model SHA-256:

`BF5343B5EEACAC6BF7CDB75DAC4A5C14BA0F94D820EAE75F00A211B6119D69FA`

The full-match benchmark showed final Rival winning essentially every match, but direct kickoff goals accounted for most Rival scoring. The full-match result therefore does not by itself establish the size of Rival's open-play advantage over Nexto.

## Active evaluation

The only authorized work is:

`handoff/rival2-nexto-open-play/README.md`

It requires a kickoff-free, first-goal open-play benchmark built from 4,096 physically continuous harvested mid-play states:

- 2,048 from final-Rival stochastic self-play;
- 2,048 from deterministic pinned-Nexto self-play.

Every base state is replayed four ways: original and exact 180-degree/team-swapped mirror, with Rival and Nexto swapping physical car roles. Total official duels: **16,384**.

Each duel begins directly in open play, has no kickoff and no goal reset, ends on the first goal, and draws at 60 simulated seconds if still unresolved. Results must remain separated by Rival side, source distribution, mirror status, role assignment, initial state characteristics, and four-duel paired family.

## Future curriculum note — not active

The user intends to train later against fake-kickoff behavior, including an opponent immediately backflipping/retreating to boost and conceding first contact so Rival's kickoff hit is received by the defender. This is deliberately deferred until after open-play skill is measured.

## Boundaries

No training is authorized. Do not change Rival rewards, PPO, policy architecture, `RIVAL2_OBS_V1`, `RIVAL2_ACTION_V1`, controller semantics, or simulator physics. Do not begin fake-kickoff curriculum work, build the viewer, or begin v0.6. General release/lint/regression ceremony remains out of scope; perform only the targeted integrity checks required by the active handoff.
