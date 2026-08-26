# Active Codex Handoff — RocketSim Reciprocal Validation v2

The last completed execution is the RivalSim kickoff-free Rival-vs-Nexto benchmark at commit:

`9807da8b3c404beb63a5426959132de549332128`

Do **not** repeat that benchmark. Its implementation and evidence are complete and are now reference evidence for the next objective.

## Active objective

Build and validate the adapter required to run the frozen final-45B Rival policy inside the pinned RocketSim/RLGym-style environment, while keeping public Nexto on its native/source RocketSim semantics.

Then execute **both** authorized RocketSim comparisons:

1. **Normal 1v1 full matches** — standard 5:00 regulation, ordinary kickoffs from all standard layouts, goal scoring and kickoff resets, and overtime when tied. This is the primary gameplay benchmark. Rival must play its normal frozen policy, including its own kickoff behavior; do not substitute a scripted Rival kickoff. Nexto must retain its stock source kickoff controller.
2. **Kickoff-free open play** — the paired restored-state benchmark used to isolate ordinary play from kickoff advantage and to compare the Blue/Orange asymmetry against RivalSim.

Read and execute the controlling handoff:

`handoff/rival2-rocketsim-crosscheck/README.md`

Required adapter flow:

`RocketSim state -> RIVAL2_OBS_V1 adapter -> frozen Rival policy -> native 8 controls -> RocketSim`

Frozen Rival checkpoint SHA-256:

`4DC158DC2A9D16B79FB5FE7D868E3B50928AB113B55DFCC753F3734F8D87372E`

Pinned Nexto upstream:

`Rolv-Arild/Necto@2e6ed7d6ed2b352e8ff529d4a12a0c9c70c28cca`

Pinned Nexto model SHA-256:

`BF5343B5EEACAC6BF7CDB75DAC4A5C14BA0F94D820EAE75F00A211B6119D69FA`

Pinned RocketSim reference physics:

`ZealanL/RocketSim@c2baacb8f4b441dd8505e63c2aeb5a1679b60b02`

## Critical side-asymmetry target

The completed RivalSim kickoff-free benchmark found:

- Rival as Blue: `46.948%` decisive win rate;
- Rival as Orange: `62.545%` decisive win rate;
- Orange-minus-Blue difference: approximately `15.597` percentage points;
- original states: `55.158%`;
- mirrored states: `54.384%`;
- Rival inheriting original Blue car: `54.678%`;
- Rival inheriting original Orange car: `54.866%`.

Adapter parity must therefore be checked separately for Blue and Orange and with exact team/mirror pairs before official gameplay. Both normal-match and kickoff-free results must preserve Blue/Orange as explicit dimensions; never hide them inside an aggregate-only result.

No training is authorized. Do not change Rival or Nexto weights, rewards, PPO, policy architecture, observation/action contracts, controller behavior, or either simulator's physics. Do not begin fake-kickoff curriculum work, viewer work, or v0.6.

When the complete reciprocal-validation handoff is finished, commit and push implementation/evidence to `origin/main` and stop for review.
