# Active Codex Handoff — Rival vs Nexto Kickoff-Free Open Play

The behavioral telemetry, GPU-native Nexto port, and full-match Rival-vs-Nexto benchmark are complete in the current lineage. Do not repeat them.

The completed benchmark showed that final-45B Rival dominated public Nexto in full matches, but the scoreline was heavily driven by direct kickoff goals. The next authorized work is therefore a kickoff-free open-play skill evaluation only.

Start from current `origin/main` and read:

`handoff/rival2-nexto-open-play/README.md`

in full. Treat that file as the controlling requirement.

Mission: reuse the frozen final-45B Rival policy and the already-validated pinned public Nexto GPU port, harvest the specified 4,096 physically continuous ordinary open-play states, construct the required four-way role/side/mirror pairing, then run exactly 16,384 deterministic first-goal open-play duels with a 60-second maximum and no kickoff or goal reset anywhere in the duel. Publish overall, side-separated, source-separated, role-separated and paired-family outcome evidence plus open-play touch/possession/trajectory telemetry.

Frozen Rival checkpoint SHA-256:

`4DC158DC2A9D16B79FB5FE7D868E3B50928AB113B55DFCC753F3734F8D87372E`

Pinned Nexto upstream:

`Rolv-Arild/Necto@2e6ed7d6ed2b352e8ff529d4a12a0c9c70c28cca`

Do not train Rival against Nexto yet. Do not change Rival rewards, PPO, architecture, observation/action contracts, controller behavior, or simulator physics. Do not begin fake-kickoff training; record the user's fake-kickoff/backflip-to-boost opponent idea as future curriculum work only. Do not build the viewer or begin v0.6. Avoid unrelated release/lint/regression ceremony; only the targeted integrity checks specified by the handoff are authorized.

When complete, commit and push implementation/evidence to `origin/main`, return the final commit SHA and the kickoff-free open-play results, then stop for review.
