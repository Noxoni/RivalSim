# Active Codex Handoff - Rival 2.0 Opponent Curriculum V1

The active objective is the bounded mixed-opponent Gameplay V2 continuation in:

`handoff/rival2-opponent-curriculum-v1/README.md`

Read the entire package before implementation:

- `handoff/rival2-opponent-curriculum-v1/README.md`
- `handoff/rival2-opponent-curriculum-v1/WISP_SOURCE.md`
- `handoff/rival2-opponent-curriculum-v1/PACKAGE_MANIFEST.md`

Use only the selected healthy Gameplay V1 +239 checkpoint:

`checkpoints/rival2/gameplay_v1/rival2_gameplay_resume.pt`

Expected SHA-256:

`77BF257131FB71DDEAEAE49D668C5E25AB1D06EE26149AB0D0AE303573CA5F21`

The required experiment-source commit is:

`1a4437fe92fa7ab66efd0e4100d74bb90302ea46`

It must be an ancestor of the implementation head; do not reset `main` back to
that commit after this package is published.

Implement the pinned Wisp v2-75B adapter and its targeted fidelity gate, create
`RIVAL2_REWARD_GAMEPLAY_V2` as the immutable Gameplay V1 reward plus only the
strict successful-double-dash `+0.005` competitive event, then run the authorized
35% Nexto / 35% Wisp / 20% current self-play / 10% historical Rival short-episode
curriculum.

Preserve the existing transactional KL safety guard. Stop immediately on a
rejected policy update; otherwise stop after exactly +120 PPO updates and the
scheduled compact evaluations. Do not train Nexto/Wisp, use imitation learning,
run five-minute training matches, alter physics/PPO/network/observation/action
contracts, begin v0.6, or continue beyond +120.
