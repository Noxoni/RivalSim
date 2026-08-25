# RivalSim v0.4 Completion Boundary

RivalSim v0.4 is complete with **`PASS_GREEN`**.

The authorized v0.4 implementation is fixed to the standard-Soccar world containing exactly two
Octanes, one ball, the accepted static arena, 34 boost pads, goals/scoring, deterministic kickoff
resets, demolition disable/respawn, world/episode clocks, raw lifecycle events, and deterministic
full-world reset. The accepted v0.3 physics remains unchanged and all prior published evidence is
byte-for-byte intact.

Published custody and reproduction material:

- `docs/V0_4_RESULTS.md`;
- `docs/REPRODUCING_V0_4.md`;
- `docs/V0_4_AUTHORITY.md`;
- `results/v0.4/manifest.json`;
- `handoff/v0.4/README.md`;
- `handoff/v0.4/ACCEPTANCE.md`;
- `handoff/v0.4/LIFECYCLE_POLICY.md`.

No v0.5 work is authorized by this prompt. Stop at this boundary unless a later explicit handoff
authorizes the training-integration milestone. In particular, do not add observations, rewards,
training-specific action parsing, rollout buffers, tensor interop, PyTorch policy inference, GAE,
PPO, or Rival policy training.

Do not broaden RivalSim into arbitrary body counts, other game modes, rendering, a generic Bullet
API, or a Rocket League client replacement.
