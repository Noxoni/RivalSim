# Active Codex Handoff — RivalSim v0.4 Complete Standard 1v1 Game Transition

RivalSim v0.4 is now **authorized**.

Start from the current `origin/main`. The completed v0.3 release immediately below this handoff is:

`d6ca3912418a3dd7ca8979415142cd861e0c0ddb`

The v0.3 implementation boundary is:

`a63d317b0de0522e6d3cbe243bf282c6b93a9d58`

v0.3 is `PASS_GREEN`; all published v0.1 through v0.3 evidence is frozen and must remain byte-for-byte unchanged.

Before changing runtime behavior, read in full:

- `handoff/v0.4/README.md`;
- `handoff/v0.4/ACCEPTANCE.md`;
- `handoff/v0.4/LIFECYCLE_POLICY.md`;
- `docs/V0_3_RESULTS.md`;
- `docs/V0_3_ORACLE_CACHE.md`;
- `results/v0.3/manifest.json`.

Mission order:

1. freeze the standard-Soccar 1v1 lifecycle/source map and new v0.4 authority;
2. integrate the existing 34-pad pickup/cooldown state into complete world reset and episode lifecycle;
3. implement goals/scoring plus deterministic standard kickoff/reset transitions;
4. implement demolition disable/removal-from-physics and source-correct respawn while preserving car-container visitation semantics;
5. implement bounded match/reset state and generic terminal/truncation event outputs needed by the later training layer;
6. validate complete headless two-Octane/one-ball 1v1 episodes, including repeated goals, kickoffs, pad cycles, demos, and respawns;
7. rerun all v0.3/v0.2.2/v0.1 regressions, deterministic stress/residency, then final performance and reset-heavy throughput gates.

Use the exact pinned RocketSim source/build as primary authority for behavior it defines. Where a training-facing lifecycle decision is not defined by RocketSim, do not silently invent or import policy: freeze an explicit v0.4 contract or stop at the decision boundary.

Do not rebuild the validated v0.3 physics. Do not use case-specific rules, expected-output lookup, hidden tables, tolerance broadening, or host-side fixes. Any randomness used for kickoff/respawn selection must be explicit, deterministic, and per-world/stateful so authority is reproducible.

Do not begin v0.5 observations, rewards, action parsing, rollout buffers, PyTorch integration, PPO, or Rival policy inference in this run.

Complete and push v0.4 only if every gate in `handoff/v0.4/README.md` and `handoff/v0.4/ACCEPTANCE.md` passes. Otherwise stop at the first genuine source/semantics/performance boundary with all work preserved and an explicit report.
