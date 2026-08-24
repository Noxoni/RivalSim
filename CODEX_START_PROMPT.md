# Active Codex Handoff — RivalSim v0.3 Dynamic Contacts

RivalSim v0.3 is now **authorized**.

Start from the current `origin/main`. The completed v0.2.2 release immediately below this handoff is:

`6dfd44ad9afeb3d1164da7e0e38c097fb74d07b8`

v0.2.2 is `PASS_GREEN` and its published evidence is frozen. Do not modify prior `results/v0.1/`, `results/v0.2/`, `results/v0.2.1/`, or `results/v0.2.2/` artifacts.

Before changing physics, read in full:

- `handoff/v0.3/README.md`;
- `handoff/v0.3/ACCEPTANCE.md`;
- `handoff/v0.3/SOURCE_PORT_POLICY.md`;
- `docs/V0_2_2_RESULTS.md`;
- `docs/V0_2_2_ORACLE_CACHE.md`;
- `results/v0.2.2/manifest.json`.

Mission order:

1. ball ↔ arena;
2. car ↔ ball;
3. car ↔ car plus the bounded physical bump/demo classification needed by that contact path;
4. integrated static + dynamic multi-contact validation;
5. regression, stress/residency, then performance.

For each phase, map the exact pinned RocketSim/Bullet source path **before** implementation, freeze a content-addressed native authority cache once, and debug GPU failures against cached operation-level truth. Do not use behavioral stabilizers, face/case exceptions, tie epsilons, tolerance broadening, or repeated live-RocketSim runs to fit outcomes.

Hard authoritative horizons remain ticks `1, 4, 8, 12`. Long open-loop identity is diagnostic only.

Do not begin v0.4 game rules or v0.5 training integration in this run.

Complete and push v0.3 only if every gate in `handoff/v0.3/README.md` and `handoff/v0.3/ACCEPTANCE.md` passes. Otherwise stop at the first genuine scope/performance/correctness boundary with all work preserved and an explicit report.
