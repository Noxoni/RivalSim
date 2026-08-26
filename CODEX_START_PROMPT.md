# Closed Codex Boundary — Rival 2.0 Campaign 01 Complete

RivalSim v0.5 remains complete with `PASS_GREEN`. Rival 2.0 Campaign 01 has also completed its
exact bounded execution and is now closed.

Campaign 01 used the frozen v0.5 trainer and contracts from a fresh seed-`20260826`
initialization. The preferred 131,072-world, horizon-32 capacity passed preflight. Training
stopped at update 12 with 100,663,296 agent decision samples, the first completed PPO update at
or beyond 100,000,000. Initialization plus the first 10M/25M/50M/100M threshold checkpoints and
fixed evaluations were preserved and published.

Independent closeout statuses:

- execution status: `COMPLETE`;
- behavioral result: `DEGRADED`;
- frozen v0.5 trainer status: unchanged (`PASS_GREEN`).

The final full resumable checkpoint is published at
`checkpoints/rival2/campaign01/rival2_campaign01_100m_resume.pt`. Its SHA-256 is
`704F2B887BF50E767C86B7080C1E881644480D41A3302D245E833BDE65752B4A` and its exact size is
21,126,324 bytes. It passes the frozen v0.5 loader/config/contract checks and reproduces the next
stochastic sample exactly.

Read the completed evidence in:

- `docs/RIVAL2_CAMPAIGN01_RESULTS.md`;
- `results/rival2/campaign01/summary.json`;
- `results/rival2/campaign01/config.json`;
- `results/rival2/campaign01/checkpoints.json`;
- `results/rival2/campaign01/evaluation_000m.json` through `evaluation_100m.json`;
- `results/rival2/campaign01/training_curve.json`;
- `handoff/rival2-c01/README.md`;
- `handoff/rival2-c01/CAMPAIGN.md`;
- `handoff/rival2-c01/ACCEPTANCE.md`.

All published `results/v0.1/` through `results/v0.5/` remain byte-for-byte unchanged. Do not
reinterpret the negative behavioral outcome by changing the frozen campaign after the fact.

No further work is authorized by this prompt. Do not resume Campaign 01 training, begin v0.6,
build RocketSim/RLBot deployment or transfer validation, tune rewards, add curricula, or alter
the frozen v0.5 contracts. A new explicit controlling handoff is required for any subsequent
milestone.
