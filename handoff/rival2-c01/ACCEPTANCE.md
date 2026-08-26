# Rival 2.0 Campaign 01 — Acceptance

Campaign 01 is complete when all items below are satisfied or when a genuine documented failure boundary is reached.

## Required execution

- start from release `cc3aa34e0bac4531c2750e0d05e2b4980621c642`;
- verify v0.5 implementation commit `676ef6bd3ca48376d706a2dbccbdec26fce3e4fb` remains the training implementation basis;
- verify all four Rival 2.0 contract hashes before training;
- use a fresh policy initialization and explicit recorded seed;
- use the frozen default PPO configuration including entropy coefficient `0.01`;
- complete the horizon-32 world-count capacity preflight;
- train through the first completed update at or beyond 100M agent decision samples;
- preserve initialization plus 10M/25M/50M/100M threshold checkpoint/evaluation snapshots.

## Training integrity gates

Every completed update must remain finite:

- no NaN/Inf observations, actions, rewards, values, log probabilities, advantages, returns, losses, gradients, parameters, or optimizer state;
- sampled buttons remain exactly binary;
- analog actions remain within `[-1,+1]`;
- selective resets remain valid;
- historical opponents, once eligible, remain frozen and gradient-free;
- policy-version/sample-age accounting remains correct;
- checkpoint reload must reproduce the next stochastic sample at least at the final campaign boundary;
- ordinary rollout/GAE/PPO path must retain zero routine H2D/D2H state traffic.

A numerical-integrity failure blocks continued training. Do not repair it by changing the frozen contracts or silently skipping bad samples.

## Fixed evaluation gates

Evaluation configuration and seeds must be frozen before the first PPO update and reused unchanged at all checkpoints.

For every checkpoint, publish the metrics specified in `CAMPAIGN.md`, including:

- termination/truncation mix;
- touches/goals/demos per simulated minute;
- episode duration;
- action activation/distribution statistics;
- policy exploration statistics;
- side-balanced deterministic checkpoint-vs-initialization results;
- side-balanced stochastic checkpoint-vs-initialization results.

Skill improvement is not a blocking acceptance requirement. Negative or flat outcomes must be published without reinterpretation.

## Checkpoint gates

- every threshold checkpoint has a SHA-256 and exact cumulative sample count;
- the final checkpoint loads under the frozen v0.5 contract/config checks;
- the final checkpoint is resumable from its saved training state;
- final inference outputs are deterministic under deterministic inference;
- the final checkpoint artifact is committed when it satisfies the campaign size policy, otherwise the compact inference artifact is committed and the full local resume artifact path/hash is documented.

## Regression and evidence gates

Before closeout:

- run the normal repository unit/quality checks relevant to files changed for campaign orchestration;
- verify published `results/v0.1/` through `results/v0.5/` remain byte-for-byte unchanged;
- do not require rerunning every expensive simulator authority corpus unless campaign code changes simulator/trainer implementation behavior; if implementation code is changed to make the campaign runnable, rerun the appropriate v0.5 correctness/regression suite before training continues;
- publish a compact machine-readable campaign summary and a human-readable campaign report.

Suggested committed evidence:

- `results/rival2/campaign01/config.json`;
- `results/rival2/campaign01/preflight.json`;
- `results/rival2/campaign01/checkpoints.json`;
- `results/rival2/campaign01/evaluation_000m.json`;
- `results/rival2/campaign01/evaluation_010m.json`;
- `results/rival2/campaign01/evaluation_025m.json`;
- `results/rival2/campaign01/evaluation_050m.json`;
- `results/rival2/campaign01/evaluation_100m.json`;
- `results/rival2/campaign01/training_curve.json`;
- `results/rival2/campaign01/summary.json`;
- `docs/RIVAL2_CAMPAIGN01_RESULTS.md`.

## Completion verdict

Campaign closeout should report two independent statuses:

1. execution status: `COMPLETE`, `STOP_NUMERICAL`, `STOP_ARCHITECTURAL`, or `STOP_RESOURCE`;
2. behavioral result: `CLEAR_EMERGENCE`, `WEAK_EMERGENCE`, `NO_CLEAR_EMERGENCE`, or `DEGRADED`.

Do not relabel v0.5 itself based on Campaign 01 behavior.

## Hard stop

No v0.6 RocketSim/RLBot loader or transfer evaluation is part of Campaign 01.