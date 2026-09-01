# Corrected fresh-human-seed PPO campaign: blocked evidence

## Verdict

The exploration-ramp implementation, frozen authority, focused tests, and exact-scale
preflight passed. The campaign accepted zero PPO updates because the first optimizer
proposal exceeded the existing hard minibatch KL limit at every authorized learning
rate. Each proposal was transactionally rolled back.

## Exact failed transition

- Source: Stage-1 selected step 2800
- Source SHA-256: `CA2DB62A709BBD7DBA9D2997D701E2E6010584F8119BDA6F5D1686AD7425F9D2`
- Effective update-1 analog sigma: `0.01`
- Effective update-1 button temperature: `0.02`
- Hard minibatch KL limit: `0.10`
- Post-step minibatch KL at policy LR `1e-4`: `1309.495361328125`
- Post-step minibatch KL at policy LR `5e-5`: `328.05499267578125`
- Post-step minibatch KL at policy LR `2.5e-5`: `81.37278747558594`
- Final accepted update: `0`
- Transactional rollback: completed for every proposal
- Saved rollback checkpoint iteration/policy version: `0 / 0`
- Saved rollback checkpoint optimizer-state entries: `0`
- Historical policy count: `0`

The near-deterministic effective distribution makes a normal fresh-Adam actor update
extremely large in distribution space. The package explicitly prohibits inventing an
additional learning-rate schedule and permits stopping when the existing hard guard and
authorized retry/backoff cannot recover. No safety limit, reward, source model, or
exploration parameter was changed to force continuation.

## Consequences

The 600-update campaign and post-training Nexto comparison were not run because no safe
PPO update was accepted and therefore no trained candidate exists. The preserved
`u0000` checkpoint is rollback/provenance evidence, not a trained successor.

## Required next authority

A new prospective transition authority is required before another run. It should permit
a rollback-only search below `2.5e-5` or another explicitly specified trust-region-aware
optimizer transition for the narrow scheduled distribution, while retaining the current
KL guards. This cannot be retrofitted into the frozen V1 package without violating its
learning-rate and safety requirements.

Machine-readable details are in `ppo_hard_safety_failure.json`, `ppo_summary.json`,
`ppo_first_rollout.json`, `ppo_preflight.json`, and `ppo_snapshot_manifest.json`.
