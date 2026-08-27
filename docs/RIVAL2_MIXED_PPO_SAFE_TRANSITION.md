# Rival 2.0 mixed-PPO safe transition

Verdict: `PASS_GREEN`.

This is an exact disposable update-360 replay through the production mixed-curriculum optimizer path. It did not resume the +120 campaign and wrote no training checkpoint.

## Production replay

- accepted optimizer steps: `19` / `154`;
- maximum post-step minibatch KL: `0.017498828`;
- completed-update mean KL: `0.005559046`;
- retention-corpus mean KL: `0.019727562`;
- policy LR start/end: `0.0001` / `2.5e-05`;
- retries/backoffs: `3` / `2`;
- PPO early stop: `true` (`retention_mean_kl_at_minimum_policy_lr`).

The fixed `1e-4` strategy previously completed all 154 steps with maximum minibatch KL `0.017498828` and completed mean KL `0.010052922`. The production replay matched that maximum before the independent retention probe became binding. It accepted prior safe steps, backed policy LR down without changing critic LR, and stopped at the configured minimum rather than accepting a corpus-KL violation.

## Fixed retention corpus

- observations/dimension: `512` / `182`;
- observation-content SHA-256: `0085011CECDF7BFCADCFD832AC7D3FDBF5A08A7D36E84E0A8AF85C827947E1C7`;
- source checkpoint SHA-256: `77BF257131FB71DDEAEAE49D668C5E25AB1D06EE26149AB0D0AE303573CA5F21`;
- selected categories: `96` near-ball, `96` approach, `80` recovery, `96` airborne, `96` ordinary-ground, and `48` remaining-diversity states;
- field/orientation coverage: `3/3` x regions, `3/3` y regions, `8/8` heading octants.

## Family statistics

| family | samples | raw advantage mean/std | normalized mean/std | return mean/std | value mean/std | empirical KL |
|---|---:|---:|---:|---:|---:|---:|
| current | 1668800 | -0.009848/0.093854 | -0.000000/1.000000 | 0.081374/0.101883 | 0.091222/0.108042 | 0.005631865 |
| historical | 416864 | 0.167361/0.620083 | -0.000000/1.000000 | 1.476120/1.860044 | 1.308759/1.792467 | 0.005815286 |
| nexto | 1467488 | -0.047747/0.462011 | -0.000000/1.000000 | 0.852542/0.565896 | 0.900289/0.687704 | 0.005046330 |
| wisp | 1475552 | 0.790694/0.573805 | -0.000000/1.000000 | 2.661029/1.057386 | 1.870335/1.248113 | 0.005914213 |

## Transactional retry proof

Verdict: `PASS_GREEN`. The diagnostic-only `0.002` soft target forced the same first minibatch to roll back and retry at `5e-5`. Model parameters, Adam moments, and Adam step counters restored exactly before only the policy-group LR changed; the critic group remained at `3e-4`.

## Boundary

Nexto, Wisp, Gameplay V2, opponent probabilities, physics, lifecycle, network architecture, and both hard KL limits were unchanged. No live Rival training continuation was started.
