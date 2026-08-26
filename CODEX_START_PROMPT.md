# Active Codex Handoff — Rival 2.0 Campaign 02

RivalSim v0.5 remains complete with `PASS_GREEN`. Rival 2.0 Campaign 01 completed correctly but produced a `DEGRADED` behavioral result. This handoff authorizes one controlled follow-up campaign to test the identified entropy-pressure failure mode.

Start from current `origin/main`. Required starting HEAD:

`1ce5932cadd66b14032e61750836763499567bc9`

Do not modify or reinterpret Campaign 01 evidence. All published `results/v0.1/` through `results/v0.5/`, all Campaign 01 artifacts, and the four frozen Rival 2.0 environment/action contracts remain immutable.

Read in full before starting:

- `handoff/rival2-c02/README.md`;
- `handoff/rival2-c02/DIAGNOSIS.md`;
- `handoff/rival2-c02/CAMPAIGN.md`;
- `handoff/rival2-c02/ACCEPTANCE.md`;
- `docs/RIVAL2_TRAINING_CONTRACT.md`;
- `docs/RIVAL2_CAMPAIGN01_RESULTS.md`;
- `results/rival2/campaign01/training_curve.json`;
- `results/rival2/campaign01/evaluation_000m.json`;
- `results/rival2/campaign01/evaluation_100m.json`.

## Controlling change

Campaign 02 changes exactly one learning hyperparameter from Campaign 01:

`entropy_coefficient: 0.01 -> 0.0`

Everything else remains the same unless this handoff explicitly says otherwise. In particular, preserve the same model architecture, initialization procedure, observation/action/reward/episode contracts, gamma, GAE lambda, PPO clip range, value coefficient, gradient limit, Adam learning rate, epochs, horizon, minibatch target, self-play rules, historical-opponent rules, world count, campaign seed, evaluation seed, and evaluation protocol.

Do not edit the frozen v0.5 PPO or policy implementation merely to change this value. Instantiate a Campaign 02 PPO configuration with `entropy_coefficient=0.0` at the campaign layer. The existing entropy metric may continue to be logged diagnostically but must contribute zero weight to the Campaign 02 optimization loss.

## Mission

1. prove the Campaign 02 initialization is the same model initialization used by Campaign 01 and rerun the unchanged initialization evaluation;
2. run the same 131,072-world, horizon-32 training configuration from scratch with entropy coefficient exactly zero;
3. train through the first completed update crossing 100,000,000 agent decision samples;
4. preserve checkpoints/evaluations at initialization and the first completed updates crossing 10M, 25M, 50M, and 100M samples;
5. use the exact Campaign 01 held-out evaluation protocol and seeds so the campaigns are directly comparable;
6. publish per-update PPO metrics including approximate KL, clip fraction, policy/value loss, gradient norm, diagnostic entropy, and analog policy standard deviations;
7. publish a direct Campaign 01 vs Campaign 02 comparison and an independent Campaign 02 behavioral classification;
8. preserve the final full resumable checkpoint and stop.

Do not change reward shaping, add curricula, add action masks, change the continuous-control distribution, change the model, tune learning rate, introduce KL early stopping, or perform a hyperparameter search in this run. If a non-finite/correctness/integrity failure occurs, stop with evidence. Otherwise complete the bounded 100M campaign even if behavior is poor.

Do not begin v0.6 RocketSim/RLBot transfer work.