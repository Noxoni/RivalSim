# Rival 2.0 Campaign 02 — Execution Contract

## Objective

Run a fresh 100M-sample Rival 2.0 campaign that differs from Campaign 01 only by setting PPO entropy coefficient to zero.

## Initialization control

Use campaign seed `20260826` and the same initialization procedure as Campaign 01.

Before the first optimizer update:

1. verify the fresh model state SHA-256 equals Campaign 01 initialization model SHA-256:
   `890F224879DB6E458472985B226A664D8AE49B8303C21CFB0FD83A485CF42848`;
2. run the same initialization evaluation using evaluation seed `920260826`;
3. confirm the unchanged evaluation metrics match Campaign 01 initialization within exact/deterministic expectations of the existing evaluator;
4. record any unavoidable non-semantic metadata differences separately.

If the initialization itself differs, stop before training and document why. Do not proceed with a non-comparable A/B run.

## PPO configuration

Campaign 02 uses:

- gamma: `0.995`;
- GAE lambda: `0.95`;
- clip range: `0.20`;
- value coefficient: `0.50`;
- **entropy coefficient: `0.0`**;
- max gradient norm: `0.50`;
- Adam learning rate: `3e-4`;
- epochs: `2`;
- rollout horizon: `32`;
- CUDA minibatch target: `65,536`.

The entropy coefficient is the only intended Campaign 01 -> Campaign 02 learning change.

Do not change `Rival2PPOConfig` defaults in the frozen v0.5 module. Build the explicit Campaign 02 config at the campaign runner/configuration layer.

## Training scale

Use 131,072 worlds unless the exact Campaign 02 run cannot satisfy the already-demonstrated horizon-32 capacity/integrity requirements. Because Campaign 01 passed at this point, any fallback requires a documented new environmental reason; do not casually choose a smaller batch.

Each update should therefore contribute the same nominal 8,388,608 agent decision samples as Campaign 01. Stop at the first fully completed update whose cumulative sample count is at least 100,000,000.

Expected stop remains update 12 / 100,663,296 samples if all other campaign mechanics are unchanged.

## Checkpoints

Preserve full resumable checkpoints at:

- initialization;
- first completed update crossing 10M;
- first completed update crossing 25M;
- first completed update crossing 50M;
- first completed update crossing 100M.

Store Campaign 02 separately from Campaign 01.

Checkpoint validation must retain the v0.5 exact-continuation requirements: weights, optimizer state, counters, config identities, RNG states, historical-policy metadata/weights, deterministic inference, and next stochastic sample.

## Evaluation

Evaluate each snapshot with the same 4,096-world held-out protocol used by Campaign 01:

- all five standard kickoff layouts;
- first completed episode per world;
- ordinary stochastic self-play;
- side-balanced deterministic play against frozen initialization;
- side-balanced stochastic play against frozen initialization.

Use the same evaluation seed `920260826`.

Record at minimum:

- touches/minute;
- goals/minute;
- goal-terminated fraction;
- no-touch truncation fraction;
- episode duration;
- goal/touch differentials against initialization;
- wins/losses/draws against initialization;
- mean absolute analog actions;
- mean analog policy standard deviations per channel;
- button activation probabilities/rates;
- button entropy.

## Per-update diagnostics

For all optimizer updates record:

- policy loss;
- value loss;
- diagnostic entropy;
- total loss;
- approximate KL;
- clip fraction;
- gradient norm before/after clipping;
- mean or representative analog `log_std` / standard deviation;
- finite-state/integrity gates;
- hot-path H2D/D2H bytes;
- policy/sample version age.

Explicitly flag any update with approximate KL >= `0.1` or clip fraction >= `0.3` for diagnosis, but do not introduce automatic KL stopping in this campaign. Non-finite state remains a hard stop.

## Behavioral classification

Execution correctness and behavioral outcome remain separate.

At closeout, classify Campaign 02 as one of:

- `IMPROVED`: at least two of these three primary final metrics improve versus initialization — ordinary stochastic self-play touches/minute, stochastic-vs-initialization goal differential, stochastic-vs-initialization touch differential — and Campaign 02 is not worse than Campaign 01 on any of the same three;
- `DEGRADED`: at least two of the three primary final metrics are worse than initialization;
- `INCONCLUSIVE`: neither condition is met.

Also report deterministic-vs-initialization results, no-touch fraction, analog standard-deviation trajectory, and KL/clip behavior as secondary diagnostics. Do not rewrite the classification after seeing results.

## Hard exclusions

Campaign 02 must not:

- alter reward terms or weights;
- alter observation/action/episode contracts;
- add action masks;
- change the actor/critic architecture;
- change tanh-Gaussian/Bernoulli action semantics;
- tune learning rate, epochs, clip range, horizon, gamma, lambda, or minibatch size as an optimization experiment;
- add curricula;
- add imitation/expert data;
- begin v0.6 transfer work.
