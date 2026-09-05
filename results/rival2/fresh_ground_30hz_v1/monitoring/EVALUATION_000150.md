# Update 150: easier-contact coverage grows, broad gameplay remains poor

Evaluation completed 2026-09-05 15:38:30 UTC, at 884,736,000 trainable samples.
The live learner was not interrupted. Each deterministic case has 64 original
episodes, fixed seeds, and a maximum of 30 seconds; these are not match win rates.

| Metric | Initial | Update 100 | Update 150 |
| --- | ---: | ---: | ---: |
| Acquisition touch coverage | 12/64 (18.75%) | 10/64 (15.625%) | 11/64 (17.1875%) |
| Acquisition contact rate/min | 0.97275 | 0.63375 | 0.68807 |
| First-touch median, touched cases only | 3.500 s | 0.683 s | 0.683 s |
| Acquisition goals scored / conceded | 0 / 0 | 2 / 1 | 1 / 0 |
| Acquisition no-touch truncations | 63/64 | 61/64 | 63/64 |
| Finishing touch coverage | 24/64 (37.5%) | 41/64 (64.0625%) | 46/64 (71.875%) |
| Finishing goals scored / conceded | 3 / 0 | 16 / 0 | 15 / 1 |
| Finishing native contacts | 67 | 45 | 52 |
| Finishing no-touch truncations | 58/64 | 48/64 | 48/64 |
| Nexto kickoff goals scored / conceded | 0 / 64 | 0 / 64 | 0 / 64 |
| Nexto kickoff Rival contacts | 0 | 0 | 0 |

More finishing setups obtain a touch, but goals have not increased since 100.
Acquisition remains below initial contact coverage/rate and predominantly ends
in no-touch timeouts. The short first-touch median applies to the eleven
contacting cases, not all 64. A timeout is 15 seconds since the last contact;
it does not mean the episode necessarily had no contact at all. There is still
no Nexto competitiveness, reliable broad pursuit, or established possession or
mechanic competence. Continued numerical updates alone do not meet that objective.

## Additional CPU-only investigation of initial controls

To go beyond aggregate gradient checks, a new read-only diagnostic reconstructed
the **same frozen initial scenario states** with `Rival2WorldSim(device='cpu')`
and the production `Rival2TensorBridge.observation()` function. It evaluated
initial,50,100,150 checkpoint copies deterministically with zero initial GRU
state. It performed **zero physics ticks and zero optimizer steps**, allocated
zero Torch CUDA bytes, and verified every source-checkpoint SHA before/after.
No GPU rollout competed with training. This is CPU recomputation of the native
observation/policy path, **not a claim of bitwise parity to a captured GPU forward**.

Script: `benchmarks/inspect_fresh_ground_30hz_initial_actions.py`.
Full initial observations, per-state actions, button probabilities, analog
standard deviations, scenario/checkpoint hashes, and summaries:
`initial_action_diagnostic_through_u000150.json`.

| Initial-state deterministic control, update 150 | Acquisition | Finishing | Kickoff |
| --- | ---: | ---: | ---: |
| Mean throttle | 0.09159 | 0.08957 | 0.15071 |
| Throttle range | 0.03783..0.16231 | 0.02750..0.15160 | 0.12625..0.16891 |
| Boost on | 59/64 | 64/64 | 64/64 |
| Jump on | 0/64 | 0/64 | 0/64 |
| Handbrake on | 0/64 | 0/64 | 0/64 |
| Mean jump probability | 0.02158 | 0.00177 | 0.00313 |
| Mean boost probability | 0.86251 | 0.98078 | 0.97109 |
| Mean handbrake probability | 0.06053 | 0.03492 | 0.06128 |

This rules out the simple account that initial deterministic behavior refuses
boost or begins by jumping/handbraking in these cases. It also clarifies the
earlier aggregate-rollout observation: rising mean jump/handbrake probabilities
over naturally visited training states do **not** describe these initial states.
Low throttle alone cannot establish low acceleration while boost is active.
The first-action inspection does not establish what happens later in failed
trajectories; poor acquisition can still result from steering, timing, or later
decisions. No isolated causal failure or justified operational repair was found.

## Integrity and continuation

Frozen source/authority/preflight identities reverified; no learning-source
changes. Latest CPU-only checkpoint audit in this monitor: update 165,
973,209,600 trainable samples, PASS. Model and Adam finite, real steps and changed
weights, exact fresh lineage/cadence/learning rates, KL telemetry only, numerical
rollback intact. Worker PID 35748 active; stderr empty; no failure record.

The routine-acquisition criterion remains unmet; training is still pure current
self-play with Nexto evaluation-only. No rewards, exploration, optimizer,
curriculum, architecture, detectors, or training parameters were changed.
Continue per the user's until-stopped instruction; next evaluation update 200.
The persistent lack of broad acquisition/Nexto competence is reported explicitly.

Permanent update-150 checkpoint:
`checkpoints/rival2/fresh_ground_30hz_v1/u000150.pt`.
SHA-256: `FE5550ECB669B5B0325F0A665959FACDFF8D712D2B131669A59D0F4A1FA08D4C`.
Full evaluation: `../evaluations/u000150.json`.
Latest accepted-curve/CPU integrity evidence: `curve_through_u000165.json`, `u000165.json`.
