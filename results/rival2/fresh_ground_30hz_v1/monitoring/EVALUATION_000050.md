# Update 50: finishing improves, acquisition regresses

This evaluation completed during the monitor's publication of update 20. It
supersedes update 20 as the latest completed evaluation. Training was not
interrupted or modified. The next deterministic evaluation is update 100.

| Deterministic metric, 64 initial episodes per case | Random | Update 20 | Update 50 |
| --- | ---: | ---: | ---: |
| Acquisition focal touch coverage | 12/64 | 13/64 | 7/64 |
| Acquisition contact count | 17 | 25 | 7 |
| Acquisition contacts/min | 0.97275 | 1.46972 | 0.43753 |
| Conditional median first contact | 3.500 s | 2.208 s | 0.675 s |
| Acquisition goals for / against | 0 / 0 | 0 / 0 | 1 / 1 |
| Acquisition no-touch truncations | 63/64 | 64/64 | 62/64 |
| Finishing touch coverage | 24/64 | 35/64 | 37/64 |
| Finishing goals | 3 | 8 | 15 |
| Finishing contact count | 67 | 70 | 40 |
| Finishing no-touch truncations | 58/64 | 56/64 | 49/64 |
| Nexto kickoff goals for / conceded | 0 / 64 | 0 / 64 | 0 / 64 |
| Nexto kickoff Rival contacts | 0 | 0 | 0 |

The shorter acquisition first-contact median is **not general faster ball
acquisition**: it describes only the seven cases that got a contact, versus
thirteen previously. Coverage and contact rate regressed below the initial
random baseline. Finishing now scores in more easy scenarios, but Nexto
kickoff performance is unchanged. No broad gameplay/possession/mechanics claim
is justified. These are scenario results, not match win rates.

## Focused read-only investigation

The frozen source/package/preflight hashes still match. The latest accepted
checkpoint audited here was update 51, 300,810,240 trainable samples: model and
Adam finite, real optimizer steps, fresh lineage, correct cadence and LRs,
no KL rejection, no old policy opponent, and no numerical failure. Goals,
truncations and resets are nonzero in training; no recurrence of the former
missing-reward/reset-cadence bug is indicated.

From the existing accepted rollout/optimizer logs (not a competing GPU run):

| Logged statistic | Update 20 | Update 50 |
| --- | ---: | ---: |
| Stochastic training contacts/min | 0.25726 | 0.53589 |
| Mean movement speed | 493.88 | 625.03 |
| Ended player-episode touch fraction | 0.03132 | 0.12389 |
| Mean actor throttle, pre-tanh | -0.03831 | -0.00395 |
| Mean actor jump probability | 0.18586 | 0.22509 |
| Mean actor boost probability | 0.34519 | 0.40643 |
| Mean actor handbrake probability | 0.13647 | 0.17512 |
| Mean log throttle standard deviation | -0.34942 | -0.34852 |
| Completed-update likelihood KL telemetry | 0.003485 | 0.002937 |

Learning/exploration has not numerically frozen. However, improving stochastic
rollout contact statistics are not transferring into reliable deterministic
acquisition. Average throttle stays near neutral and jump/handbrake sampling
probabilities have increased. These aggregate quantities cannot identify the
action on an individual failed acquisition state or establish causation; they
are a reason to watch the acquisition/Nexto tests, not a proof that a reward
coefficient or recurrent architecture is defective. The training distribution
also changes naturally as episodes reset, so single-rollout comparisons are
not a controlled behavioral evaluation.

No further Nexto training was enabled: the prospectively frozen routine-
acquisition criterion remains unmet. No reward, exploration, model, PPO,
curriculum or guard setting was changed. Continue the authorized run, report
the regression explicitly, and inspect the next comparable evaluation rather
than calling finishing-only progress complete gameplay competence.

Preserved checkpoint: `checkpoints/rival2/fresh_ground_30hz_v1/u000050.pt`.
SHA-256: `0B634C3A6367A4A00D819986230901E55B9D64A7DFC24586B51B698DCDDA8F58`.
Full evaluation: `../evaluations/u000050.json`.
CPU audit: `u000051.json`; full accepted curve: `curve_through_u000051.json`.
