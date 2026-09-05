# Update 100: numerical progress is not reliable acquisition

Latest newly completed deterministic evaluation: update 100, 2026-09-05
15:19:32 UTC; 589,824,000 accepted trainable samples at that boundary.
Each case uses the same 64 initial scenarios and 30-second maximum protocol.
These are not full-match win rates.

| Metric | Initial | Update 50 | Update 100 |
| --- | ---: | ---: | ---: |
| Acquisition touch coverage | 12/64 (18.75%) | 7/64 (10.9375%) | 10/64 (15.625%) |
| Acquisition contacts/min | 0.97275 | 0.43753 | 0.63375 |
| First-touch median, touched cases only | 3.500 s | 0.675 s | 0.683 s |
| Acquisition no-touch truncations | 63/64 | 62/64 | 61/64 |
| Acquisition goals for / against | 0 / 0 | 1 / 1 | 2 / 1 |
| Finishing touch coverage | 24/64 (37.5%) | 37/64 (57.8125%) | 41/64 (64.0625%) |
| Finishing goals | 3 | 15 | 16 |
| Finishing native contacts | 67 | 40 | 45 |
| Finishing no-touch truncations | 58/64 | 49/64 | 48/64 |
| Nexto kickoff goals scored / conceded | 0 / 64 | 0 / 64 | 0 / 64 |
| Nexto kickoff Rival contacts | 0 | 0 | 0 |

Finishing improved substantially over the initial policy, but only slightly
since update 50. Acquisition partly recovered from 50 and remains below
initial touch coverage/rate. Its conditional first-touch median reflects only
the ten cases that touched; it must not be presented as broad fast acquisition.
No-touch truncation means 15 seconds since the last touch, not necessarily
an episode with zero lifetime contacts. All 64 Nexto kickoff cases still ended
in concessions, without a Rival ball contact. There is no demonstrated broad
ball-pursuit, possession, kickoff, or mechanic competence.

## Focused non-interrupting investigation

No competing GPU work was launched. The frozen code/authority/preflight hashes
were reverified. Live worker PID 35748 remained active. No failure record or
stderr error was present. CPU-only latest-checkpoint audit at update 116:
PASS, 684,195,840 trainable samples, finite model/Adam, real positive optimizer
step counters, changed model tensors, correct fresh lineage/cadence/LRs,
KL telemetry only, preserved nonfinite rollback. No changes were made.

Existing training logs show stochastic contacts/min increasing 0.5359 ->
0.7147 and mean speed 625.03 -> 729.61 between updates 50 and 100. These do
not establish improvement in deterministic pursuit. Sampling diagnostics:

| Existing PPO diagnostic | Update 50 | Update 100 |
| --- | ---: | ---: |
| Mean pre-tanh throttle | -0.00395 | -0.02984 |
| Jump probability | 0.22509 | 0.34527 |
| Boost probability | 0.40643 | 0.36371 |
| Handbrake probability | 0.17512 | 0.29101 |
| Throttle log standard deviation | -0.34852 | -0.31413 |
| Completed-update likelihood KL | 0.002937 | 0.002908 |
| Gradient norm | 0.25329 | 0.25923 |
| Accepted Adam minibatch steps in that update | 182 | 182 |

The learner is not numerically frozen, exploration has not collapsed, and
goals/reset/truncation events are functioning. The logged average throttle
remains near neutral while jump/handbrake sampling has grown. These aggregates
do not show the controls at individual failed acquisition states and cannot
prove causation. They support the narrower conclusion that improved noisy
training contact statistics have not translated into dependable deterministic
ball pursuit. No claim that the reward/architecture/cadence is proven correct
for fast learning follows from a numerical audit.

The lack of broad acquisition improvement after about 590 million samples
at the evaluation boundary is reported explicitly, not hidden behind finishing
gains. The routine-acquisition criterion remains unmet, so Nexto is still
evaluation-only. Per the user's until-stopped authorization, training continues
unchanged with the next evaluation at update 150. No automatic reward,
exploration, curriculum, or optimizer retuning was performed.

Permanent checkpoint:
`checkpoints/rival2/fresh_ground_30hz_v1/u000100.pt`
SHA-256: `75C946A0E2803A75943BE8CE5C1E6AC9FEEDB19B81DFEDBE98AC823C3132838F`.
Full evaluation: `../evaluations/u000100.json`.
Latest numerical audit/accepted curve: `u000116.json`, `curve_through_u000116.json`.
