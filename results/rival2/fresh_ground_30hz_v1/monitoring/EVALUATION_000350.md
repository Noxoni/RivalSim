# Update 350: acquisition coverage rises; Nexto scoring improvement did not hold

Evaluation completed 2026-09-05 16:53:42 UTC at 2,064,384,000 accepted trainable
samples. Each deterministic case uses the same frozen 64 original episodes,
up to 30 seconds. The healthy learner was not interrupted.

| Metric | Initial | Update 300 | Update 350 |
| --- | ---: | ---: | ---: |
| Acquisition touch coverage | 12/64 (18.75%) | 11/64 (17.1875%) | 15/64 (23.4375%) |
| Acquisition native contacts | 17 | 13 | 16 |
| Acquisition contacts/min | 0.97275 | 0.79252 | 0.99678 |
| Median first touch, touched acquisition cases only | 3.500 s | 0.742 s | 0.767 s |
| Acquisition goals scored / conceded | 0 / 0 | 1 / 0 | 2 / 0 |
| Acquisition no-touch truncations | 63/64 | 63/64 | 62/64 |
| Finishing touch coverage | 24/64 (37.5%) | 43/64 (67.1875%) | 45/64 (70.3125%) |
| Finishing native contacts | 67 | 47 | 52 |
| Finishing goals scored / conceded | 3 / 0 | 15 / 0 | 16 / 0 |
| Finishing no-touch truncations | 58/64 | 49/64 | 47/64 |
| Nexto kickoff-start touch coverage | 0/64 | 51/64 (79.6875%) | 51/64 (79.6875%) |
| Nexto kickoff-start Rival contacts | 0 | 51 | 64 |
| Nexto kickoff-start goals scored / conceded | 0 / 64 | 13 / 51 | 0 / 64 |
| Nexto first-touch median, touched cases only | unavailable | 2.342 s | 2.342 s |
| Nexto kickoff-start no-touch truncations | 0/64 | 0/64 | 0/64 |

## Interpretation

Acquisition coverage is now above initialization and update 300, but still only
15/64 cases, with 62/64 episodes eventually reaching the 15-second no-touch
timeout. Total acquisition contacts remain below the initial count of 17;
the slightly higher per-minute rate uses the measured exposure denominator.
The 0.767-second median is conditional on contact and does not describe the
49 cases without a touch. A no-touch timeout may follow an earlier contact.
Finishing coverage and goals increased slightly from 300.

The newly learned ability to obtain contacts in the fixed Nexto kickoff-start
corpus persisted: 51/64 cases obtain contact, with more total contacts. However,
**the update-300 scoring gain did not persist**: 13/51 for/against became 0/64.
This is mixed progress, not steady improvement in competitive gameplay. These
fixed scenario outcomes are not full-match win rates or kickoff win rates;
contacts may occur after the opening contest. More contacts, and a positive
ball-velocity sign on some of them, do not prove possession or useful shots.
No named mechanics or specific downstream movement failure is inferred.

The previous CPU full-trajectory diagnostic remains blocked by the recorded
vehicle-kernel compilation failure. It was not rerun. No new GPU evaluation or
benchmark competed with training. This monitoring pass used completed production
results plus a CPU checkpoint audit, not newly constructed capability gates.

## Integrity and continuation

Source/authority/package/preflight and initial checkpoint hashes verified.
CPU audit at accepted update 353 passed: 2,082,078,720 accepted trainable samples,
finite model/Adam, changed weights and real steps, exact fresh lineage/cadence/
learning rates, KL telemetry only, and preserved numerical rollback. Worker
PID 35748 active, campaign stderr empty, no failure record or STOP marker.
Numerical health is separate from gameplay quality.

The routine-acquisition criterion remains unmet, so training remains pure
current self-play and Nexto is evaluation-only. No reward, exploration,
curriculum, architecture, optimizer or training source was changed. Both 300
and 350 permanent snapshots remain available; neither is overwritten or
silently substituted as a new training parent. Continue under the user's
until-stopped instruction; next scheduled evaluation is update 400.

Permanent checkpoint: `checkpoints/rival2/fresh_ground_30hz_v1/u000350.pt`.
SHA-256: `4244C688C27A711389B0C996969959FBC50E4048D77BAEDCE04F236DBA042C46`.
The hash matches the checkpoint bound to the completed evaluation.
Full evaluation: `../evaluations/u000350.json`.
Stable accepted curve/audit: `curve_through_u000353.json`, `u000353.json`.
