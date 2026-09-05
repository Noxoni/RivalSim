# Update 500: acquisition coverage returns to baseline; no sustained competitive progress

Evaluation completed 2026-09-05 17:46:19 UTC at 2,949,120,000 accepted trainable
samples. Same frozen 64 original episodes per deterministic case, maximum 30
seconds. The evaluator completed and the learner resumed without intervention.

| Metric | Initial | Update 450 | Update 500 |
| --- | ---: | ---: | ---: |
| Acquisition touch coverage | 12/64 (18.75%) | 16/64 (25%) | 12/64 (18.75%) |
| Acquisition native contacts | 17 | 16 | 13 |
| Acquisition contacts/min | 0.97275 | 0.99512 | 0.81852 |
| Median first touch, touched acquisition cases only | 3.500 s | 0.900 s | 0.875 s |
| Acquisition goals scored / conceded | 0 / 0 | 1 / 0 | 3 / 0 |
| Acquisition no-touch truncations | 63/64 | 63/64 | 61/64 |
| Finishing touch coverage | 24/64 (37.5%) | 40/64 (62.5%) | 40/64 (62.5%) |
| Finishing native contacts | 67 | 45 | 44 |
| Finishing goals scored / conceded | 3 / 0 | 15 / 0 | 13 / 0 |
| Finishing no-touch truncations | 58/64 | 49/64 | 51/64 |
| Nexto kickoff-start touch coverage | 0/64 | 51/64 (79.6875%) | 39/64 (60.9375%) |
| Nexto kickoff-start Rival contacts | 0 | 77 | 52 |
| Nexto kickoff-start goals scored / conceded | 0 / 64 | 0 / 64 | 0 / 64 |
| Nexto first-touch median, touched cases only | unavailable | 2.342 s | 2.342 s |
| Nexto kickoff-start no-touch truncations | 0/64 | 0/64 | 0/64 |

## Interpretation

Three acquisition goals are above the initial zero, but contact coverage has
returned to initialization and the contact rate/count are below it. The faster
conditional first-touch median describes only the twelve contacting cases.
It must not be presented as improved time to reach the ball across all 64.
No-touch truncation means 15 seconds since the last contact and can occur after
a touch; 61/64 acquisition episodes still end this way.

Finishing remains better than initialization, but goals are in the same 13-16
range observed since early training, with two fewer than at 450. Nexto touch
coverage declined again and all 64 original episodes ended in a goal against.
The 13 goals for Rival at update 300 have not repeated at 350, 400, 450 or 500.
These fixed scenario outcomes are not match or kickoff win rates. Positive
ball-velocity sign at contact is not shot quality or possession, and no named
mechanics are inferred. At approximately 2.95 billion samples, these results
do not establish reliable acquisition or sustained competitive improvement.

## Integrity and continuation

Frozen source/authority/package/preflight and initial checkpoint hashes verified.
CPU checkpoint audit at 500 passed: model/Adam finite, exact fresh lineage,
changed weights and real optimizer steps, cadence/LRs preserved, KL telemetry
only, numerical rollback intact. The audit was taken while evaluation was
running and therefore embeds the then-latest update-450 evaluation; the completed
500 evaluation is separately preserved here. This timing is not a mismatch of
checkpoint or model identity. Worker PID 35748 remained active, campaign stderr
empty, with no failure record or STOP marker, and returned to rollout afterward.

Initial-control and numerical investigations remain in Git, as does the blocked
CPU full-trajectory diagnostic. No unsupported path-level diagnosis is claimed,
no CPU compilation retry repeated, and no competing GPU workload launched.
No reward, exploration, architecture, optimizer, curriculum or training source
was changed. Routine acquisition remains below its frozen transition criterion:
training is still pure current self-play; Nexto remains evaluation-only.
Continue per the user's until-stopped instruction. Next evaluation: update 550.

Permanent checkpoint: `checkpoints/rival2/fresh_ground_30hz_v1/u000500.pt`.
SHA-256: `C6B889F5CC773C035EA803EA092AF19F3D38A72DBA2C1C8E9469C08C0058527E`.
Matches the identity bound to the completed evaluation. The checkpoint, stable
curve and CPU audit were already pushed in
`ca541fe3154613f3a6f69f8f5c923bd01908dc63` during evaluation.
Full evaluation: `../evaluations/u000500.json`.
Stable accepted curve/audit: `curve_through_u000500.json`, `u000500.json`.
