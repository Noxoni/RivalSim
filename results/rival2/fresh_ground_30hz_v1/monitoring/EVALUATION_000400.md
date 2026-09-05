# Update 400: no sustained competitive gain; Nexto contact coverage declines

Evaluation completed 2026-09-05 17:11:06 UTC at 2,359,296,000 accepted trainable
samples. Same frozen seeds, 64 original episodes per deterministic case, at
most 30 seconds. These are scenario outcomes, not full-match win rates.

| Metric | Initial | Update 350 | Update 400 |
| --- | ---: | ---: | ---: |
| Acquisition touch coverage | 12/64 (18.75%) | 15/64 (23.4375%) | 14/64 (21.875%) |
| Acquisition native contacts | 17 | 16 | 16 |
| Acquisition contacts/min | 0.97275 | 0.99678 | 0.98296 |
| Median first touch, touched acquisition cases only | 3.500 s | 0.767 s | 0.767 s |
| Acquisition goals scored / conceded | 0 / 0 | 2 / 0 | 0 / 1 |
| Acquisition no-touch truncations | 63/64 | 62/64 | 63/64 |
| Finishing touch coverage | 24/64 (37.5%) | 45/64 (70.3125%) | 47/64 (73.4375%) |
| Finishing native contacts | 67 | 52 | 54 |
| Finishing goals scored / conceded | 3 / 0 | 16 / 0 | 15 / 0 |
| Finishing no-touch truncations | 58/64 | 47/64 | 48/64 |
| Nexto kickoff-start touch coverage | 0/64 | 51/64 (79.6875%) | 39/64 (60.9375%) |
| Nexto kickoff-start Rival contacts | 0 | 64 | 39 |
| Nexto kickoff-start goals scored / conceded | 0 / 64 | 0 / 64 | 0 / 64 |
| Nexto first-touch median, touched cases only | unavailable | 2.342 s | 2.342 s |
| Nexto kickoff-start no-touch truncations | 0/64 | 0/64 | 0/64 |

## Interpretation

The update-300 result of 13 goals for and 51 against Nexto has not repeated at
350 or 400. Contact acquisition against Nexto is still above initialization,
but both contact coverage and count fell this interval. Every original Nexto
episode ended with a goal against Rival. No kickoff win rate, possession,
on-target shot, or named-mechanic claim is supported by these aggregate fields.
The reported 100% goalward-contact fraction at 400 is the end-decision canonical
ball-velocity sign, not successful offense; it coexists with zero goals for.

General acquisition coverage is only two cases above initialization and one
case below 350. Its contact rate is approximately the initial rate. The fast
conditional first-touch median applies to the fourteen contacting cases, not
the 50 without contact. 63/64 episodes ultimately hit the 15-second inactivity
limit, which can occur after an earlier touch. Finishing contact coverage is
better than initialization, but goals remain around the 13-16 range seen since
50 rather than steadily rising. This is not evidence that broad gameplay is
solved or that sample accumulation itself ensures continued improvement.

Prior focused investigations are retained: finite optimizer/changed weights,
CPU deterministic initial controls, and the blocked CPU full-trajectory probe.
The vehicle CPU compilation limitation remains unresolved; no downstream path
failure (overshoot, circling, abandonment) is asserted without trajectory data.
No competing GPU diagnostic was launched or new capability criterion introduced.

## Integrity and continuation

Frozen training source/authority/package/preflight and initial checkpoint hashes
verified. CPU audit at accepted update 421 passed, with 2,483,159,040 accepted
trainable samples, finite model/Adam, real steps and changed weights, exact fresh
lineage/cadence/LRs, KL telemetry only and numerical rollback preserved. Worker
PID 35748 active; campaign stderr empty; no failure or STOP file. A sample KL
magnitude is telemetry, not a guard failure under this authority.

Routine acquisition remains below the frozen transition requirement. Opponents
remain current self-play only, Nexto evaluation-only. No reward, exploration,
optimizer, curriculum, model architecture, or training source was changed.
The user authorized continuation until stopped; these weak and unstable gameplay
results are reported explicitly rather than hidden by numerical-health PASS.
Next scheduled evaluation: update 450.

Permanent checkpoint: `checkpoints/rival2/fresh_ground_30hz_v1/u000400.pt`.
SHA-256: `33ED5228F173DED7E23F3DD2617ACC77459BE762AB52E75FD7B46B294A17022B`.
Its hash matches the evaluated checkpoint identity.
Full evaluation: `../evaluations/u000400.json`.
Stable accepted curve/audit: `curve_through_u000421.json`, `u000421.json`.
