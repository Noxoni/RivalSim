# Update 300: first clear change in fixed Nexto kickoff-start outcomes

Evaluation completed 2026-09-05 16:36:12 UTC at 1,769,472,000 accepted trainable
samples. Each case uses the same frozen 64 original episodes and at most 30
seconds. The live GPU learner was not interrupted or changed.

| Metric | Initial | Update 250 | Update 300 |
| --- | ---: | ---: | ---: |
| Acquisition touch coverage | 12/64 (18.75%) | 10/64 (15.625%) | 11/64 (17.1875%) |
| Acquisition native contacts | 17 | 10 | 13 |
| Acquisition contacts/min | 0.97275 | 0.62240 | 0.79252 |
| First-touch median, touched acquisition cases only | 3.500 s | 0.750 s | 0.742 s |
| Acquisition goals scored / conceded | 0 / 0 | 0 / 1 | 1 / 0 |
| Acquisition no-touch truncations | 63/64 | 63/64 | 63/64 |
| Finishing touch coverage | 24/64 (37.5%) | 43/64 (67.1875%) | 43/64 (67.1875%) |
| Finishing native contacts | 67 | 46 | 47 |
| Finishing goals scored / conceded | 3 / 0 | 13 / 0 | 15 / 0 |
| Finishing no-touch truncations | 58/64 | 51/64 | 49/64 |
| Nexto kickoff-start touch coverage | 0/64 | 0/64 | 51/64 (79.6875%) |
| Nexto kickoff-start Rival contacts | 0 | 0 | 51 |
| Nexto kickoff-start goals scored / conceded | 0 / 64 | 0 / 64 | 13 / 51 |
| Nexto first-touch median, touched cases only | unavailable | unavailable | 2.342 s |
| Nexto kickoff-start no-touch truncations | 0/64 | 0/64 | 0/64 |

## Interpretation

This is the first completed evaluation in the run with Rival contact or goals
for its side in the fixed Nexto kickoff-start corpus. Those are substantive
changes from every previous checkpoint, not merely finite optimizer progress.
They remain **scenario outcomes, not match win rates, kickoff win rates, or
proof of possession**. Contacts can occur after the opening contest. The test
has fixed layouts/seeds; one evaluation does not establish broad Nexto strength.
No new GPU rerun or independent test was performed during this monitor.

General acquisition remains weak: 11/64 cases touched versus 12/64 initially,
with 63/64 ultimately reaching the 15-second no-touch truncation. The conditional
0.742-second first-touch median describes those eleven cases, not the other 53.
A no-touch truncation can follow an earlier contact. Finishing goals recovered
to 15 after 13 at 250, with unchanged coverage. The routine-acquisition criterion
is still unmet, so training remains pure current self-play; Nexto is still
evaluation-only. No opponent transition was forced.

## Additional non-competing CPU initial-action inspection

Ran the existing `benchmarks/inspect_fresh_ground_30hz_initial_actions.py` with
`--updates 250 300`. It uses the production native initial-state observation
builder on CPU, zero initial GRU state, and deterministic actor inference.
It performs zero physics ticks and zero optimizer steps, allocates zero Torch
CUDA bytes, and verifies checkpoint hashes before/after. It does not require
the vehicle CPU kernel that blocked the previous trajectory diagnostic.

| Initial-state summary | Update 250 | Update 300 |
| --- | ---: | ---: |
| Acquisition mean throttle | 0.19968 | 0.25168 |
| Acquisition boost requested | 62/64 | 62/64 |
| Finishing mean throttle | 0.21439 | 0.25751 |
| Finishing boost requested | 64/64 | 64/64 |
| Kickoff mean throttle | 0.27922 | 0.33483 |
| Kickoff mean steer | 0.20590 | 0.13358 |
| Kickoff boost requested | 64/64 | 64/64 |
| Initial jump / handbrake in all three cases | 0 / 0 | 0 / 0 |

This confirms changed deterministic analog controls while initial boost/jump/
handbrake decisions were unchanged. It does not prove a specific causal reason
for the improved trajectories. Low throttle is not itself low acceleration when
boost is active. The output is CPU recomputation, not GPU bitwise-parity proof.
Full observations, per-state actions/probabilities/stds and hashes are retained
in `initial_action_diagnostic_through_u000300.json`.

The separate full CPU trajectory diagnostic remains blocked as recorded at
250. It was not rerun, and no subsequent path or named mechanic is inferred.

## Integrity and continuation

Frozen source, authority, initial checkpoint and preflight hashes verified.
CPU rolling-checkpoint audit passed at accepted update 315, 1,857,945,600
accepted trainable samples: model/Adam finite, fresh lineage and real optimizer
steps, unchanged cadence/LRs, KL telemetry only, numerical rollback preserved.
Worker PID 35748 remains active, with empty campaign stderr, no failure record
and no STOP marker. No production code, reward, curriculum, exploration,
architecture or optimizer settings were changed.

Permanent checkpoint: `checkpoints/rival2/fresh_ground_30hz_v1/u000300.pt`.
SHA-256: `AB7A75E7E0E8D375F8BC39211D075F834B0FF794B8FDA95846CF9F2D07C72BF2`.
Its hash matches the checkpoint bound to the completed evaluation.
Full evaluation: `../evaluations/u000300.json`.
Stable accepted curve/audit: `curve_through_u000315.json`, `u000315.json`.
Training continues until the user stops it; next evaluation is update 350.
