# Update 450: contact coverage fluctuates; scoring against Nexto remains absent

Evaluation completed 2026-09-05 17:28:32 UTC at 2,654,208,000 accepted trainable
samples. Same frozen 64 original episodes per deterministic case, at most 30
seconds. These are scenario outcomes, not full-match or kickoff win rates.

| Metric | Initial | Update 400 | Update 450 |
| --- | ---: | ---: | ---: |
| Acquisition touch coverage | 12/64 (18.75%) | 14/64 (21.875%) | 16/64 (25%) |
| Acquisition native contacts | 17 | 16 | 16 |
| Acquisition contacts/min | 0.97275 | 0.98296 | 0.99512 |
| Median first touch, touched acquisition cases only | 3.500 s | 0.767 s | 0.900 s |
| Acquisition goals scored / conceded | 0 / 0 | 0 / 1 | 1 / 0 |
| Acquisition no-touch truncations | 63/64 | 63/64 | 63/64 |
| Finishing touch coverage | 24/64 (37.5%) | 47/64 (73.4375%) | 40/64 (62.5%) |
| Finishing native contacts | 67 | 54 | 45 |
| Finishing goals scored / conceded | 3 / 0 | 15 / 0 | 15 / 0 |
| Finishing no-touch truncations | 58/64 | 48/64 | 49/64 |
| Nexto kickoff-start touch coverage | 0/64 | 39/64 (60.9375%) | 51/64 (79.6875%) |
| Nexto kickoff-start Rival contacts | 0 | 39 | 77 |
| Nexto kickoff-start goals scored / conceded | 0 / 64 | 0 / 64 | 0 / 64 |
| Nexto first-touch median, touched cases only | unavailable | 2.342 s | 2.342 s |
| Nexto kickoff-start no-touch truncations | 0/64 | 0/64 | 0/64 |

## Interpretation

Acquisition reaches two additional cases, but the same total sixteen contacts.
Since sixteen cases touched and there were sixteen focal contacts, each touched
case had only one focal contact. This does not demonstrate sustained control
or reacquisition. Coverage is four cases above initialization while total count
is one lower, and the rate remains approximately initial. The conditional
first-touch median increased to 0.9 seconds; the other 48 cases had no focal
contact. A no-touch truncation denotes fifteen seconds since the last contact,
so the 63 timeouts do not imply 63 episodes with zero contacts.

Nexto touch coverage recovered to its 300/350 level, with more total contacts
than any previous evaluation. Additional contacts can occur in the same case,
but do not prove possession, useful shot quality, or any named mechanic. All
64 episodes ended with a goal against Rival. The update-300 13-goal result has
not repeated at 350, 400 or 450. Finishing goals remain 15 while contact coverage
dropped. There is still no convincing general scoring or competitive progression;
this is mixed contact improvement rather than all-round capability.

Existing investigation evidence remains available: real finite optimizer steps,
changed weights, CPU initial deterministic controls, and the blocked CPU full
trajectory probe. No unmeasured subsequent path failure is asserted. No competing
GPU work was launched, no diagnostic compilation retry was repeated, and no
new mechanic detector or capability gate was added.

## Integrity and continuation

Frozen source, authority, package, initial checkpoint and preflight identities
verified. CPU checkpoint audit passed at accepted update 463: 2,730,885,120
accepted trainable samples, finite model/Adam, real steps and changed weights,
fresh lineage, exact cadence/LRs, KL telemetry only, numerical rollback intact.
Worker PID 35748 active; campaign stderr empty; no failure record or STOP marker.
The differing sample counter in live campaign state can include an in-progress
rollout; the accepted checkpoint counter is used here.

The routine-acquisition requirement is not met. Training remains pure current
self-play, Nexto evaluation-only. No training source, reward, exploration,
optimizer, curriculum or architecture was changed. Continue under the explicit
user until-stopped instruction, without claiming numerical health is learning
success. Next scheduled evaluation: update 500.

Permanent checkpoint: `checkpoints/rival2/fresh_ground_30hz_v1/u000450.pt`.
SHA-256: `76A454C3FA8C79CB86C6E2C6851EE81487A93A4DE4DCB8E3F71F0B3272200435`.
This matches the checkpoint bound to the completed evaluation.
Full evaluation: `../evaluations/u000450.json`.
Stable accepted curve/audit: `curve_through_u000463.json`, `u000463.json`.
