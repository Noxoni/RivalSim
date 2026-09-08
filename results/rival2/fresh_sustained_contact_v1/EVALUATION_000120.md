# Update 120: varied acquisition gives back its last gain

Scheduled deterministic evaluation on the frozen 1,024 Nexto starts. Publication
audit is CPU-only: no extra GPU evaluation, interruption or configuration change.

| Update | Easy own contact within 5 s | Varied own contact within 8 s |
| --- | ---: | ---: |
| Untrained | 31/512 (6.0546875%) | 11/512 (2.1484375%) |
| 20 | 247/512 (48.2421875%) | 48/512 (9.375%) |
| 100 | 93/512 (18.1640625%) | 93/512 (18.1640625%) |
| 110 | 68/512 (13.28125%) | 115/512 (22.4609375%) |
| 120 | 71/512 (13.8671875%) | 89/512 (17.3828125%) |

Easy gains 3 successes versus update 110; varied loses 26, giving back its last
gain. Both cohorts are below update 100. This is not sustained progress or a
general gameplay improvement. No promotion is claimed.

Conditional successful-contact lower medians: 1.05 s easy and
2.0083333333333333 s varied. Full-8-second no-own-contact fractions:
0.841796875 and 0.826171875. Easy 5-second success and full-8-second no-contact
are different quantities. Own contact is measured in the original episode;
this does not establish beating Nexto to first touch, possession or winning.

Training updates 111-120: learner first awards 17,801, reward
17795.865633547306; off-ground awards 6,434, reward 1608.036428451538.
Eligible learner off-ground contacts: 6,446; twelve are unpaid under the frozen
episode budget. Capped agent-decisions: 2,180, not a count of players or unpaid
contacts. Excluded Nexto first awards: 14,914; off-ground awards: 38,625.
Repeated contacts without another first award: 9,097. All six families exposed.
Low bounces count as off-ground; these are not aerial-mechanic detections.

Nexto training worlds: 6 Rival goals, 15,332 opponent goals, 7,844 Rival touches
over 491519.9999957899 world-seconds, and 5,855 inactivity resets. Previous
updates 101-110 had 4/15,999 goals, 9,269 touches and 5,494 inactivity resets.
Mean learner speed: 466.3193149142795 uu/s, versus 487.99755601671006.
These are stochastic training scenario outcomes, not match scores or win rates.
Scoring remains extremely weak. Touches and movement declined while inactivity
resets increased; declining concessions alone do not demonstrate good defense.

All 17 CPU report checks passed: finite model/Adam, lineage/contracts, exact
update/sample/physics/Adam counters, no KL rejection, all-family exposure,
contact accounting, and read-only evaluation/checkpoint identity. Independently
recomputed raw tick counts/fractions, lower medians and no-contact fractions;
checkpoint matches start receipt and completed probe; scenario/spec hashes match
baseline. All 42 frozen source LF-normalized hashes remain unchanged. No
numerical, corruption or contact-accounting failure was found.

Update 120: 530,841,600 learner decisions and 17,070 Adam steps. Completed-update
mean KL: 0.004353058449810811; maximum sample KL: 0.27867424488067627;
entropy: 3.7912988662719727. KL remains telemetry only. Integrity PASS is not
behavioral PASS.

Checkpoint: `checkpoints/rival2/fresh_sustained_contact_v1/plus_000120.pt`

SHA-256: `BAB0495F3D9275B41FCD960D73B230EAD05897C7A1971E3D69FE2B7B14179C0E`

Raw probe SHA-256: `1B350715F7E9354C75EBA514EFF1BC45CB9B1B0814EEFAAD170F091BC6F1EF29`

Raw probe/start receipt, audit and closed prefix are preserved. The frozen
bounded run continues unchanged; no reward retuning, restart, new guard or
promotion. No inference about aerial offense or SSL capability is supported.
