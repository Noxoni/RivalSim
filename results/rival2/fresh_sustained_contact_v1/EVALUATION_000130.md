# Update 130: small easy acquisition gain, varied unchanged in practice

Scheduled deterministic evaluation on the frozen 1,024 Nexto starts. Publication
audit is CPU-only; no extra GPU evaluation, interruption or configuration change.

| Update | Easy own contact within 5 s | Varied own contact within 8 s |
| --- | ---: | ---: |
| Untrained | 31/512 (6.0546875%) | 11/512 (2.1484375%) |
| 20 | 247/512 (48.2421875%) | 48/512 (9.375%) |
| 100 | 93/512 (18.1640625%) | 93/512 (18.1640625%) |
| 110 | 68/512 (13.28125%) | 115/512 (22.4609375%) |
| 120 | 71/512 (13.8671875%) | 89/512 (17.3828125%) |
| 130 | 78/512 (15.234375%) | 88/512 (17.1875%) |

Easy gains 7 successes versus update 120; varied loses 1. Both remain below
update 100. This modest easy gain does not establish broad, sustained progress
or general gameplay improvement. No promotion is claimed.

Conditional successful-contact lower medians: 0.925 s easy and 1.95 s varied.
Full-8-second no-own-contact fractions: 0.833984375 and 0.828125. Easy 5-second
success and full-8-second no-contact are different quantities. The probe measures
own contact in the original episode, not first-to-ball advantage, possession,
scoring or winning.

Training updates 121-130: learner first awards 17,867, reward 17861.87294614315;
off-ground awards 7,309, reward 1826.7245417684317. Eligible learner off-ground
contacts: 7,339; thirty unpaid under the frozen episode budget. Capped
agent-decisions: 6,185, not a count of players or unpaid contacts. Excluded
Nexto first awards: 13,233; off-ground awards: 37,498. Repeated contacts without
another first award: 11,504. All six families exposed. Low bounces count as
off-ground; these awards do not demonstrate aerial mechanics or aerial offense.

Nexto training worlds: 6 Rival goals, 13,504 opponent goals, 7,412 Rival touches
over 491519.9999958 world-seconds, and 6,613 inactivity resets. Previous updates
111-120 had 6/15,332 goals, 7,844 touches and 5,855 inactivity resets. Mean
learner speed: 445.35302612304685 uu/s, versus 466.3193149142795. These are
stochastic training scenario outcomes, not match scores or win rates. Scoring
remains extremely weak; Nexto-world touches and overall learner movement
declined while inactivity resets increased. Declining concessions alone do not
demonstrate improved defense. Contact awards rising across the full training
mixture do not establish improved competitive gameplay.

All 17 CPU report checks passed: finite model/Adam, lineage/contracts, exact
update/sample/physics/Adam counters, no KL rejection, all-family exposure,
contact accounting and read-only evaluation/checkpoint identity. Independently
recomputed raw tick counts/fractions, lower medians and no-contact fractions;
checkpoint matches start receipt and completed probe; scenario/spec hashes match
baseline. All 42 frozen source LF-normalized hashes remain unchanged. No
numerical, corruption or contact-accounting failure was found.

Update 130: 575,078,400 learner decisions and 18,450 Adam steps. Completed-update
mean KL: 0.005382351660543665; maximum sample KL: 0.4285850524902344;
entropy: 3.6700539588928223. KL remains telemetry only. Integrity PASS is not
behavioral PASS.

Checkpoint: `checkpoints/rival2/fresh_sustained_contact_v1/plus_000130.pt`

SHA-256: `A69D9ED4F02E51B90755382873A9B89DD7CDBB4A3E366D6897974DF801BBF26C`

Raw probe SHA-256: `5136D71CFC7E3F5F51A150782922EC26F506DD70F5EFC00B111175A16EB6EF52`

Raw probe/start receipt, audit and closed prefix are preserved. The frozen
bounded run continues unchanged; no retuning, restart, new guard or promotion.
