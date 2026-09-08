# Update 140: both acquisition cohorts improve, scoring still very weak

Scheduled deterministic evaluation on the frozen 1,024 Nexto starts. Publication
audit is CPU-only; no extra GPU evaluation, interruption or configuration change.

| Update | Easy own contact within 5 s | Varied own contact within 8 s |
| --- | ---: | ---: |
| Untrained | 31/512 (6.0546875%) | 11/512 (2.1484375%) |
| 20 | 247/512 (48.2421875%) | 48/512 (9.375%) |
| 100 | 93/512 (18.1640625%) | 93/512 (18.1640625%) |
| 110 | 68/512 (13.28125%) | 115/512 (22.4609375%) |
| 130 | 78/512 (15.234375%) | 88/512 (17.1875%) |
| 140 | 97/512 (18.9453125%) | 110/512 (21.484375%) |

Easy gains 19 successes and varied gains 22 versus update 130. Both exceed
update 100. This is a positive acquisition boundary, but easy remains far below
its early 247-success peak and varied remains below its 115-success peak. Prior
fluctuations and weak scoring do not support a general gameplay breakthrough
or promotion.

Conditional successful-contact lower medians: 0.9416666666666667 s easy and
1.9 s varied. Full-8-second no-own-contact fractions: 0.8046875 and 0.78515625.
Easy 5-second success and full-8-second no-contact are different quantities. The
probe measures own contact in the original episode, not first-to-ball advantage,
possession, scoring, winning or aerial offense.

Training updates 131-140: learner first awards 17,556, reward 17550.97688895464;
off-ground awards 8,841, reward 2209.614387586713. Eligible learner off-ground
contacts: 8,878; thirty-seven unpaid under the frozen episode budget. Capped
agent-decisions: 9,717, not a count of players or unpaid contacts. Excluded
Nexto first awards: 12,273; off-ground awards: 36,759. Repeated contacts without
another first award: 16,644. All six families exposed. Low bounces count as
off-ground; rising counts do not establish aerial mechanics.

Nexto training worlds: 7 Rival goals, 12,552 opponent goals, 7,398 Rival touches
over 491519.9999957919 world-seconds, and 7,171 inactivity resets. Previous
updates 121-130 had 6/13,504 goals, 7,412 touches and 6,613 inactivity resets.
Mean learner speed: 423.88782886646413 uu/s, versus 445.35302612304685. These
are stochastic training scenario outcomes, not match scores or win rates.
Scoring remains extremely weak; Nexto-world touches are almost unchanged,
overall learner movement declined, and inactivity resets increased. Fewer
concessions alone do not establish better defense. Rising off-ground awards
across the full mixture are not evidence of competitive aerial offense.

All 17 CPU report checks passed: finite model/Adam, lineage/contracts, exact
update/sample/physics/Adam counters, no KL rejection, all-family exposure,
contact accounting and read-only evaluation/checkpoint identity. Independently
recomputed raw tick counts/fractions, lower medians and no-contact fractions;
checkpoint matches start receipt and completed probe; scenario/spec hashes match
baseline. All 42 frozen source LF-normalized hashes remain unchanged. No
numerical, corruption or contact-accounting failure was found.

Update 140: 619,315,200 learner decisions and 19,830 Adam steps. Completed-update
mean KL: 0.0045792829216393495; maximum sample KL: 0.5124881267547607;
entropy: 3.601033926010132. KL remains telemetry only. Integrity PASS is not
behavioral PASS.

Checkpoint: `checkpoints/rival2/fresh_sustained_contact_v1/plus_000140.pt`

SHA-256: `3DD98B4282554504EB3D907EE0A97991D37FB4C3F11C42D4C5F9F2DB1E1401E3`

Raw probe SHA-256: `AE7C5C68AEB39B58AC8BDABE64DF6AB0976D9808D63788B66535EE049FE8347E`

Raw probe/start receipt, audit and closed prefix are preserved. The frozen
bounded run continues unchanged toward 150; no retuning, restart, extension,
new guard or promotion.
