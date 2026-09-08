# Update 110: varied acquisition improves, easy acquisition regresses

Scheduled deterministic evaluation on the frozen 1,024 Nexto starts. This audit
is CPU-only; no additional GPU evaluation, interruption or configuration change.

| Update | Easy own contact within 5 s | Varied own contact within 8 s |
| --- | ---: | ---: |
| Untrained | 31/512 (6.0546875%) | 11/512 (2.1484375%) |
| 20 | 247/512 (48.2421875%) | 48/512 (9.375%) |
| 90 | 76/512 (14.84375%) | 97/512 (18.9453125%) |
| 100 | 93/512 (18.1640625%) | 93/512 (18.1640625%) |
| 110 | 68/512 (13.28125%) | 115/512 (22.4609375%) |

Varied improves by 22 successes versus update 100, setting a new high for this
run. Easy loses 25 successes and remains well below its earlier peak. Combined
successes fall from 186 to 183, although the two cohorts have different deadlines.
This is mixed progress, not evidence of general gameplay improvement or promotion.

Conditional successful-contact lower medians: 1.1 s easy and
1.8333333333333333 s varied. Full-8-second no-own-contact fractions are
0.845703125 and 0.775390625. Easy 5-second success and full-8-second no-contact
are different quantities. The probe measures own contact in the original
episode, not beating Nexto to first touch, maintaining possession or winning.

Training updates 101-110: learner first awards 17,698, reward
17692.923563539982; off-ground awards 5,979, reward 1494.3221829384565.
Eligible learner off-ground contacts: 5,981; two are unpaid under the frozen
episode budget. Capped agent-decisions: 227, not a count of players or unpaid
contacts. Excluded Nexto first awards: 15,661; off-ground awards: 38,322.
Repeated contacts without another first award: 7,824. All six families remain
exposed. Low bounces count as off-ground; these counters do not prove aerial
offense or named mechanics.

Nexto training worlds: 4 Rival goals, 15,999 opponent goals, 9,269 Rival touches
over 491519.999995792 world-seconds, and 5,494 inactivity resets. Previous
updates 91-100 had 7/17,819 goals, 10,218 touches and 5,138 inactivity resets.
Mean learner speed: 487.99755601671006 uu/s, versus 498.2283802173755.
These are stochastic training scenario outcomes, not match scores or win rates.
Rival scoring remains extremely weak; declining concessions alone do not prove
better defense. Touches declined and inactivity resets increased in this window.

All 17 CPU report checks passed, including finite model/Adam, lineage/contracts,
exact update/sample/physics/Adam counters, no KL rejection, every-family
exposure, contact accounting, and read-only evaluation/checkpoint identity.
Independently recomputed raw tick counts/fractions, lower medians and no-contact
fractions; checkpoint matches start receipt and completed probe; scenario/spec
identities match the baseline. All 42 frozen source LF-normalized hashes match.
No numerical, corruption or contact-accounting failure was found.

Update 110: 486,604,800 learner decisions and 15,688 Adam steps. Completed-update
mean KL: 0.00423980987855396; maximum sample KL: 0.27717357873916626;
entropy: 3.925894021987915. KL remains telemetry only. Integrity PASS is not
behavioral PASS.

Checkpoint: `checkpoints/rival2/fresh_sustained_contact_v1/plus_000110.pt`

SHA-256: `805AE7A4ED6297D5F7D764451A2E293B561738EDBB781C266385C4FDF3A8D4C1`

Raw probe SHA-256: `EA37D6071948A7862AFC8BCDDC5504C4F623B9F62AF2F1138DFB0C74079197BD`

Raw probe/start receipt, audit and closed training prefix are preserved. The
frozen bounded run continues unchanged: no reward retuning, restart, new guard,
extra evaluation or promotion.
