# Update100: easy acquisition rises, varied slips slightly

Scheduled deterministic evaluation on the frozen1,024 Nexto starts. This audit
uses CPU only; no extra GPU evaluation, interruption or configuration change.

| Update | Easy own contact within5s | Varied own contact within8s |
| --- | ---: | ---: |
| Untrained | 31/512 (6.0546875%) | 11/512 (2.1484375%) |
| 20 | 247/512 (48.2421875%) | 48/512 (9.375%) |
| 80 | 49/512 (9.5703125%) | 64/512 (12.5%) |
| 90 | 76/512 (14.84375%) | 97/512 (18.9453125%) |
| 100 | 93/512 (18.1640625%) | 93/512 (18.1640625%) |

Easy improves17 successes versus90; varied loses4. Easy has improved at the last
two boundaries but remains far below its earlier peak. Varied is close to its
update90 best, not a new best. No full-gameplay recovery or promotion is claimed.
Conditional successful-contact lower medians:1.05s easy,1.7083333333333333s varied.
Full8s no-own-contact fractions:0.802734375 and0.818359375. Easy5s success and
full8s no-contact are different quantities. The probe measures own contact in
the original episode, not beating Nexto to first touch or maintaining possession.

Training91-100: learner first awards18,852/reward18846.512792289257; off-ground
awards6,319/reward1579.2958117127419. Eligible learner off-ground contacts6,319;
capped agent-decisions900 does not mean900 unpaid contacts or900 players.
Excluded Nexto first17,454 and off-ground41,299 awards. Repeated contacts without
another first award7,924. All6families exposed. Low bounces count as off-ground;
these counters do not establish aerial offense or named mechanics.

Nexto training worlds:7 Rival goals,17,819 opponent goals,10,218 Rival touches
over491519.9999958482 world-seconds,5,138 inactivity resets. Previous81-90 had
12/19,108 goals,11,201 touches,5,346 inactivity resets. Mean learner speed
498.2283802173755uu/s versus458.3995683684172. These are stochastic training
scenario outcomes, not match scores or win rates. Lower concessions alone are
not evidence of good defense; Rival scoring remains extremely weak and own
touches have not improved in this window despite the easy-probe gain.

All17 CPU report checks passed, including finite model/Adam, lineage/contracts,
exact update/sample/physics/Adam counters, no KL rejection, every-family exposure,
contact accounting and read-only evaluation/checkpoint identity. Independently
recomputed raw tick counts/fractions, lower medians and no-contact fractions;
checkpoint matches started receipt and probe; scenario/spec match baseline.
Frozen source LF-normalized hashes unchanged. No numerical/payment failure found.

Update100:442,368,000 learner decisions,14,302 Adam steps. Completed update mean
KL0.004057436080721446, max sample KL0.296164870262146,
entropy3.9838125705718994; KL is telemetry only. Integrity is not behavioral PASS.

Checkpoint: `checkpoints/rival2/fresh_sustained_contact_v1/plus_000100.pt`

SHA-256: `8CEB8189E16B43B958E279320E1778FAEC49F6A88031589F44294A5D26DCC89B`

Raw probe/start receipt, audit and closed prefix preserved. The frozen bounded
run continues unchanged; no reward retuning, restart, new guard or promotion.
