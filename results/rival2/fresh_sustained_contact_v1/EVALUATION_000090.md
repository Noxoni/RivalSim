# Update90: both contact tests improve; varied reaches a new high

Scheduled deterministic evaluation on the frozen1,024 Nexto starts. No extra GPU
evaluation, interruption, training mutation or configuration change in this audit.

| Update | Easy own contact within5s | Varied own contact within8s |
| --- | ---: | ---: |
| Untrained | 31/512 (6.0546875%) | 11/512 (2.1484375%) |
| 20 | 247/512 (48.2421875%) | 48/512 (9.375%) |
| 40 | 203/512 (39.6484375%) | 77/512 (15.0390625%) |
| 80 | 49/512 (9.5703125%) | 64/512 (12.5%) |
| 90 | 76/512 (14.84375%) | 97/512 (18.9453125%) |

Both improve versus80 (+27 easy, +33 varied). Varied is a new best for this run;
easy remains far below its update10/20 peak. This is a positive acquisition
boundary, not proof of consistent overall gameplay improvement or promotion.
Conditional successful-contact lower medians:0.8666666666666667s easy,1.75s varied.
Full8s no-own-contact fractions:0.83984375 and0.810546875. Easy5s success is not
the complement of full8s no-contact. The test measures own contact, not beating
Nexto to first touch, retained possession, scoring or aerial mechanics.

Training81-90: learner first awards17,189/reward17184.007693111897; off-ground
awards5,384/reward1345.6091427654028. There were5,386 eligible learner off-ground
contacts:2 beyond the frozen episode budget were not paid. Capped agent-decisions
424 is not424 contacts/players. Excluded Nexto first18,653 and off-ground44,648
awards. Repeated contacts without another first award7,369. All6families exposed;
off-ground includes low bounces and is not proof of aerial offense.

Nexto training worlds:12 Rival goals,19,108 opponent goals,11,201 Rival touches
over491519.99999592704 world-seconds,5,346 inactivity resets. Previous71-80 had
8/21,322 goals,11,260 touches and4,961 inactivity resets. Mean learner speed
458.3995683684172uu/s versus475.6440993923611. These are stochastic scenario
outcomes, not match scores or win rates. Scoring is still very poor; do not equate
improved acquisition probes or fewer concessions alone with good general play.

All17 CPU report checks passed: model/Adam finite, lineage/contracts, exact
update/sample/physics/Adam counters, no KL rejection, family exposure, bonus
accounting and checkpoint/read-only evaluation identity. Independently recomputed
raw tick counts/fractions, conditional lower medians and no-contact fractions;
actual checkpoint matches started receipt and probe. Scenario/spec match baseline;
frozen source LF-normalized hashes unchanged. No numerical/payment failure found.

Update90:398,131,200 learner decisions,12,902 Adam steps. Completed update mean
KL0.003993570260003994, max sample KL0.18440043926239014,
entropy4.072636127471924; KL is telemetry only. Integrity is not behavioral PASS.

Checkpoint: `checkpoints/rival2/fresh_sustained_contact_v1/plus_000090.pt`

SHA-256: `292F430820412B82066A963ADDBA2D118CCC209A8C82A701BC1010C2A640DAC3`

Raw probe/start receipt, audit and closed prefix preserved. Frozen bounded
training continues; no retuning, restart, new acceptance guard or promotion.
