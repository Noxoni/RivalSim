# Update70: partial acquisition rebound, not recovered gameplay

The scheduled deterministic frozen1,024-start Nexto probe completed without
training mutation. No additional GPU evaluation or interruption was performed.

| Update | Easy own contact within5s | Varied own contact within8s |
| --- | ---: | ---: |
| Untrained | 31/512 (6.0546875%) | 11/512 (2.1484375%) |
| 20 | 247/512 (48.2421875%) | 48/512 (9.375%) |
| 40 | 203/512 (39.6484375%) | 77/512 (15.0390625%) |
| 60 | 25/512 (4.8828125%) | 20/512 (3.90625%) |
| 70 | 68/512 (13.28125%) | 52/512 (10.15625%) |

Both acquisition tests rebound versus60 (+43 easy, +32 varied), but neither
recovers its earlier peak. This is fluctuating acquisition, not evidence of
sustained improvement, good possession or scoring. The probe measures own
first contact in the original episode, not beating Nexto to first touch.
Conditional lower-median successful contact times:1.0083333333333333s easy,
2.158333333333333s varied. Full8s no-own-contact fractions:0.84375 and0.8984375.
Easy5s success is not the complement of full8s no-contact.

## Stochastic training61-70

All6families remain exposed. Learner first awards20,296 with discounted reward
20290.127035975456; off-ground awards5,478 with reward1369.1086609512568.
Excluded Nexto first awards22,385 and off-ground48,701. Repeated contacts without
another first payment6,708; air-budget-capped agent-decisions0. Low bounces
qualify as off-ground; these events do not prove aerial mechanics.

Mean learner speed478.4199354835793uu/s versus473.1836117440683 in51-60.
Nexto worlds recorded9 Rival goals,22,931 opponent goals,12,210 Rival touches
over491519.999996014 world-seconds, and5,223 inactivity resets. Compared51-60,
Rival touches decline14,565 to12,210 and inactivity resets increase186 to5,223.
Fewer opponent goals must not be interpreted as better defense or gameplay from
these data. These are training scenario outcomes, not match scores/win rates.
The acquisition rebound does not establish functional overall play.

## Integrity

All17 CPU report checks passed, including lineage/contracts, finite model/Adam,
exact update/sample/physics/Adam counters, no KL rejection, all-family exposure,
contact reward accounting and read-only evaluation/checkpoint identity.
Independently recomputed raw contact counts/fractions, conditional lower medians
and full8s no-contact fractions; actual checkpoint hash matches the started
receipt and probe. Probe/spec identities match baseline. Frozen sources match
the authority's LF-normalized text hashes. No numerical, corruption or payment
failure was found; this does not diagnose the fluctuating behavior.

Update70:309,657,600 trainable decisions,10,102 Adam steps. Completed-update mean
KL0.004194262464173975, maximum sample KL0.23221933841705322,
entropy4.157422065734863; KL telemetry only, no KL rejection.

Checkpoint: `checkpoints/rival2/fresh_sustained_contact_v1/plus_000070.pt`

SHA-256: `624108BF03562FEFAD9EC3F0E6FECD36AA058FED533B71EFDA7D7F577D9746EA`

Raw probe, start receipt, audit and closed prefix accompany this report.
The frozen bounded run continues; no reward retuning, restart or promotion.
