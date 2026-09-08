# Update80: mixed acquisition movement, no overall recovery

Scheduled deterministic frozen1,024-start Nexto evaluation; no extra GPU probe.

| Update | Easy own contact within5s | Varied own contact within8s |
| --- | ---: | ---: |
| Untrained | 31/512 (6.0546875%) | 11/512 (2.1484375%) |
| 40 | 203/512 (39.6484375%) | 77/512 (15.0390625%) |
| 60 | 25/512 (4.8828125%) | 20/512 (3.90625%) |
| 70 | 68/512 (13.28125%) | 52/512 (10.15625%) |
| 80 | 49/512 (9.5703125%) | 64/512 (12.5%) |

Easy loses19 successes versus70 while varied gains12. Both remain below their
respective peaks (easy247 at10/20; varied77 at40). No sustained acquisition or
overall gameplay recovery is established. Conditional successful-contact lower
medians:1.2833333333333334s easy,1.8583333333333334s varied. Full8s no-own-contact
fractions:0.884765625 and0.875. Easy5s success and full8s no-contact differ.
This probe measures own contact in the original episode, not winning possession,
scoring, aerial mechanics or full-match competence.

Training71-80: learner first awards18,295/reward18289.815958619118; off-ground
awards5,335/reward1333.3700658679008. Nexto first20,775 and off-ground47,454
awards excluded. Repeated contacts without another first payout6,636; capped
agent-decisions0. All6families exposed; off-ground includes low bounces.

Nexto training worlds:8 Rival goals,21,322 opponent goals,11,260 Rival touches
over491519.9999959893 world-seconds,4,961 inactivity resets. Previous61-70 had
9/22,931 goals,12,210 touches and5,223 inactivity resets. Mean learner speed
475.6440993923611uu/s versus478.4199354835793. These are stochastic scenario
outcomes, not match scores/win rates; the varied-probe gain does not imply good
gameplay. Own scoring and acquisition remain weak.

All17 CPU report checks passed: finite model/Adam, lineage/contracts, exact
update/sample/physics/Adam counters, zero KL rejection, family exposure, contact
reward accounting and read-only evaluation/checkpoint identity. Independently
recomputed raw tick counts/fractions, conditional lower medians and no-contact
fractions; checked actual checkpoint SHA against receipt/probe and unchanged
scenario/spec identities. All frozen sources match LF-normalized hashes.

Update80 contains353,894,400 learner decisions and11,502 Adam steps. Completed
update mean KL0.003934366998900832, maximum sample KL0.18026113510131836,
entropy4.075860500335693. KL is telemetry only. Integrity is not behavioral PASS.

Checkpoint: `checkpoints/rival2/fresh_sustained_contact_v1/plus_000080.pt`

SHA-256: `EA392ED54F81A5B55B71925AD1F6421ED93FB8E834A7F25D3A762AB37D56A12A`

Raw probe/start receipt, audit and closed prefix preserved. Training continues
under unchanged bounded authority; no interruption, retuning, restart or promotion.
