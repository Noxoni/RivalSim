# Update 50: sharp deterministic acquisition regression

The scheduled frozen1,024-start deterministic Nexto probe completed. No additional
GPU evaluation, training interruption or reward change was made for this audit.

| Update | Easy own contact within5s | Varied own contact within8s |
| --- | ---: | ---: |
| Untrained | 31/512 (6.0546875%) | 11/512 (2.1484375%) |
| 20 | 247/512 (48.2421875%) | 48/512 (9.375%) |
| 40 | 203/512 (39.6484375%) | 77/512 (15.0390625%) |
| 50 | 39/512 (7.6171875%) | 27/512 (5.2734375%) |

This is a large regression, not healthy continued improvement. Relative to40,
easy loses164 successes (-32.03125 percentage points) and varied loses50
(-9.765625 points). Both remain numerically above the untrained baseline, but
most earlier easy acquisition improvement is gone. Do not promote the checkpoint
or infer good gameplay from successful serialization or finite training.

Conditional successful-contact lower medians:0.6416666666666667s easy and1.775s
varied. The easy conditional median improving does NOT imply overall improvement:
it describes only39 remaining successes. Full8s no-focal-contact fractions are
0.921875 easy and0.947265625 varied. Easy5s success differs from full8s contact.
This probe tests own contact in the original episode, not beating Nexto to the
first touch, retaining possession, scoring, or aerial mechanics.

## What remains verified

CPU report passed all17 frozen integrity checks: finite model/Adam, lineage and
contracts, contiguous update/sample/physics/Adam counters, zero KL rejection,
all-six-family exposure, both bonus accounting, actual checkpoint identity and
read-only evaluation. Independently recomputed all counts/fractions, conditional
lower medians and no-contact fractions from raw ticks; checked unchanged probe
and specification identities and checkpoint SHA against started receipt. Frozen
source identities remain unchanged under the authority's LF text convention.
Thus no numerical/identity/contact-payment failure was found in these checks;
they do not determine the behavioral cause of regression.

Update50 contains221,184,000 trainable decisions and7,294 Adam steps. Its
completed-update mean KL is0.00416567469022724, maximum sample KL
0.1529456377029419, and entropy4.241283893585205. KL remains telemetry only;
no KL rejection, rollback, guard adjustment or automatic retuning occurred.

## Stochastic training window41-50

Learner first-touch awards26,985, discounted reward26977.192861437798.
Learner off-ground awards7,585, discounted reward1895.6983144432306.
Excluded Nexto awards55,680 first and90,836 off-ground. Repeated contacts without
another first award7,757; air-budget-capped agent-decisions0. Both new rewards
continue reaching learner samples. Off-ground includes low bounces, not proof
of aerial offense.

Mean learner speed611.4676901584202uu/s versus701.5909081127026 in31-40.
Nexto worlds:44 Rival goals,57,436 opponent goals,23,766 Rival touches over
491519.9999976569 world-seconds; no inactivity resets. These stochastic scenario
outcomes are not full-match scores/win rates and do not countermand the poor
deterministic acquisition result. Scoring remains extremely weak against Nexto.

Checkpoint: `checkpoints/rival2/fresh_sustained_contact_v1/plus_000050.pt`

SHA-256: `B602F1BC3411B7B02E6056E385AA85FB3FD8B2841426DD97DDA992607D467047`

Raw probe/start receipt, audit and closed training prefix accompany this report.
The healthy process continues under the frozen bounded150-update authority.
No new failure threshold, reward tuning, restart or promotion was introduced.
