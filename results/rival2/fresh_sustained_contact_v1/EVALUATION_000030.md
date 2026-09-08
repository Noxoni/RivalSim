# Update 30 acquisition evaluation

The scheduled read-only deterministic evaluation completed on the frozen 1,024
starts against active Nexto. This audit used CPU only and did not interrupt the
healthy runner or add an evaluation. Reward/configuration remain frozen.

| Update | Easy own contact within 5s | Varied own contact within 8s |
| --- | ---: | ---: |
| Untrained | 31/512 (6.0546875%) | 11/512 (2.1484375%) |
| 20 | 247/512 (48.2421875%) | 48/512 (9.375%) |
| 30 | 218/512 (42.578125%) | 42/512 (8.203125%) |

Both measures regressed from update 20: minus29 easy successes and minus6 varied
successes, or -5.6640625 and -1.171875 percentage points. They remain above the
untrained baseline. This is not steady improvement, and no claim of functional
gameplay or promotion is warranted.

Conditional lower-median contact times are 0.6333333333333333s easy and
1.1583333333333334s varied, versus 0.5916666666666667s and1.05s previously.
Over the full8s, no-focal-contact fractions are0.57421875 and0.91796875. This
measures own contact in the original episode, not winning first touch, scoring,
possession or aerial mechanics.

## Training window 21-30

All six scenario families had measured exposure in every update. The window
contains44,236,800 trainable decisions. Learners received20,008 first-touch
awards (20002.238678514957 discounted reward) and6,426 off-ground awards
(1606.029643625021 reward). Nexto's71,995 first and90,474 off-ground awards
were excluded. Repeated contacts without another first-touch award:6,119.
No off-ground payment budget was reached. Low bounces count as off-ground;
these counters are not evidence of aerial offense.

Aggregate learner-perspective touches26,127, goals1,254, concedes74,140 and
world inactivity resets7,558. Mean sampled learner speed675.8507951298467uu/s.
In Nexto worlds specifically, focal goals68, opponent goals72,954, focal
touches23,342 over491519.9999981194 world-seconds; zero inactivity resets.
Those are stochastic training scenario outcomes, not full-match win rates.
The policy remains very weak against Nexto.

## Integrity and persistence

The CPU report passed all17 checks, including checkpoint/parent/contracts,
finite model and Adam, exact training/optimizer counters, zero KL rejection,
all-family exposure, contact-payment accounting and read-only evaluation.
Independently recomputed counts, fractions, conditional lower medians and
no-contact fractions from raw ticks. Checkpoint hash matches the evaluation
and started receipt; scenario/spec hashes match baseline. All frozen source
hashes match using the authority's LF-normalized text convention. An initial
raw-byte source check differed only on existing CRLF working copies; committed
blobs and normalized source hashes match. No source edit was made.

Checkpoint: `checkpoints/rival2/fresh_sustained_contact_v1/plus_000030.pt`

SHA-256: `DDECF9427E610D26C9DEA7BFE4CC91AC9D1DAF1B9E8DDA34828E2A2C3C4DB199`

Raw probe, started receipt, audit and closed training prefix accompany this
report. Keep the frozen bounded run and scheduled probes; do not silently
retune rewards, resume an old lineage or promote this checkpoint.
