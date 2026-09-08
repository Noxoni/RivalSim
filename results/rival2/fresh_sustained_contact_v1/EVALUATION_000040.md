# Update 40: mixed acquisition movement, still poor scoring

The scheduled deterministic Nexto probe completed while update30 evidence was
being published. No additional GPU evaluation was launched. Audit is CPU-only.

| Update | Easy own contact within5s | Varied own contact within8s |
| --- | ---: | ---: |
| 0 | 31/512 (6.0546875%) | 11/512 (2.1484375%) |
| 20 | 247/512 (48.2421875%) | 48/512 (9.375%) |
| 30 | 218/512 (42.578125%) | 42/512 (8.203125%) |
| 40 | 203/512 (39.6484375%) | 77/512 (15.0390625%) |

Easy acquisition declined again; varied acquisition improved to this run's best
so far. This is mixed progress, not a general improvement or functional-gameplay
claim. Relative to20 there are44 fewer easy successes and29 more varied successes.
Successful-contact conditional lower medians are0.6916666666666667s easy and
1.1666666666666667s varied. Full8s no-focal-contact fractions are0.599609375 and
0.849609375. Easy5s success and full8s no-contact are different quantities.

In training updates31-40, learners received25,495 first-touch awards
(25487.592095077038 discounted reward) and7,940 off-ground awards
(1984.416649594903 reward). Nexto's67,035 first and88,184 off-ground awards
were excluded. Repeated contacts without another first award total7,831.
Air-budget-capped agent-decisions total186; this is not186 players or contacts.
Eligible and paid learner air awards both equal7,940 in this window. Mean learner
movement speed is701.5909081127026uu/s. These low-bounce-inclusive awards do not
prove aerial mechanics.

Nexto training worlds recorded58 Rival goals and67,709 opponent goals, with24,294
Rival touches over491519.99999799556 world-seconds and1 inactivity reset. These
are stochastic scenario outcomes, not match scores or win rates. Scoring against
Nexto remains very poor despite improvements in some acquisition starts.

CPU audit passed all17 checks: finite model/Adam, fresh lineage and contracts,
update/sample/physics/Adam counters, no KL rejection, all6family exposure,
contact-accounting, checkpoint identity and read-only evaluation. Independently
recomputed all contact fractions/counts, conditional lower medians and failures
from raw ticks; verified unchanged probe/spec identities and actual checkpoint
SHA against the start receipt. The prior update30 audit also verified frozen
source hashes, with existing CRLF working copies normalized as authority defines.

Update40 contains176,947,200 trainable decisions. Checkpoint:
`checkpoints/rival2/fresh_sustained_contact_v1/plus_000040.pt`

SHA-256: `ACB2FF5BA5E1CEDB6371AB9682E1AC77A8DD3670DFE2CF99D254C169AAB44FA6`

Raw probe/start receipt, audit and closed prefix are preserved alongside this
report. Training continues unchanged within the bounded authority; no reward
retuning, old-lineage restart or promotion was performed.
