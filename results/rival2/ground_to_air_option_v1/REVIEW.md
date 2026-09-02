# Ground-to-Air Option V1 Review

Verdict: **REJECTED / NOT PROMOTED**

The prospective V1 curriculum proved that Rival can reliably create the initial
low-ball pop, but it did not meet the physical continuation gates.  The
validation-selected block was block 80.  On the once-opened test corpus, the
two side-specific policies produced:

- pop touch: 99.90% / 99.95%;
- qualified pop: 90.28% / 89.84%;
- elevated follow touch: 5.03% / 5.37%;
- high follow touch: 0.20% / 0.59%;
- second airborne touch: 0.73% / 1.07%;
- goal after pop: 0% / 0%.

The required gates were 15% elevated follow touches, 3% high follow touches,
and 1% second airborne touches on both sides.  Consequently, no V1 option
checkpoint was copied into the repository and nothing was integrated into the
protected V23 competitive policy.  The opened V1 test seed is diagnostic only
and must not be reused as prospective authority.

The monotonic learning curve and strong pop rate isolate the remaining problem
to the post-pop launch/recontact phase.  A successor curriculum should spend
most of its optimization budget on states at or after the neutral second jump,
train a single canonically oriented policy across both team perspectives, and
retain full-chain pop-to-follow evaluation as the promotion gate.

The external block-80 rolling checkpoints remain diagnostic parents only:

- blue SHA-256: `B13D6EABCA6075EC33371ED544D512DF42838DA6687437282E381DFE336E45A1`;
- orange SHA-256: `F353A8B784AD5619CE56C8E8B66D98A4E690F6C40652A629D90345CA753FF28A`.

All raw per-block telemetry is in `training_curve.jsonl`; the validation and
once-opened test summaries are in `result.json`.
