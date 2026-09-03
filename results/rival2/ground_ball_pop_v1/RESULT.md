# Ground-ball pop calibration V1

Verdict: **PASS** for source-launch bootstrap selection.  This is not a claim
that the learned policy can yet complete the connected aerial continuation.

The frozen one-run calibration selected candidate 5:

- trigger distance: 185 uu
- first-jump hold / release: 8 / 6 physics ticks
- second jump: enabled
- pitch: +1.0
- approach boost: enabled

On 256 deterministic worlds per mirrored side, the selected source primitive
launched in 100% of attempts, raised the ordinary 92.75-uu ball above 180 uu in
97.66% / 99.22%, and above 250 uu in 54.69% / 58.98%.  The previously trained
aerial option produced only 0% / 0.39% elevated follow contacts after this new
ground-ball entry, so connected continuation remains the next training gate.

The calibration performed zero optimizer steps, changed no production reward,
used no named-mechanic classifier, and did not mutate state after episode reset.
