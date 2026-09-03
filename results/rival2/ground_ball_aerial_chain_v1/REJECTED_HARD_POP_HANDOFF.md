# Rejected hard-pop continuation run

Verdict: **REJECTED; not promoted and not used for self-play.**

The prospectively frozen V1 continuation ran 25 complete training blocks (300
PPO minibatch optimizer steps) before manual diagnostic stop.  Dense tracking
improved, but neither stochastic training nor deterministic validation produced
a single qualifying elevated follow contact.  No successful-trajectory or goal
rehearsal sample existed.

The preserved block-24 rolling checkpoint is diagnostic evidence only:

- external path: `G:/dev/RivalSim-runs/ground-ball-aerial-chain-v1/rolling.pt`
- SHA-256: `6CC1F88AFD84161E588A71B6282B44CBE6D51AB4A73F6A40A865D9D5CCF25C22`
- deterministic validation elevated follow touches: 0 / 1024 on both sides
- training elevated follow touches: 0 through block 25 on both sides
- mean training return improved from approximately -4.09/-4.08 at block 1 to
  -1.98/-1.93 at block 25

A focused 512-world-per-side deterministic trajectory diagnosis found the
causal geometry problem.  The source pop launched the ball to a median peak of
about 254 uu while the car reached 306--327 uu, but within 13--19 ticks the ball
was already about 297 uu ahead in the planar direction.  Any later second touch
was predominantly a ground-level reacquisition: median ball height 96--97 uu
and car height about 69--70 uu.  The policy was airborne, but the hard source
contact had created an uncatchable gap before learned control could help.

The next prospective source calibration must optimize a soft, low-separation
pop and immediate follow geometry.  Merely continuing to optimize the improved
dense proxy would not establish the requested physical capability.
