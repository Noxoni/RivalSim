# Natural ground-to-air V5 review

## Verdict

**NOT PROMOTED.** V5 was stopped prospectively after block 20 when source
inspection, prompted by the user's detailed pop mechanics, exposed a control
boundary that prevented the complete intended action family.  The untouched
test seed was not opened.  The competitive V23 policies and the accepted
controlled-scorer parent remain byte-identical.

The best validation checkpoint is block 16, stored externally for diagnosis
only.  Its score improved from `2.4631754557291665` to `3.8343505859375`.
It must not be used as the next training parent.

## What improved

- Low-bounce/live elevated follow reached `12–16%` at early selected
  boundaries, compared with the V4 best's `12.9%` and `0.4%` high-side tail.
- Incoming-chip/live at block 16 reached `57–64%` elevated follow,
  `29–34%` high follow, `20–21%` productive continuation, and `8–9%` goals.
- Matched-dribble/live reached `13–19%` elevated follow and `2–5%`
  productive continuation across selected boundaries.
- No selected row exceeded the six-contact budget.

## Why it stopped

`GroundToAirController.step` replaced the entire sampled action during the
first-jump hold, release, and second-jump ticks.  It forced a fixed pitch and
zero yaw/roll, then returned learned control after the second jump.  The policy
could learn a plain double-jump continuation, but it could not begin a partial
tornado/front-corner alignment during the pop, even though the V5 mechanics
interpretation intended to allow it.

The persistent failure pattern was consistent with this restriction:
defended matched-dribble produced elevated contacts but zero contacts with both
car and ball above 300 uu at the best boundary.  The run was stopped instead of
spending the remaining budget optimizing around an incomplete action space.

## Prospective successor requirement

Restart from the accepted controlled scorer, not either V5 diagnostic file.
The next authority must:

1. keep the scripted first/second-jump button timing;
2. expose steer, pitch, yaw, and roll to the learned option during the pop;
3. use channel-masked likelihood so scripted throttle/buttons receive no PPO
   credit or blame;
4. aggregate all setup/defender/side losses into a truly equal-weight update;
5. remove success-volume-dependent rehearsal from the entry stage;
6. retain the same physical outcome reward, six-contact maximum, live-defender
   validation, and no raw-airtime or named-mechanic reward.

Detailed block telemetry is in `training_curve.jsonl`.
