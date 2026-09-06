# Native kickoff isolation: residual handbrake affects motion

The prescribed diagnostic completed without any optimizer, production edits
or checkpoint mutation. Source/authority/code were committed and all12files
remotely read back at `91063785e5ae852b7c730bbf82972b1afa7a83fc` before execution.
It replayed only the first4,748 ticks of the established ten matches, then ten
private fixed-control arms of32ticks. Original common trace rows were exact.

Both initial and actual first-post-goal snapshot replays reproduced every
recorded position, velocity, quaternion, angular velocity, boost, ground bit,
ball state and wheel-contact bit **exactly**. All375 mutable per-world arrays
(including match/telemetry/controller storage) were captured/restored, with
the scalar native solver clock handled explicitly. All20 actual control tapes
were captured every physics tick. Matching initial and post-goal tapes are
identical in all ten worlds: opponent scheduling is not this comparison's cause.

## Positive causal result

Five of20 cars retain handbrake smoothing values after the first native goal
reset:0.4000004,0.1666667,0.1333333,0.4000004,0.1666667. Native kickoff reset
sets previous handbrake input to0 but does not clear `vehicle.handbrake_value`.
The vehicle prepass reads/decays this old value and uses it for wheel friction.

Replacing **only that cache** with the corresponding initial value (zero),
while restoring all other375array entries and replaying identical120Hz controls,
changes motion for exactly those five cars, and none of the other15. Maximum
position change over the recorded32tick window is4.936768uu; maximum velocity
change25.106720uu/s. The five per-car maximum position changes are4.936768,
0.997559,0.453125,4.936768,0.997559uu.

This confirms residual handbrake affects a fresh kickoff. It does **not** prove
that this small early difference causes the entire scoring regression or that
the policy would otherwise be good. No corrected match performance has yet
been measured. The existing curriculum reset already clears this field, so
the proposed fix should not change its training transitions; exact native
parity must still verify that claim before another campaign.

## Null results and limitations

- Replacing chassis contact count/normal changes no motion in these ten cases;
  both were already zero. This does not establish behavior for every later reset.
- Replacing wheel/world-contact/count caches alone changes no motion here.
- Quaternion-only replacement changes position by at most0.000061uu and
  velocity0.000542uu/s. It does not explain the handbrake effect.
- Cache replacement leaves up to1.865967uu/8.638029uu/s difference from a cold
  initial world. Residual differences are not fully attributed by this run.
- **The planned initial-warm clock-only arm is not a valid initialized
  reference.** Cold world resident car positions are zero until tick0
  initializes them. Skipping tick0 without populating those positions causes
  an artificial4608uu displacement. Preserve this failed control arm, but do
  not treat its large differences as a production defect or performance result.
  Native first-tick semantics must not be changed on this evidence.
- Resetting the private post-goal clock to0 after replacing caches/quaternion
  does not remove the residual cold-start difference. No claim that clock
  alone explains every discrepancy is supported.
- Only first post-goal windows were intervened on. The preceding full replay
  retains all205origins and shows larger later differences; those magnitudes
  must not be substituted for this narrower causal result.

## Integrity and next step

Raw `capture/native_replay.npz` SHA256:
`EB7F08E5EC794B2C2B608318BFCFF0391FF8F4BCD48C8E86361225DE9862D2D8`.
This580,123byte archive contains881 named arrays: raw origins, controls,
captured physical traces, both baseline replays and every intervention arm.
`capture/report.json` is exactly reproducible with the CPU reducer in
`benchmarks/diagnose_direct_skills_native_kickoff.py`.
`capture/manifest.json` binds authority, implementation,375-array inventory,
source trace, checkpoint and20 frozen runtime sources.

The selected675 checkpoint remains
`77E6EF076F7073549F45168FA6FB34BE9A2D4C25E1580F17F7434A59195B32F8`.
Focused tests additionally check raw replay, per-tick control identity,
handbrake-specific causal scope and the invalid warm-reference limitation.

Next: clear residual handbrake on selected standard-kickoff resets, prove
unselected worlds and the existing curriculum reset path unchanged, then use
an explicitly versioned corrected-match method. Do not silently compare
corrected675 against an old method or attribute a runtime correction to
learning. Preserve current review STOP until the new prospective training
decision. No new reward, opponent, model or exploration change is warranted
by this diagnostic alone.
