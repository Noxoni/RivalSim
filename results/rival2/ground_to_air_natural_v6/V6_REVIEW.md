# Natural ground-to-air V6 review

## Verdict

**NOT PROMOTED.** V6 proved that learned pop orientation was physically wired
and that channel-masked, equal-stratum PPO was stable.  It did not show that
the broad residual parameterization could preserve the simple low-bounce route
while learning the optional partial-tornado/front-corner route.  The run was
stopped after four complete boundaries (block 16).  The untouched test was not
opened and neither diagnostic checkpoint is an accepted parent.

## What the experiment established

- The scripted throttle/jump/boost/handbrake channels were excluded from PPO
  likelihood during the pop.  Only steer/pitch/yaw/roll received credit.
- Every optimizer step averaged equal samples from all twelve
  setup/defender/side strata.  Approximate KL remained telemetry only and was
  about `0.0013–0.0014` at reported boundaries.
- The parent actor's newly exposed orientation produced a defended
  matched-dribble contact above 300 uu even in the zero-step preflight, proving
  the action path was physically active.
- Incoming-chip performance improved during training, so the policy and reward
  could learn through the latent orientation mapping.

## Why it was rejected

The full pitch residual disturbed the proven fixed-pitch jump path.  At block
16, low-bounce/live elevated follow remained `5–9%`, below the `10%` live
floor and below the V5 diagnostic.  Both defended matched-dribble sides again
had zero >300-uu follow contacts.  The worst-row gate ratio was therefore still
zero.  Continuing solely because the aggregate tiebreaker rose from `1.263` to
`1.444` would repeat the exact average-masking failure the balanced design was
created to prevent.

## Prospective correction

Restart from the accepted controlled scorer.  Preserve its fixed `0.5` pop
pitch and scripted jump buttons.  Expose only a bounded steer/yaw/roll residual
during the pop (initially no learned pitch residual), with the same causal
channel mask and equal-stratum gradient aggregation.  This keeps the plain
double-jump solution exactly available while giving the policy a small
orientation correction for front-corner contact.  Continue to select by the
worst physical row, not the aggregate incoming-chip score.
