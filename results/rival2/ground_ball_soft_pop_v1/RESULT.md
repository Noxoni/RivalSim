# Ground-ball soft-pop exploration V1

Verdict: **negative exploratory result; no source primitive selected.**

The bounded 24-candidate, 32-world-per-side exploration reduced the car/ball
closing speed and removed approach boost.  It confirmed that a slower frontal
ground-ball contact creates many eventual recontacts, but not an air-dribble
entry.  The nominal best candidate raised the ball above 220 uu in only 12.5%
/ 21.875%, produced zero close airborne follow states and zero elevated follow
contacts, and left median follow gaps of roughly 299 / 338 uu.

Together with the rejected hard-pop run, this distinguishes two failure modes:

- a fast frontal hit lifts the ball but throws it too far ahead;
- a slow frontal hit is more recoverable but usually does not lift the ball.

The validated aerial option should therefore enter from its demonstrated
slightly-elevated possession geometry.  A ground-resting ball first requires a
separate possession/catch transition; it must not be represented as though a
single frontal jump hit were already a controlled air-dribble pop.

This exploration performed zero optimizer steps and changed no reward or
production policy.
