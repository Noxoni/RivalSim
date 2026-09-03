# V15 Ground-to-Air Handoff Diagnosis

This is a read-only diagnosis. It took no optimizer step, changed no policy or
reward, and used the protected V3 aerial scorer byte-for-byte at SHA-256
`F7049F8EF6CC4D1EE3F7303D6D9CE1AA2207A10F6651A33BC71B7C344CC77154`.
The V23 Blue and Orange policies also remained frozen.

## Natural exact-V3 result

In 512 deterministic V23 self-play worlds for 6,000 native 120 Hz ticks, the
default V12 router activated V3 274 times: 160 rising-double-jump routes and
114 incoming-chip routes. No roof-carry route occurred. The chip routes all
ended on the ball reaching the floor. Of the rising routes, 103 reached a first
airborne contact and 57 remained active at the bounded evaluation endpoint.
There were zero separated second airborne contacts and zero V3-attributed
goals.

The 103 reset releases were real scoring events, not stale lifecycle
bookkeeping: all 103 were opponent goals while the option was active, and none
were goals for the routed player. This also rules out a side/canonical-frame
sign error because the same exact V3 policy has side-symmetric controlled
scoring evidence.

## State-distribution mismatch

All 103 natural entry events converged to the same deterministic state. At the
first airborne contact, the ball was moving goalward at 2,255.641 uu/s, the car
at 1,592.313 uu/s, and only 0.193 boost remained. The ball was 150.360 uu from
the car and the opponent was 781.239 planar uu from the ball.

Across the new 4,096-attempt controlled replay, V3 produced 983 rising-ball
entries and 93 separated second contacts. In the live-defender rising-ball
success subset, the corresponding means were 1,106.817 uu/s ball speed,
977.651 uu/s car speed, 0.644 boost, and 935.525 uu opponent-ball distance.
Relative to all controlled rising successes, the natural contact was about
4.99 standard deviations high in ball speed and 4.62 standard deviations high
in car speed, while boost was about 2.02 standard deviations low. The natural
handoff is therefore physically outside V3's demonstrated continuation
envelope even though its contact height, distance, and orientation are broadly
compatible.

The original router begins rising-ball control at a mean planar gap of 475.278
uu while the V11 rising-ball training distribution starts at 150-225 uu. It
also permits low boost and has no car-speed or ball-speed upper bound. That
explains why it can obtain a forceful first contact without being able to stay
with the ball afterward.

## Aligned-router falsification

A no-update probe tightened the router to the controlled envelope: at least
0.5 boost, ball in the opponent half, 0.75 forward alignment, 260 uu maximum
planar distance, no ownership slack, and a shorter 240-tick option horizon. It
produced zero activations in the same 512-world natural run. Thus simply
tightening the router would preserve competitive play by never invoking the
option, but it cannot integrate aerial offense because V23 does not naturally
visit V3's learned entry distribution.

## Decision

V3 remains the protected aerial parent, and V14 remains unpromoted. The next
prospective stage must broaden the protected scorer to the measured high-speed,
low-boost natural takeover distribution, with explicit second-contact/goal
success and concession/floor failure outcomes. It must first pass controlled
high-speed validation and then a no-update natural self-play gate. More PPO in
the old broad router, or a strict router that never activates, is not supported
by this evidence.
