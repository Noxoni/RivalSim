# Ground-to-air mechanics and curriculum decision

## Status

The protected V23 competitive policies and the passing controlled aerial scorer
remain byte-identical.  This work is read-only calibration and scenario design;
it has taken zero optimizer steps and has not changed production rewards.

The controlled scorer already transfers to realistic ground-to-air entries:

| setup | side | pop/contact | elevated follow | high follow | second airborne | goal <= 6 contacts | productive continuation |
|---|---:|---:|---:|---:|---:|---:|---:|
| low bounce | 0 | 80.86% | 19.92% | 3.12% | 2.34% | 1.76% | 8.20% |
| low bounce | 1 | 80.47% | 18.16% | 2.93% | 2.73% | 1.95% | 7.42% |
| incoming chip | 0 | 100.00% | 17.19% | 16.21% | 2.34% | 2.93% | 4.69% |
| incoming chip | 1 | 100.00% | 14.84% | 14.26% | 0.98% | 2.34% | 4.69% |
| matched dribble | 0 | 100.00% | 71.29% | 19.73% | 10.55% | 3.52% | 30.86% |
| matched dribble | 1 | 100.00% | 64.65% | 13.87% | 8.20% | 4.88% | 27.15% |

Source: `calibration.json`, 512 deterministic attempts per physical side and
setup (3,072 attempts total).  A native touch onset is one distinct contact;
continuous low-separation contact cannot be counted repeatedly.  A scoring
chain is capped at six distinct contacts.

## Physical technique represented

The supplied Rocket League clip is consistent with the ordinary low-bounce
route: a small grounded touch from behind/under the ball preserves close
separation, the car leaves the ground under the rising ball, and boost plus
pitch keeps the nose behind the ball for a short carry and finish.  It is not a
dead-ball vertical launch.

The three scenario families preserve the useful distinctions:

1. **Low bounce:** ball and car are already moving goalward.  The first touch is
   deliberately light; its purpose is separation and alignment, not maximum
   impulse.
2. **Incoming chip:** the ball rolls toward the car.  A low-front contact turns
   opposing horizontal momentum into a rising ball that can be followed.
3. **Matched dribble:** the car approximately matches the ball's ground speed.
   A double jump and pitch/roll/yaw control may use a front-corner contact to
   create space while the nose transitions upward.  A tornado-spin-like
   rotation is an optional orientation solution, not a mandatory animation.

Rocket League's first jump has a hold-dependent force interval, whereas the
second jump does not.  The useful control sequence is therefore: hold the first
jump, establish a modest nose-up attitude, release/re-jump if a fast aerial is
needed, and use boost to match the ball's flight rather than merely pointing at
its current position.  Directional air roll plus the opposite stick direction
produces the commonly named tornado-spin rotation, but the training target is
the physical corner-contact and post-contact attitude rather than a named-input
pattern.

## Why the prior dead-ball experiments were rejected

The hard direct launcher lifted the ball but left a median planar separation of
roughly 297 uu.  The softer direct hit preserved proximity but did not lift the
ball.  Both are worse starting distributions than a low bounce, incoming chip,
or matched dribble and are no longer part of the forward plan.

## Natural-play census and the real integration blocker

The deterministic V23-vs-V23 census covered 128 worlds for 6,000 ticks
(768,000 world-ticks per physical side).  It found 179 broad low-bounce
sequences without requiring boost: 13 on side 0 and 166 on side 1.  Those
sequences contained 219 native self-touch onsets.  Every one of the 6,413 broad
opportunity ticks had exactly zero boost.

Therefore the passing aerial controller is not the only bottleneck.  V23
creates low-bounce/close-ball geometry in ordinary play but reaches it after
spending its boost.  Enabling the option at zero boost would turn a promising
state into an impossible or low-ceiling jump.  The integration stage must keep
a physical minimum-boost gate and teach the ordinary policy to arrive with
enough boost, rather than hiding the problem by lowering the gate to zero.

A paired fixed-state boost sweep makes the threshold concrete.  Across both
physical sides, elevated-follow rates were:

| initial boost | low bounce | incoming chip | matched dribble |
|---:|---:|---:|---:|
| 0 | 1.56% | 1.95% | 3.12% |
| 5 | 3.52% | 6.64% | 4.69% |
| 10 | 6.25% | 12.11% | 10.55% |
| 20 | 17.58% | 32.42% | 53.12% |
| 35 | 19.14% | 33.20% | 64.84% |
| 50 | 21.09% | 26.95% | 72.27% |

The same 128 states per side/setup were reused at every boost level.  Twenty
boost is the first broadly useful operating point.  More boost chiefly helps a
matched dribble become a connected carry; incoming-chip results saturate
earlier because contact geometry, not fuel, becomes the limiting variable.

## Frozen forward decision

The next capability stage should:

1. retain the passing controlled scorer as the parent;
2. train only on a committed mixture of low-bounce, incoming-chip, and
   matched-dribble setups;
3. preserve the six-contact maximum and never reward raw airtime;
4. add a live defender after transfer competence is measured;
5. measure outcomes through touch height, physical contact count, sustained
   close control, goalward momentum, goals, and ground/time failures;
6. separately improve ordinary-play boost acquisition/conservation before the
   option is promoted into full self-play;
7. keep V23 responsible for ordinary play until a gated hybrid passes the
   existing competitive and physical telemetry checks.

The data supports continuing toward scenario self-play now.  It does not yet
support replacing the competitive policy or declaring natural aerial offense
solved.

## Sources

- Psyonix, *Using the New Free Play Controls and Training Pack Refresh*:
  possession, dribble, pass, and launch controls are explicit training setup
  primitives. <https://www.rocketleague.com/news/using-the-new-free-play-controls-and-training-pack-refresh>
- Rocket Science, *Rocket League: Boost and Jump*: empirical jump-hold,
  double-jump, tilt, and boost behavior. <https://rocketscience.fyi/know/videos/boost-and-jump>
- Grifflicious, *How To Ground to Air Dribble in Rocket League*: separates
  tornado-spin/pop control, rolling-ball chip setups, and bounce-touch
  correction. <https://www.youtube.com/watch?v=3miaW-kwQQg>
- Liquipedia Rocket League glossary: common directional-air-roll/tornado-spin
  terminology. <https://liquipedia.net/rocketleague/Liquipedia:Glossary>

These sources guide the state distributions and control interpretation.  They
do not substitute for RivalSim's authoritative 120 Hz physical telemetry.
