# Ground-to-air entry mechanics research for V11

Date: 2026-09-03

Status: design evidence only. This document does not change a policy, reward,
detector, optimizer, or the live V10 campaign.

## Evidence used

- The supplied 20.866-second Rocket League example:
  `C:/Users/patri/Videos/Medal/Edits/MedalTVRocketLeague20260902174434464-trim-1788386200969.mp4`,
  SHA-256
  `5085071BCADC19F9CE90CB4DBE376DE739F421E350F03F7A861B0989F12D8728`.
- The committed native 120 Hz analysis of 35 accepted human ground-to-air
  attempts in
  `results/rival2/ground_to_air_human_physics_v1/human_transition.json`,
  SHA-256
  `3B29ABE1CEBE1272ADC5D12D1FFB849209EADF1FF82C71582121791E08931B9E`.
- Rocket Science's measured jump mechanics: the first jump has up to 200 ms of
  held force; the second jump is a separate impulse and has no held-force
  phase: <https://rocketscience.fyi/know/videos/boost-and-jump>.
- AirCharged's ground-to-air tutorial, whose worked sections separately cover
  stationary chips, incoming rolling chips, rising bounces, carry entries,
  ordinary double jumps, and partial tornado/reverse-tornado adjustments:
  <https://www.youtube.com/watch?v=3miaW-kwQQg>.
- Psyonix's official training guidance treats aerial learning as a set of
  repeatable custom-training feeds rather than requiring the player to create
  every feed from live play:
  <https://www.rocketleague.com/news/community-spotlight--training-packs-for-new-drivers>
  and
  <https://www.rocketleague.com/news/using-the-new-free-play-controls-and-training-pack-refresh>.

The web/tutorial sources explain player execution. Native RivalSim physics and
the recorded human telemetry remain the acceptance evidence.

## Entry families are physically different

### 1. Assisted low bounce: primary learning entry

The ball begins with a shallow bounce in front of Rival. Rival is not required
to manufacture the bounce. The car reaches the lower/back part of the ball with
a light forward touch while the ball is rising or near its low apex, then leaves
the ground and follows the same trajectory. This is the route demonstrated at
the start of the supplied clip.

The important outcome is not a named "pop" event. It is a physical handoff:

1. a soft upward/goalward ball impulse;
2. the car remains below and close to the ball;
3. the car is already gaining vertical velocity;
4. a separated airborne contact occurs while the ball is still rising;
5. one or more later contacts continue the play toward goal.

This should be the easiest and most heavily represented V11 entry. The live V10
low-bounce row already supplies a bounce, but its one-time prompt objective is
paid at a much lower, earlier contact than the human continuation envelope.

### 2. Incoming rolling-ball chip: soft collision, delayed takeoff

When the ball rolls toward Rival, relative velocity supplies much of the upward
chip. The useful player sequence is a soft contact without driving excessive
boost through the ball, continued ground travel underneath the new rising arc,
and then a fast aerial. Immediate takeoff can leave the car behind the ball.

The live V10 incoming-chip generator uses a car moving 350--900 uu/s into a ball
moving 220--720 uu/s in the opposite direction, a 570--1,620 uu/s closing-speed
range. Its near-zero prompt-contact result is therefore not evidence that Rival
cannot learn a chip; it is evidence that the first curriculum row is too broad
and often too violent. V11 should begin with a slower, wider-gap incoming feed
and expand closing speed only after the soft-chip plus under-ball route is
routine.

### 3. Rising-bounce double-jump contact

For a ball already rising in front of the car, a plain double jump is sufficient
and often simpler than an additional ground chip. The second jump can coincide
with the setup contact, providing vertical impulse and reducing the tendency for
the collision to knock the car away. The car then pitches and boosts into the
continuation. Directional air roll is allowed but is not required.

This route should be represented independently from the assisted low-bounce
ground touch. Mixing them under one required controller sequence would teach a
false universal mechanic.

### 4. Roof carry into double jump

For a true dribble carry, the ball starts on the rear half of the roof rather
than already rolling off the front. The player holds the first jump through its
useful force window, applies the second jump after the hold, and only then leans
the nose into the rising ball. Pitching too early rolls the ball forward and
separates it from the car.

This is different from V10's `matched_dribble` feed, where the ball is already
38--92 uu in front of the car and merely has approximately matched forward
speed. A future carry row must use genuine roof-relative geometry and should not
replace the easier assisted-bounce row.

### 5. Partial tornado/front-corner carry variant

Directional air roll plus opposite yaw makes the nose rise during the first
half of a tornado rotation. Stopping or neutralizing the yaw near that half-turn
can present the hood/front corner to the ball while leaving the nose pointed
upward. The useful feature is the contact orientation and resulting soft
upward transfer, not continuous spinning.

A reverse-tornado adjustment has the complementary use of dropping the nose to
recover underneath a ball that escaped too far forward. Both are optional
orientation solutions. V11 must continue to accept a plain double-jump route
and must never reward a controller animation or named spin by itself.

## Human physical envelope to target after the first touch

The 35 accepted native human attempts establish the following broad reference:

- pop/setup to first airborne contact: median 49 ticks, observed 11--152;
- last jump to first airborne contact: median 20 ticks, observed 2--78;
- first to second airborne contact: median 130 ticks, observed 5--228;
- first-air-contact ball height: median 273.796 uu;
- first-air-contact car height: median 141.163 uu;
- first-air-contact car vertical speed: median 443.403 uu/s;
- first-air-contact car/ball distance: median 140.357 uu;
- first-air-contact car-up/ball alignment: median 0.94065;
- 27/35 attempts had at least two airborne contact episodes;
- the median total accepted contact count was three;
- 33/35 attempts used one through six contacts, with two 8/10-contact outliers.

These numbers are references and ranges, not a scripted imitation trace. They
show why V10's common immediate recontacts around car 65--80 uu and ball
180--195 uu do not yet constitute an air dribble.

## V11 staged curriculum implication

V11 should not ask one objective to solve every entry and the finish at once.

1. **Assisted handoff.** Predominantly low rising bounces, plus a bounded slower
   rolling-chip row. Establish soft upward transfer, under-ball travel, and
   takeoff. Do not reward raw airtime or a named mechanic.
2. **Human-envelope continuation.** After a genuine separated airborne contact,
   reward only bounded physical progress toward the measured car height,
   vertical speed, distance, standoff, and up-alignment envelope. This is the
   purpose of `HumanEnvelopeBridgeTrainingTracker`.
3. **Second airborne contact.** Require another native contact onset while the
   ball remains airborne and the play stays goalward. Keep the human timing
   window broad enough to include both carried and spaced touches.
4. **Short finish.** Require a goal or productive backboard continuation within
   at most six total contact onsets. Typical successful plays should use about
   one to five contacts; contact count itself receives no reward.
5. **Advanced entry branch.** Add true roof carry and optional partial-tornado
   geometry only after the primary bounce/chip continuation is physically
   reliable. The branch is selected by initial state, not by a required named
   action sequence.

This curriculum preserves the user's central correction: Rival is given a
usable natural opportunity and is trained to exploit it. It is not required to
invent the bounce before it can learn aerial possession.
