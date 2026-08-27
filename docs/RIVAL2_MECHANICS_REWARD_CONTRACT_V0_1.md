# Rival 2.0 Mechanics Reward Contract — Research Draft V0.1

Status: **research/design draft**

This document is the working authority for how Rival 2.0 should identify useful
Rocket League mechanics from authoritative physics. It records mechanical
identity, completion evidence, anti-exploitation rules, subtype/tactical
signatures, and unresolved calibration work.

This document does **not** authorize changing the currently active Rival 2.0
Opponent Curriculum V1 / Gameplay V2 run. The active handoff remains authoritative
for that bounded campaign. Any future reward implementation based on this file
must receive a new immutable reward identity and an explicit launch package.

## 1. Design doctrine

### 1.1 Physics is authoritative; names are labels

Community mechanic names are useful for communication, but Rival is not rewarded
because an input sequence resembles a tutorial. A mechanic completes only when
the simulator records the defining physical state transition and the useful
physical result.

Examples:

- a wavedash is not `jump + dodge near ground`; it is an airborne dodge/surface
  transition that actually converts the dodge into useful surface-tangent
  motion;
- a flip reset is not `car looked upside down under ball`; it is a wheel/ball
  grounding transition that restores an airborne dodge resource;
- a Musty is not merely `backflip while upside down`; it is a backward-dodge
  scoop in which the rotating hitbox transfers useful momentum through the ball;
- a pinch is not merely `car and ball near wall`; it is a short-lived compression
  topology between car, ball, and a world surface that produces a real impulse.

### 1.2 Capability reward and tactical value are separate problems

The low mechanic reward exists to tell Rival:

> This physical capability is real and worth retaining.

It should **not** try to encode every reason a human would choose that mechanic.
The normal game reward and opponent interaction should teach when the mechanic is
useful.

This distinction is especially important for flick variants. A Breezi, delayed
Breezi, Musty, nose-catch Musty, catch-pop, ordinary flick, pogo continuation,
or fake can have very different tactical value even when several share a common
terminal ball-flick physics class.

The variant therefore remains a first-class telemetry label with its own
measurable setup and release signature. Rival should be able to learn conditional
use from experienced opponent states rather than having a hand-written
`deception reward`.

### 1.3 Do not collapse meaningful variants

Mechanics may share a family without being behaviorally identical.

For example:

- a Breezi contains a Musty-class terminal scoop but adds a rotational setup and
  delay that changes release timing and readability;
- a Musty can arise from a ground dribble, aerial, wall/ceiling state, flip
  reset, catch, nose catch, or pop; those entry states change concealment,
  trajectory, recoverability, and available follow-ups;
- a reset can be soft, pop-style, rapid, pre-flip, pancake/wall-assisted,
  Musty-assisted, or chained; all restore the same resource, but the approach
  and post-reset state can be materially different.

The detector architecture must preserve those distinctions even when multiple
variants share a base primitive.

### 1.4 Anti-double-counting is hierarchical, not absolute

Labels describing the **same physical accomplishment** do not stack.

Example:

`wavedash + wall_dash + landing_wavedash`

may all describe one event, so that event receives one mechanics-tier payout.

However, a compound play may contain multiple genuinely separate accomplishments.
For example:

`flip reset acquired -> reset consumed in Musty scoop -> useful flick`

contains a resource-acquisition event and a later ball-manipulation event. Those
may be separately rewardable if the future reward contract explicitly authorizes
both. The rule is therefore:

> Never pay twice for one physical transition; do not forbid payment for two
> distinct completed transitions merely because humans give the compound play
> one name.

### 1.5 Every detector is a state machine

No mechanic reward may be implemented as `condition true every tick -> reward`.
Every rewardable detector must follow:

`IDLE -> START -> PROGRESS -> COMPLETE -> LOCKOUT/RESET`

Completion pays at most once. A new payout requires the state machine to leave
its completed state and a genuinely new qualifying sequence to begin.

## 2. Authoritative physics primitives

RivalSim already exposes most of the primitives required by this contract.
Future implementation should keep them device-resident in the training hot path.

### 2.1 Car state

Required per physics tick:

- position and orientation;
- linear velocity;
- angular velocity;
- local forward/right/up axes;
- grounded state;
- wheel contact mask;
- per-wheel contact point, contact normal, suspension length/velocity;
- per-wheel contacted-body identity: world, ball, or other car;
- `has_jumped`, `is_jumping`, `has_double_jumped`, `has_flipped`,
  `is_flipping`;
- `jump_time`, `air_time`, `air_time_since_jump`, `flip_time`;
- actual flip-relative torque / actual dodge onset;
- controller state for audit only, not as primary mechanic authority.

### 2.2 Ball state

Required:

- position/orientation;
- linear/angular velocity;
- ball/world manifold contacts;
- contacted world face/surface class;
- contact normals and impulses.

### 2.3 Car-ball state

Required:

- contact onset and persistent contact state;
- car-side and ball-side contact points;
- contact normal/tangent;
- normal/tangent/push impulses;
- pre-contact car and ball linear/angular velocity;
- post-contact ball velocity;
- RocketSim extra-hit velocity contribution;
- relative position on ball.

### 2.4 Derived quantities

The common detector layer should derive these once and reuse them:

#### Surface-tangent velocity

For contacted surface normal `n` and car velocity `v`:

`tangent(v,n) = v - n * dot(v,n)`

This is the correct basis for floor/wall/curve dash acceleration rather than
world-XY speed.

#### Contact-point velocity

For car center-of-mass linear velocity `v`, angular velocity `omega`, and contact
point offset `r`:

`v_contact = v + cross(omega, r)`

The rotational contribution is:

`v_rot = cross(omega, r)`

This is central to flick/Musty classification because it lets us determine
whether the rotating hitbox actually swept through the ball rather than merely
translating into it.

#### Ball velocity transfer

At a car-ball contact onset retain:

`delta_v_ball = ball_velocity_after - ball_velocity_before`

plus the solver impulse and RocketSim extra-hit contribution. Detectors can then
ask whether a claimed mechanic produced a real ball manipulation.

#### Dodge-resource state

The common layer should expose whether the car currently has:

- grounded jump availability;
- a timed airborne dodge originating from a jump;
- an untimed airborne dodge (`airborne && !has_jumped` with flip/double-jump
  resource still available);
- no dodge resource because it has been consumed.

This avoids re-deriving flip-reset semantics in every reset classifier.

#### Defender-relative tactical geometry

For telemetry only, retain per opponent:

- defender distance and closing velocity;
- defender-to-ball sight line;
- angular separation between defender->ball and defender->car;
- whether the ball substantially occludes the attacker's chassis from the
  defender's viewpoint;
- opponent jump/dodge/challenge onset relative to mechanic release;
- release latency from possession/setup start;
- resulting launch direction, speed, elevation, and possession outcome.

These values are **not direct mechanic rewards in V0.1**. They are how we later
measure whether a variant's tactical signature actually creates delayed reads,
concealment, forced challenges, or better follow-up states.

## 3. Reward bracket intent

Mechanics belong in a low reward bracket below ordinary useful ball interaction,
demos, saves, and goals. The exact future coefficient is not frozen by this
research document. The existing Gameplay V2 double-dash experiment uses `+0.005`
and that is a useful reference scale.

The future mechanics contract should also impose a bounded per-episode mechanics
budget or equivalent frequency control so a high-frequency movement mechanic
cannot dominate game objectives.

Frequency and reward influence must be audited empirically. Equal event payout
does not imply equal training influence when one event occurs hundreds of times
more often than another.

## 4. Movement and dash mechanics

### 4.1 Successful wavedash

**Start**

- car has zero wheel contact;
- an actual directional dodge/flip begins;
- the car is near enough to a surface that a landing can participate in the
  active dodge.

**Progress**

- first wheel contact occurs within the existing researched landing window;
- the surface interrupts/redirects the flip rotation while the dodge impulse
  contributes translation.

**Complete**

- compare car velocity tangent to the contacted surface immediately before the
  dodge against the retained post-landing sample;
- completion requires a positive surface-tangent speed result.

**Subtype labels**

- `landing_wavedash`: airborne phase need not have begun with a fresh jump;
- `wall_dash`: contacted surface is wall-like;
- `curve_dash`: contacted surface is a curved transition;
- floor/ceiling labels follow the actual surface normal.

Subtype labels do not stack a second payout on the same dash event.

The existing RivalSim dash telemetry is the starting authority for timing,
wheel-mask, suspension, landing-normal, and tangent-speed evidence.

### 4.2 Rival double dash

This project uses the observed Rival physics definition rather than an external
community naming rule.

**Start**

- first successful dash/wavedash completion is recorded.

**Progress**

- an intervening wheel-contact/landing transition occurs;
- a second successful dash begins within the established short sequence window.

**Complete**

- the second dash completes;
- retained post-second-dash surface-tangent speed is greater than the
  pre-first-dash tangent speed;
- both component dash events remain auditable.

A fresh jump between the two dash-like transitions is telemetry, not an automatic
rejection. This preserves the literal Rival-discovered sequence already measured
in this project.

### 4.3 Zapdash

**Start**

- angled landing reaches front wheels before rear wheels.

**Progress**

- the car reaches grounded jump availability before settling into an ordinary
  flat four-wheel landing;
- a fresh jump occurs in that partial/angled support state;
- the jump pops the front of the car away from the surface;
- a directional landing dodge/wavedash follows.

**Complete**

- the final dash transition succeeds;
- surface-tangent speed after the sequence exceeds the sequence's initial
  tangent speed.

Required evidence includes named wheel masks, first-contact order, suspension,
jump onset, actual dodge onset, surface normal, and speed before/after.

### 4.4 Speedflip

A speedflip is an efficient forward-acceleration/reorientation sequence, not just
`diagonal flip + cancel`.

**Start**

- car is supported or just leaving a support surface;
- actual forward-diagonal dodge begins.

**Progress**

- the pitch component of the flip is canceled/arrested early relative to a
  normal completed diagonal flip;
- yaw/roll/remaining rotation reorients the car while the translational dodge
  impulse is retained;
- car forward axis returns toward the travel direction before ordinary flip
  rotation would have completed.

**Complete**

- car establishes a controllable orientation with its forward axis substantially
  aligned with its velocity;
- useful forward/surface-tangent speed is retained or increased compared with
  the pre-sequence state.

Boost use is telemetry. It must not be required for the mechanical identity.
Thresholds for `pitch arrested`, heading alignment, and retained speed require a
positive/negative calibration corpus before reward activation.

### 4.5 Half-flip

**Start**

- actual backward dodge begins while the car is moving or oriented such that a
  reversal is useful.

**Progress**

- backward pitch rotation is canceled before completing a normal backflip;
- roll/yaw reorientation continues.

**Complete**

- the car regains a controllable wheel-supported or near-supported state facing
  approximately opposite its original heading;
- useful momentum is retained in the new forward direction rather than the car
  stopping and performing an ordinary turn.

Heading reversal, momentum retention, and stable support thresholds require
trace calibration.

### 4.6 General chain dash

Keep a generalized telemetry label for three or more successful dash events with
intervening support transitions. Do not automatically pay every component
indefinitely. This is primarily a discovery/analysis channel until frequency and
exploit behavior are understood.

## 5. Dodge-resource / flip-reset mechanics

### 5.1 Source state semantics

The pinned RocketSim behavior and current RivalSim state machine distinguish
between jumping off a surface and becoming airborne without jumping.

- a real grounded jump sets `has_jumped` and eventually starts the airborne
  `air_time_since_jump` expiration path;
- falling or being knocked from a surface without jumping leaves `has_jumped`
  false, so the airborne dodge resource is untimed;
- grounding clears consumed flip/double-jump state;
- RocketSim recognizes ground when at least three wheels are in contact;
- current RivalSim wheel rays already distinguish ball contact from world and
  other-car contact.

This is the authoritative basis for reset detection.

### 5.2 Ball flip reset

**Start**

- car is airborne;
- record the pre-contact dodge-resource state so the detector can prove a new
  resource was acquired rather than merely observing a car that already held an
  untimed flip.

**Progress**

- at least the source grounding threshold of wheels contact the ball through
  wheel rays;
- the support event causes the normal grounded resource-reset transition;
- contact-body identity proves the support object was the ball.

**Complete**

- car separates from the ball without taking a new grounded jump;
- car remains airborne;
- `has_jumped == false`, `has_flipped == false`, `has_double_jumped == false`;
- a usable untimed airborne dodge is therefore present after the event.

Reward on resource **acquisition**, not merely wheel proximity to the ball.

### 5.3 Chain / multiple reset

A second reset may complete only after the first acquired resource has been
meaningfully consumed/lost or the state has otherwise left the first acquisition
lockout.

**Complete** when a distinct later ball-wheel support event reacquires the
resource before world grounding.

Merely maintaining or re-touching the ball while already holding the same
untimed dodge is not a second reset reward.

### 5.4 Pre-flip reset

**Start**

- actual dodge is consumed before ball-wheel support.

**Complete**

- a subsequent valid ball flip-reset transition restores the just-consumed
  resource.

The pre-flip approach is a meaningful subtype because it changes approach speed,
orientation, and reachable contact geometry. The base resource restoration and
pre-flip subtype describe the same acquisition event and should not blindly
stack two reset payouts.

### 5.5 Pop reset

A pop reset combines resource acquisition with controlled ball displacement.

**Start / reset component**

- valid ball reset acquisition.

**Additional physical signature**

- wheel/ball contact is sufficiently soft/aligned that the car remains close to
  the ball after reset;
- the newly acquired dodge is consumed quickly;
- that dodge produces a controlled upward/outward ball impulse while Rival
  remains capable of continuing the aerial sequence.

This is more than a label for the same reset. It contains a later distinct ball
manipulation and should remain separately measurable. A future reward contract
may authorize the reset and the subsequent pop as separate physical completions,
subject to the episode mechanics budget.

### 5.6 Rapid reset

**Start**

- Rival already has or has just acquired a dodge;
- an actual fast pre-flip/speedflip-like airborne rotation begins toward the
  ball.

**Complete**

- that dodge is consumed;
- the rotating car obtains a new valid ball-wheel reset with very short latency
  between consumption and reacquisition.

The defining feature is resource turnover rate plus the pre-flip contact
geometry, not a community input recipe.

### 5.7 Pancake / wall-assisted reset

**Start**

- ball is simultaneously on/near a wall-like world manifold;
- car approaches with wheels oriented toward the ball.

**Complete**

- ball is temporarily compressed between wheel support and wall/world geometry;
- the wheel contact restores the airborne dodge resource;
- ball momentum after the compression is retained for telemetry because a
  useful pancake often changes/kills ball speed.

This must be distinguished from a generic wall pinch: the defining resource
result is the reset, not merely ball acceleration.

### 5.8 Car reset

Same resource-acquisition state machine as the ball reset, except the wheel-ray
contact-body identity is the other Octane rather than the ball.

### 5.9 Ceiling-retained dodge / ceiling shot resource

Falling from the ceiling without jumping naturally produces an untimed airborne
dodge, so merely leaving the ceiling should not earn a mechanics reward.

A rewardable ceiling-resource play should complete only when Rival **uses** that
retained resource in a meaningful continuation, such as:

- retained dodge -> controlled aerial ball touch;
- retained dodge -> successful flick/shot/redirect;
- retained dodge -> another separately rewardable mechanic.

The ceiling-origin label remains telemetry even when no mechanics bonus fires.

## 6. Possession and repeated-contact mechanics

### 6.1 Possession epoch

Repeated-contact rewards require a shared possession state machine.

A possession epoch begins on a legitimate unique player touch and ends on any of:

- opponent touch;
- clear loss of the ball outside a calibrated control envelope;
- goal/reset/demo state that destroys continuity;
- excessive time/separation without plausible continued control.

Contact-manifold chatter is not a new touch. A new discrete touch requires either
a genuine contact break/recontact or another source-backed unique-contact event.

### 6.2 Controlled two-touch possession

**Complete** when the same player makes a second legitimate, physically distinct
ball contact during the same possession epoch and the ball remained plausibly
controlled between touches.

This is the first repeated-contact milestone because it is achievable earlier
than a three-touch dribble and creates a usable learning signal.

### 6.3 Controlled three-touch possession / dribble

**Complete** when the same possession epoch reaches a third legitimate controlled
contact.

After the third-touch milestone, the sequence enters reward lockout. Touch four,
five, six, etc. remain telemetry but do not continuously print mechanics reward.

### 6.4 Ground carry

A carry may involve persistent support rather than clean touch onsets.

**Start**

- ball enters the car's upper/roof control region.

**Progress**

- car-ball manifold/support relation persists;
- relative car-ball motion remains within a calibrated control envelope;
- ball is not being compressed against world geometry as a pinch.

**Complete**

- support persists for a minimum meaningful duration or survives a meaningful
  change in car velocity/direction without losing the ball.

A carry is one sequence event, not a reward each tick.

### 6.5 Bounce dribble

**Start**

- Rival has possession;
- ball makes a ground/world bounce.

**Progress**

- Rival remains positioned to control the rebound;
- Rival produces the next deliberate car-ball touch before opponent possession.

**Complete**

- repeated controlled bounce/touch cycle reaches the authorized milestone.

This remains distinct from a roof carry because the ball-world contact is part of
the control pattern.

## 7. Aerial control mechanics

### 7.1 Ground-to-air pop continuation

The pop itself is a setup milestone, not necessarily a reward.

**Start**

- Rival has controlled ground possession or a controlled ground touch.

**Progress**

- Rival contact gives the ball a meaningful upward trajectory;
- Rival becomes airborne and pursues the ball.

**Complete**

- Rival obtains a second legitimate ball touch while airborne before opponent
  touch or Rival world landing.

This prevents Rival from farming meaningless upward pops.

### 7.2 Two-touch air dribble

**Start**

- first legitimate Rival touch occurs while Rival is airborne, or an authorized
  ground-to-air pop transitions into airborne control.

**Complete**

- second distinct Rival touch occurs while Rival remains airborne and before an
  opponent touch/world landing ends the aerial possession epoch.

This is the first rewarded aerial-control milestone.

### 7.3 Three-touch air dribble

**Complete** when the same aerial possession epoch reaches a third distinct Rival
contact.

This may receive an additional bounded milestone signal. Further touches remain
telemetry unless a later contract explicitly adds another capped milestone.

### 7.4 Extended aerial-control descriptors

Record without direct V0.1 reward:

- touch count;
- time between touches;
- ball height at each touch;
- car/ball relative speed;
- boost used;
- launch/exit direction;
- whether Rival remains goal-side or recovers after the sequence;
- defender position and challenge timing.

These values let later training/evaluation distinguish a controlled air dribble
from repeated accidental aerial collisions.

## 8. Flick mechanics

### 8.1 Base controlled-flick prerequisite

A controlled flick requires possession before release. A random dodge into a
loose ball is not a flick merely because the ball accelerates.

**Start**

- Rival has a controlled possession/carry/bounce-control state;
- ball is inside a calibrated car-relative flick envelope.

**Progress**

- an actual dodge begins;
- the active rotating/translating hitbox contacts the ball.

**Complete**

- the contact produces a meaningful ball velocity/trajectory change;
- the ball exits the prior control envelope as a deliberate release.

The exact flick subtype is then classified from dodge direction, car orientation,
ball-local contact geometry, rotational contact-point velocity, and release
trajectory.

### 8.2 Front / diagonal / side / 45 / 90 / 180 flick classes

Do not classify these from stick labels alone.

Record at dodge onset:

- car forward axis;
- car-to-ball vector in car-local coordinates;
- dodge impulse/flip-torque direction;
- car heading relative to attack direction;
- contact point and post-contact ball velocity.

The subtype is determined by the actual release geometry and orientation change.
A successful subtype receives one flick-mechanics completion, not `base flick +
angle label + another flick reward`.

The distinct subtype label remains important because launch direction,
recoverability, setup time, and defender readability differ substantially.

### 8.3 Delayed flick

A delayed flick is a tactical timing variant and must remain visible in telemetry.

**Start**

- Rival establishes possession and a plausible flick setup.

**Progress**

- Rival preserves possession through a measurable release delay while car
  orientation/position changes enough that an immediate flick could reasonably
  have occurred earlier;
- opponent reaction during this interval is recorded.

**Complete**

- a valid flick later releases the ball.

The delay itself is **not directly rewarded** in V0.1. Natural play should teach
whether delaying forced a bad challenge, opened a lane, or harmed the attack.

## 9. Musty-class mechanics

A Musty is treated as a distinct flick subtype because the scoop physics and
setup geometry create capabilities that ordinary flicks do not.

### 9.1 Musty-class physical invariant

The invariant must work from ground, air, wall, ceiling, catch, or reset states.
Therefore none of those entry states define the Musty.

**Start**

- Rival has controlled ball proximity or a plausible direct Musty contact setup;
- car orientation places the roof/rear/nose sweep path in a position where a
  backward dodge can pass through the ball.

**Progress**

- actual backward dodge begins;
- car rotates through the characteristic nose-down / inverted release region;
- derive the exact hitbox contact-point velocity
  `v_contact = v_car + cross(omega, r)`;
- the rotational component `cross(omega,r)` must contribute materially to the
  closing velocity between the active car surface and the ball.

**Complete**

- the moving hitbox surface sweeps/scoops through the ball;
- retained car-ball impulse and ball `delta_v` prove a real momentum transfer;
- resulting ball velocity gains a meaningful forward/upward or otherwise useful
  release component attributable to the scoop.

This detector intentionally does not require a particular starting surface.

### 9.2 Musty setup/contact variants

These variants should be separately labeled because they change the play even
when the terminal scoop is the same family.

#### Standard catch / roof catch

- ball enters or remains in the upper roof/rear control region;
- Rival matches enough ball motion to conceal the release and keep contact
  geometry stable before the backward-dodge scoop.

Telemetry:

- catch duration;
- relative speed during catch;
- defender-relative car/ball occlusion;
- release latency;
- launch vector.

#### Catch-pop Musty

- a low/controlled catch is followed by an initial upward displacement/pop;
- Rival remains attached to the play;
- the later Musty-class scoop releases the ball.

The pop and Musty can be distinct physical completions if a future reward contract
explicitly authorizes both; they must not be conflated merely because they occur
in one sequence.

#### Nose catch / nose-loaded Musty

- ball is controlled closer to the car's forward/nose region rather than the
  center/rear roof zone;
- pre-release contact geometry stores the ball in a position where the Musty
  sweep produces a different forward launch profile.

Telemetry must retain the car-local ball position and the resulting launch
vector/speed so this variant is not lost inside a generic `Musty` label.

#### Hidden/occluded Musty setup

This is a tactical signature, not a separate mechanics reward.

Record whether, from the defender's position, the ball substantially overlaps or
occludes the attacker's chassis/rotation during the setup. Also retain opponent
challenge onset relative to release.

The system should later be able to answer whether Rival learned that some Musty
setups are harder to read. V0.1 does **not** reward `being hidden` directly.

### 9.3 Musty after reset / ceiling / wall

These retain both labels:

- origin/resource label (`flip_reset`, `ceiling_retained_dodge`, `wall setup`);
- terminal `musty_class_flick` label.

The underlying resource acquisition and later scoop are separate physical events
and should remain auditable as such.

## 10. Breezi-class mechanics

A Breezi is **not collapsed into a generic Musty label**. Its terminal release is
Musty-class, but the rotational setup creates a distinct timing and information
signature.

### 10.1 Breezi physical sequence

**Start**

- Rival has controlled ball proximity/possession;
- car leaves ordinary support as needed for the setup.

**Progress**

- car executes a sustained combined roll/yaw rotation that moves through a
  tornado-like orientation path;
- nose moves upward through the setup phase;
- car rotates/settles into the nose-down / roof-to-ball geometry required for a
  Musty-class scoop;
- possession/control is not lost before release.

**Complete**

- a valid Musty-class backward-dodge scoop releases the ball after the Breezi
  setup path.

### 10.2 Breezi delay signature

Retain:

- time from setup start to final dodge release;
- periods of low angular velocity / flip-cancel pause if present;
- total orientation path length;
- time the ball remains controlled while the release is delayed;
- defender closing speed and challenge onset;
- whether defender jumps/commits before actual release;
- resulting shot direction and speed.

A delayed Breezi may therefore become tactically valuable because the release
occurs later than the defender expects. That value should emerge from opponent
interaction and game return, not from a hand-coded `delay bonus`.

A Breezi completion receives one mechanics-family completion for the Breezi
sequence; do not separately pay `generic flick + Musty + Breezi` for the same
terminal release unless a future contract deliberately authorizes a separate
physical event earlier in the compound sequence.

## 11. Redirect and rebound mechanics

### 11.1 Redirect

**Start**

- ball arrives with meaningful pre-contact velocity;
- Rival approaches from a trajectory that can materially change the ball's
  direction.

**Complete**

- legitimate Rival contact occurs;
- post-contact ball velocity changes direction by more than a calibrated minimum
  while retaining enough speed to represent a real redirect rather than a dead
  touch;
- retain projection toward opponent goal/attack space as telemetry and normal
  game reward, not as the mechanic's sole identity.

### 11.2 Double tap / double touch

**Start**

- Rival contact sends or continues the ball toward a backboard/world rebound
  region.

**Progress**

- ball makes a world contact on the opponent backboard region;
- Rival remains airborne/in pursuit;
- opponent does not take possession.

**Complete**

- Rival makes the next legitimate touch after the rebound and materially
  redirects the ball away from the board.

A goal, if produced, remains independently rewarded by the ordinary goal reward.

### 11.3 Triple/extended rebound touch

Record additional controlled rebound touches, but cap the mechanics-family payout
rather than paying indefinitely.

## 12. Pinches

### 12.1 Common pinch topology

**Start**

- ball has or is entering a world-surface manifold;
- Rival is entering car-ball contact from the opposing side.

**Progress**

- car-ball and ball-world contacts overlap in the same tick or a narrowly
  calibrated collision window;
- car-ball contact normal and ball-world normal indicate compression of the ball
  between the car and surface;
- retain both manifolds and impulses.

**Complete**

- compression releases;
- post-event ball speed/trajectory changes materially compared with pre-contact
  state.

The resulting ball acceleration must be real; mere simultaneous proximity does
not count.

### 12.2 Surface variants

Classify from authoritative world geometry/contact normals:

- `ground_pinch`;
- `wall_pinch` / Kuxir-style wall pinch;
- `ceiling_pinch`;
- `post_pinch`;
- corner/Astral-style geometry when identifiable.

Each is one pinch completion for the same compression event; the surface subtype
remains telemetry because launch angles and tactical uses differ.

## 13. Pogo mechanics

### 13.1 Pogo physical invariant

**Start**

- car is airborne and approaching a world surface with a chassis corner/edge
  rather than an ordinary stable multi-wheel landing.

**Progress**

- chassis/world manifold proves the corner/edge impact;
- surface-normal velocity is toward the surface before impact;
- contact impulse reverses or strongly redirects that surface-normal component;
- car does not immediately settle into normal grounded support.

**Complete**

- car separates from the surface with meaningful rebound velocity while retaining
  useful control/motion.

The reward must lock out until a new airborne approach occurs so repeated contact
jitter cannot farm pogo rewards.

### 13.2 Pogo with ball / disguised pogo

Retain whether:

- Rival had recent possession or Musty-like setup;
- ball and car reach the surface in the same short window;
- the ball occludes the car's surface approach from the defender;
- the rebound leads directly to another controlled touch.

This is tactically important because a pogo can emerge from a setup that also
looks capable of producing a flick or catch. V0.1 does not reward `surprise` or
occlusion directly; it records the conditions so self-play can assign value to
that choice through outcomes.

### 13.3 Reverse/ceiling/wall pogo labels

Classify from impacted hitbox corner and surface normal. Same physical rebound
family; subtype remains available for policy analysis.

## 14. Stall-assisted mechanics

A bare stall is not automatically valuable enough to reward because Rival could
learn to consume flips for no purpose.

Record actual near-zero translational dodge/flip-torque stall events, but reward
only when the stall participates in a completed useful sequence such as:

- stall -> valid flip reset;
- stall -> controlled aerial ball touch;
- stall -> another source-backed resource/control transition.

The stall remains a distinct telemetry event so a future contract can revisit it
if Rival discovers a repeatably useful standalone application.

## 15. Mechanics intentionally telemetry-only in V0.1

Do **not** add direct mechanics reward merely for:

- ordinary jump;
- ordinary dodge/flip;
- generic aerial state;
- powerslide;
- driving backward;
- being on wall/ceiling;
- boost button use;
- generic recovery/landing;
- ordinary approach-to-ball;
- being hidden behind the ball;
- delaying a shot;
- opponent committing early.

Some are already indirectly rewarded by Gameplay V1/V2; others are tactical
context variables. If telemetry later shows a real learning bottleneck, they can
be reconsidered under a new contract.

## 16. How tactical use should be learned

The intended training architecture is:

1. **small mechanics completion signal** keeps a successfully discovered
   capability in Rival's behavioral repertoire;
2. **normal gameplay reward** remains dominant;
3. **opponent/self-play curriculum** supplies the circumstances in which one
   variant is better than another;
4. PPO can then increase the probability of the successful variant in the
   opponent-conditioned states where it produces better return.

For example, if Rival can execute both an immediate flick and a delayed Breezi,
self-play can in principle learn:

- immediate flick when the defender is already close and delay is dangerous;
- delayed Breezi when the defender is shadowing or commits to the expected early
  release;
- Musty from a ball-occluding setup when the defender has poor visibility;
- nose-catch launch when a fast forward release is available;
- catch/pop or continued possession when the defender refuses to commit;
- pogo continuation when the ground approach and ball state make it viable.

The hard part is exploration and credit assignment, not a fundamental inability
to learn conditional use. The reason for preserving subtype/context telemetry is
to prove whether that learning actually happens instead of assuming it did.

No direct `deception reward` is proposed in V0.1. If natural gameplay return
cannot teach tactical selection after the mechanics themselves are established,
we should first consider state-distribution/curriculum changes or an explicit
higher-level skill-selection architecture rather than adding arbitrary deception
coefficients.

## 17. Calibration work required before implementation

Source-exact/discrete events can be implemented with little or no subjective
thresholding:

- actual dodge onset;
- wheel masks/contact-body identity;
- grounded threshold;
- flip/double-jump resource transitions;
- ball/world manifold identity;
- existing dash/zap/double-dash timing evidence.

Continuous-motion classifiers require calibration from positive and negative
traces before reward activation:

- speedflip cancellation/alignment;
- half-flip reversal;
- possession/control envelope;
- minimum meaningful redirect angle/speed;
- Musty rotational-scoop contribution;
- Breezi orientation-path/delay thresholds;
- pinch simultaneous-contact window/compression threshold;
- pogo corner-contact/rebound threshold;
- tactical occlusion descriptors.

For each such detector, create a compact corpus containing:

- clear positives;
- near-miss negatives;
- ordinary actions likely to be false positives;
- representative floor/wall/ceiling/orientation variants where applicable.

Tune the narrowest thresholds that separate the physical event. Keep raw evidence
for every classified event so the labels can be audited later.

Do not turn this calibration into a broad simulator acceptance benchmark.

## 18. Research/source notes

Primary simulator authority:

- `ZealanL/RocketSim@c2baacb8f4b441dd8505e63c2aeb5a1679b60b02`
- RivalSim's accepted resident mechanics and collision paths.

Existing Rival evidence:

- `results/rival2/gameplay_v1_nexto/dash_mechanics_contract.json`
- `rivalsim/nexto_short_eval.py`

Useful external mechanic descriptions used for naming/setup cross-checks:

- Rocket League Mechanics Database:
  `https://0byte-coding.github.io/rocket_league_mechanics/`
- Dignitas flick guide with Joreuz, including Breezi setup/use:
  `https://dignitas.gg/articles/a-rocket-league-guide-to-flicks-with-joreuz`
- Psyonix/Rocket League flip-reset indicator notes (three-or-more-wheel reset):
  `https://www.rocketleague.com/news/rocket-league-patch-notes-v2-66`
- GamePress Musty mechanics discussion of the scoop/trajectory:
  `https://www.gamepressunited.com/rocket-league-blog/musty-flick-tutorial/`

External community/tutorial descriptions are not simulator authority. They are
used to identify the human concept and expected qualitative behavior; the reward
detectors must still be grounded in RivalSim/RocketSim physics evidence.

## 19. V0.1 catalog summary

Reward-candidate families currently documented:

- wavedash / landing / wall / curve dash;
- Rival double dash;
- zapdash;
- speedflip;
- half-flip;
- generalized chain dash telemetry;
- ball flip reset;
- chain reset;
- pre-flip reset;
- pop reset;
- rapid reset;
- pancake/wall reset;
- car reset;
- ceiling-retained-dodge continuation;
- controlled 2-touch possession;
- controlled 3-touch possession;
- ground carry;
- bounce dribble;
- ground-to-air continuation;
- 2-touch air dribble;
- 3-touch air dribble;
- ordinary directional flick classes;
- delayed flick telemetry;
- Musty-class flick plus catch/nose/pop/origin variants;
- Breezi-class flick plus delay signature;
- redirect;
- double tap / extended rebound touch;
- ground/wall/ceiling/post pinch;
- pogo and surface/corner variants;
- stall-assisted useful sequences.

This catalog should expand only when a candidate mechanic can be expressed as an
auditable physical state transition or a useful tactical subtype of one of those
transitions.
