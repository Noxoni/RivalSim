# Rival 2.0 Mechanics Reward Contract — Flick Variant Appendix V0.1

Status: **research/design draft**

Parent working contract:

`docs/RIVAL2_MECHANICS_REWARD_CONTRACT_V0_1.md`

This appendix expands the flick-family portion of the parent contract. It exists because useful flicks can share a broad terminal event while differing materially in setup geometry, release timing, concealment, launch vector, recovery, and opponent readability. Those distinctions must not be erased merely because the mechanics reward bracket is intentionally small.

This document does **not** authorize changing the active Rival training reward. It defines future detector semantics and calibration targets only.

## 1. Core rule: same reward bracket does not mean same mechanic

A future mechanics reward may assign the same low completion value to multiple flick variants. That is a statement about **reward scale**, not identity.

For example, a successful 45-degree flick, Mawkzy-style power flick, Musty, Breezi, Classy flick, Suffo flick, or Wizard/Evoh-style late-contact flick may all receive one mechanics-tier completion payout when authorized. They must still retain different physical labels and telemetry because the game state in which each is useful can differ dramatically.

The detector therefore has two outputs:

1. **mechanical completion** — did Rival actually execute a useful physical capability?;
2. **variant/tactical signature** — what physical setup, timing, visibility, release, and recovery characteristics did that execution have?

Only the first is directly eligible for the low mechanics reward. The second remains observable state from which ordinary self-play/game reward can teach selection.

## 2. Shared flick physics

### 2.1 Possession prerequisite

A controlled flick starts from a possession/catch state rather than a random collision.

A possession epoch is valid when:

- Rival owns the most recent legitimate ball contact;
- the opponent has not touched the ball since;
- ball position remains inside a calibrated reachable/control envelope around Rival;
- relative car-ball velocity remains compatible with continued control rather than the ball simply passing through the envelope;
- the sequence has not exceeded the calibrated control-gap timeout.

A roof carry, bounce dribble, soft catch, nose catch, and controlled pop may all satisfy possession through different contact patterns.

### 2.2 Actual release

A flick cannot complete merely because Rival dodged while near the ball.

Completion requires:

- an actual dodge/flip state transition;
- car-ball contact during the physically relevant part of that dodge sequence;
- measurable ball momentum/trajectory change;
- the ball exits the pre-release control state into a new launch trajectory.

Retain:

- car pose and local axes at dodge onset;
- ball position in car-local coordinates;
- actual flip-relative torque and dodge impulse direction;
- pre-contact car linear/angular velocity;
- pre-contact ball linear/angular velocity;
- car-ball contact point and normal;
- normal/tangent/push impulses;
- RocketSim extra-hit velocity;
- post-contact ball velocity;
- car pose/velocity after release.

### 2.3 Contact-point velocity

For car linear velocity `v_car`, angular velocity `omega`, and contact offset `r` from the car center of mass:

`v_contact = v_car + cross(omega, r)`

and

`v_rot = cross(omega, r)`

At the contact normal `n`, derive the translational and rotational closing components separately. This lets the detector distinguish a ball struck because the entire car translated into it from a ball struck because a rotating surface of the hitbox swept through it.

For scoop-style flicks, this decomposition is mandatory.

### 2.4 Reward de-duplication

One terminal release receives one flick-family mechanics payout even if several nested labels describe it.

Examples:

- `45_flick + mawkzy_signature` -> one flick payout;
- `musty_class + roof_catch_origin + hidden_setup` -> one Musty/flick payout;
- `musty_class + breezi_setup` -> one Breezi/flick payout for the terminal release;
- `generic_flick + delayed_release` -> one flick payout.

Distinct earlier physical accomplishments may remain separately rewardable under a future contract. Example: a real flip reset is acquired and later that resource is consumed in a Musty. The reset and later flick are separate state transitions.

## 3. Catch and setup descriptors

These descriptors are important because the same terminal flick can begin from very different control states.

### 3.1 Soft catch / control acquisition

A soft catch occurs when an incoming ball contact converts a relatively uncontrolled trajectory into a controllable one.

Physical signature:

- ball arrives with meaningful velocity relative to Rival;
- Rival contact reduces relative car-ball separation speed rather than simply accelerating the ball away;
- post-contact ball remains inside the control envelope for the required confirmation window;
- no opponent contact interrupts the confirmation.

Telemetry:

- relative speed before/after catch;
- car-local contact position;
- ball vertical velocity before/after;
- resulting possession duration;
- defender distance/challenge state.

This may become a separately rewardable ball-control capability later, but this appendix uses it primarily as flick-origin state.

### 3.2 Roof/rear catch

Ball is controlled in the upper/rear roof region with low enough relative motion that Rival can delay release or transition into a backward-dodge scoop.

Retain:

- ball car-local `x/y/z`;
- contact duration or repeated-touch cadence;
- relative tangential motion across the roof;
- release latency;
- defender visibility/occlusion.

### 3.3 Nose catch / nose-loaded control

Ball is controlled materially farther forward on the car than a normal roof carry.

The detector should not use a guessed fixed world position. Classify from ball position in the car's local frame relative to the calibrated control envelope.

This state is tactically important because a release can produce a flatter/faster forward trajectory and because the attacker's chassis can remain hidden behind the ball from some defender viewpoints.

### 3.4 Catch-pop

A catch-pop is a controlled state transition rather than merely positive ball `z` velocity.

Physical sequence:

- Rival establishes a catch/control state;
- a subsequent distinct Rival action/contact adds an upward component to ball velocity;
- Rival remains close enough and appropriately moving to continue the play;
- possession is not immediately surrendered.

A later Musty, reset, air dribble, or ordinary flick receives its own label. The pop is an origin/transition descriptor unless a later reward contract explicitly authorizes it as a separate completion.

## 4. Directional flick family

### 4.1 Front flick

Physical sequence:

- possession/carry state;
- actual predominantly forward dodge;
- rotating/front portion of the hitbox contacts the ball during the dodge;
- ball receives a meaningful forward/upward release.

Tactical signature:

- very short setup/release latency;
- comparatively small orientation tell;
- immediate recovery direction.

Do not reward short latency itself.

### 4.2 Side/diagonal flick

Classify from actual dodge direction and ball/contact geometry, not controller labels.

Physical sequence:

- possession;
- actual lateral/diagonal dodge;
- active dodge contact releases the ball;
- post-contact launch has the corresponding lateral/diagonal component.

### 4.3 45-degree flick

The `45` label describes the approximate car/ball release geometry, not a requirement for exactly 45.000 degrees.

Physical signature:

- possession;
- Rival yaws/positions off the direct attack axis before release;
- diagonal/side-biased dodge contacts the ball;
- launch direction reflects the offset geometry and produces a useful shot/pass trajectory.

Calibration should derive an angle band from positive traces rather than hard-code the community name as an exact threshold.

### 4.4 90/180/270 rotational flick labels

These are orientation-path subclasses of the same controlled release family.

Retain:

- cumulative yaw/orientation change from possession reference to release;
- dodge direction;
- contact point;
- launch vector;
- setup time;
- recovery heading.

Their tactical difference is largely setup time, release angle, power, and defender readability. They should not stack multiple flick rewards.

## 5. Mawkzy-style power flick

This is preserved separately from a generic 45-degree label because the useful physical signature is not merely `yaw about 45 degrees`.

### 5.1 Physical invariant

**Start**

- stable dribble/carry possession;
- Rival begins a rapid off-axis release setup.

**Progress**

- car orientation places more of the hitbox **behind** the ball rather than directly beneath it;
- actual dodge is strongly backward-diagonal relative to the car's local frame, closer to a backflip power transfer than an ordinary side-flip 45 release;
- roll/yaw positioning permits that backward-biased dodge surface to pass through the ball without requiring the long setup of a full 180 flick.

**Complete**

- dodge-driven hitbox motion produces a real high-energy ball release;
- launch is comparatively forward/flat rather than only a high arc;
- Rival retains or quickly regains a forward-useful recovery orientation.

Do **not** require a particular air-roll button input. The detector uses the resulting orientation, dodge direction, contact geometry, and release.

### 5.2 Tactical signature

Retain:

- release latency from stable dribble;
- ball launch speed/elevation;
- defender distance at release;
- car forward-axis recovery time;
- whether the defender challenged before release;
- ball/chassis occlusion.

The fast, flat power and recovery profile is what later gameplay may learn to value.

## 6. JZR-style high-gain 45 flick

Treat this as a distinct 45-family physical signature rather than a synonym.

**Start**

- possession and off-axis 45-like setup.

**Progress**

- side/diagonal dodge begins;
- flip cancellation/reorientation changes the later hitbox path relative to an uncancelled 45 flick;
- contact occurs while the car is positioned to add a strong vertical/high-forward component.

**Complete**

- valid controlled flick release;
- ball launch has materially higher elevation/gain than the baseline 45-family trace while retaining useful speed.

Retain backboard-intercept likelihood and subsequent double-tap opportunity as telemetry, not as a requirement or extra reward.

The exact flip-cancel/orientation thresholds require trace calibration.

## 7. Musty-class flick

The parent contract's Musty invariant remains authoritative. This appendix sharpens the scoop measurement.

### 7.1 Physical invariant

**Start**

- controlled ball proximity or a direct plausible Musty setup;
- the ball lies in the future sweep volume of the roof/rear/nose surface during a backward dodge.

**Progress**

- actual backward dodge begins;
- car rotates through the release geometry;
- compute `v_contact`, `v_rot`, and ball-relative contact velocity;
- the rotational component must contribute materially to closing speed along the ball-contact normal;
- where contact persists across ticks, retain the car-local path of the ball/contact point across the hitbox.

**Complete**

- the rotating surface sweeps/scoops through the ball;
- solver impulse and `delta_v_ball` prove meaningful momentum transfer;
- release produces a useful new ball trajectory.

A simple backflip collision where translation dominates and the rotating surface does not materially drive the contact is **not** a Musty completion.

### 7.2 Clean scoop vs slap/bonk

Retain a `scoop_quality` diagnostic derived from:

- rotational share of contact-normal closing velocity;
- duration/path of moving-surface contact when present;
- continuity of ball-relative movement across the roof/rear/nose sweep;
- impulse direction;
- post-contact launch vector.

Do not convert this into a larger reward. It is for calibration and tactical analysis.

### 7.3 Entry-state labels

Preserve independently:

- ground-dribble Musty;
- roof/rear catch Musty;
- nose-loaded Musty;
- catch-pop Musty;
- aerial Musty;
- wall-origin Musty;
- ceiling-origin Musty;
- reset-origin Musty;
- pogo-origin/continuation where physically valid.

These do not change the terminal Musty physics but materially change available timing, concealment, momentum, and follow-ups.

### 7.4 Hidden/occluded Musty

This is a tactical descriptor only.

From the defender viewpoint, measure whether the ball angularly overlaps enough of Rival's chassis/rotation to reduce direct visibility of the setup. Retain defender challenge onset and release latency.

Never reward occlusion directly. If it is useful, the resulting opponent errors/game outcomes should teach that.

### 7.5 Reverse Musty

Use the same rotational-scoop invariant but classify the release relative to Rival's pre-sequence travel/attack frame.

A reverse Musty must produce the opposite/backward-style release geometry rather than merely being a poorly aimed ordinary Musty.

## 8. Breezi-class flick

A Breezi is a compound orientation path ending in a Musty-class scoop.

### 8.1 Physical sequence

**Start**

- controlled ball state;
- Rival begins the rotational setup without losing possession.

**Progress**

- sustained combined roll/yaw angular motion produces the tornado-like path;
- car nose rises through the setup;
- orientation path then brings the car toward the nose-down/roof-to-ball geometry needed for the backward-dodge scoop;
- ball remains in the controllable future sweep region.

**Complete**

- valid Musty-class scoop releases the ball after that orientation path.

### 8.2 Delay/readability signature

Retain:

- time from first setup rotation to release;
- integrated orientation-path length;
- pauses/slowdowns in angular motion;
- defender closing speed;
- defender jump/dodge/challenge onset;
- whether defender commits before the actual release;
- ball/chassis occlusion;
- launch vector and speed.

The longer and less obvious release path is a core tactical difference from an immediate Musty/45 flick, but it receives no hand-coded deception bonus.

## 9. Classy/tornado-inverted backflip flick

Keep this distinct from Breezi even though both involve air-roll/tornado-like setup and a backward final release.

### 9.1 Physical sequence

**Start**

- controlled dribble/carry.

**Progress**

- Rival jumps while maintaining ball control;
- combined roll/yaw moves the car into a clearly inverted/near-inverted orientation beneath/behind the ball;
- unlike a generic 180 yaw flick, the orientation path contains substantial roll/inversion;
- actual backward dodge begins from that inverted setup.

**Complete**

- backward-dodge contact produces a valid flick release with meaningful ball momentum transfer.

### 9.2 Tactical signature

Retain:

- setup duration;
- inversion duration before release;
- defender reaction timing;
- launch power/elevation;
- recovery heading.

The setup can look like continued dribble or an aerial transition before the release. That information difference is measured, not directly rewarded.

## 10. Suffo-style fake-away flick

This variant is mechanically important because the setup deliberately increases apparent separation before the terminal flick.

### 10.1 Physical sequence

**Start**

- Rival has controlled possession and a plausible immediate flick state.

**Progress**

- Rival jumps/moves its chassis away from the ball enough that an observer could reasonably interpret the initial release as a miss/fake;
- possession is not actually lost beyond the calibrated recoverable envelope;
- actual diagonal-backward dodge then reverses the relevant contact-point motion toward the ball.

**Complete**

- late returning hitbox contact releases the ball with meaningful speed/direction change.

A normal delayed flick where Rival simply waits under the ball is not automatically a Suffo. The signature requires the initial away motion followed by a dodge-driven return contact.

### 10.2 Tactical signature

Retain opponent challenge onset during the apparent miss/fake interval. No reward is paid for fooling the opponent directly.

## 11. Wizard / Evoh-style late-dodge underside flick

Names for this family vary; the physics detector should therefore use a neutral internal label such as `late_dodge_underball_flick` and preserve community aliases only for display.

### 11.1 Physical sequence

**Start**

- controlled dribble/carry;
- actual dodge begins in a trajectory that initially carries the main hitbox path under/past the ball rather than immediately releasing it.

**Progress**

- the ball remains near enough to the later flip path for a second/late contact;
- car continues through the active dodge rotation;
- the underside/front/nose region later re-enters the ball's path.

**Complete**

- a late-phase dodge contact produces the actual ball release;
- contact creates meaningful upward/forward velocity change rather than incidental grazing.

The defining property is **late contact from the continuing dodge path after the apparent first release opportunity has passed**.

### 11.2 Tactical signature

Retain:

- latency from dodge onset to ball-release contact;
- whether there was an earlier non-releasing/grazing contact;
- defender reaction between dodge onset and actual release;
- launch height/speed.

## 12. Bismillah/inverted front-scoop family

This family is useful to retain if Rival discovers it because its terminal impulse geometry differs from the backward-dodge Musty family.

**Start**

- controlled ball state;
- Rival rotates into an inverted/near-inverted orientation beneath the ball.

**Progress**

- actual forward-biased dodge begins from the inverted state;
- a front/corner portion of the rotating hitbox sweeps upward/through the ball.

**Complete**

- rotational contact contributes materially to the ball impulse;
- ball is released upward/forward with meaningful momentum.

Keep this separate from Musty because the dodge direction and active hitbox sweep are physically different.

## 13. Variant selection is learned from experience, not directly rewarded

Once Rival can physically execute these variants, the important problem becomes **which release is useful in the current opponent state**.

For every flick-capable possession state, retain enough telemetry to reconstruct:

- defender position and velocity;
- defender distance to ball and Rival;
- defender challenge/jump/dodge onset;
- defender heading and whether retreating/shadowing/committing;
- Rival-ball-defender angular geometry;
- ball occlusion of Rival's chassis from the defender viewpoint;
- time since possession was established;
- available boost;
- Rival dodge-resource state;
- goal angle/open-net geometry;
- chosen flick variant;
- release latency;
- launch speed/elevation/azimuth;
- immediate possession outcome;
- next-touch ownership;
- goal/save/concede outcome.

This allows self-play to discover relationships such as:

- immediate front/45 release is useful when the defender is already close;
- a Mawkzy-style release can produce flatter power without the full delay of a 180;
- Breezi/Classy setup may be worth the extra time when the defender is shadowing or commits early;
- a Suffo-style away motion can punish an early challenge;
- hidden Musty setup may reduce the defender's visual information;
- a nose-loaded catch may favor a fast forward launch;
- catch-pop may preserve options when the defender refuses to commit;
- a high-gain JZR-style release may intentionally create a backboard follow-up.

These are hypotheses to be learned and measured. They are **not hand-coded reward conditions**.

## 14. Calibration requirements

Do not activate these variant classifiers from guessed thresholds.

For each continuous detector, build a compact positive/negative trace corpus from authoritative physics state.

Required calibration sets include:

- ordinary front/diagonal/45/90/180 flicks;
- Mawkzy-like power traces vs ordinary 45 traces;
- Musty clean scoops vs backflip bonks/slaps;
- Breezi vs ordinary Musty vs Classy orientation paths;
- Suffo fake-away vs ordinary delayed flick;
- Wizard/Evoh late-contact vs ordinary pre-flip ball contact;
- JZR high-gain vs ordinary 45;
- catch/roof/nose/pop setup variants.

Tune only the narrowest thresholds required to separate the physical event. Retain raw evidence for every classification so later false positives/negatives can be audited.

## 15. Source notes

Primary simulator authority remains:

- `ZealanL/RocketSim@c2baacb8f4b441dd8505e63c2aeb5a1679b60b02`;
- RivalSim's accepted resident vehicle, wheel, car-ball, and ball-world physics paths.

External concept references used to identify human mechanic meanings and tactical distinctions:

- Rocket League Mechanics Database: `https://0byte-coding.github.io/rocket_league_mechanics/`;
- Dignitas/Joreuz flick guide: `https://dignitas.gg/articles/a-rocket-league-guide-to-flicks-with-joreuz`;
- Dignitas recent flick surveys for Classy/Suffo/180 behavior;
- Rocket Science shot-power/flick analysis: `https://rocketscience.fyi/know/videos/shot-power`;
- Musty tutorial/discussion sources for roof-to-ball scoop geometry.

External naming/tutorial sources are not simulator authority. The physical detector definitions above are the working contract.