# Rival 2.0 Mechanics Reward Contract — Detector Physics Appendix V0.1

Status: **research/design draft**

Parent documents:

- `docs/RIVAL2_MECHANICS_REWARD_CONTRACT_V0_1.md`
- `docs/RIVAL2_MECHANICS_FLICK_VARIANTS_V0_1.md`

This appendix converts the reward-candidate mechanic descriptions into a common set of physics observables, derived quantities, completion rules, and lockouts. It does **not** authorize a reward change or training run.

The goal is to make future reward events auditable from authoritative RivalSim state instead of controller recipes or visual guesses.

## 1. Global detector rules

### 1.1 Physics tick authority

Mechanic classification runs conceptually at the 120 Hz physics boundary. Policy decisions may remain at 30 Hz, but no mechanic whose identity depends on wheel/contact/flip timing may be inferred only from 30 Hz action samples.

### 1.2 Inputs are evidence, not identity

Controller state can be retained for audit and for cases where the engine state transition itself contains the result of the input. The primary detector authority is:

- actual jump/dodge state transition;
- wheel/body contact identity;
- rigid-body pose and velocity;
- contact manifolds and impulses;
- dodge-resource state;
- ball trajectory before/after contact.

### 1.3 Common state machine

Every rewardable detector follows:

`IDLE -> START -> PROGRESS -> COMPLETE -> LOCKOUT -> IDLE`

A `COMPLETE` event emits once. Lockout ends only after the physical state has genuinely left the completed sequence and a new attempt can begin.

### 1.4 Hierarchical de-duplication

One physical transition may receive several labels but only one payout within its mechanics family.

Examples:

- one dash can be `wavedash + landing_wavedash + wall_dash`;
- one reset can be `flip_reset + preflip_reset + wall_origin`;
- one flick can be `45_flick + mawkzy_signature`.

Separate later physical accomplishments can still complete separately. A reset acquisition followed by a Musty release is two real state transitions.

## 2. Shared authoritative observables

Use existing RivalSim resident fields where already available.

### 2.1 Car

Per physics tick retain or make directly available:

- `car_pos`, `car_quat`;
- `car_vel`, `car_ang_vel`;
- local `forward`, `right`, `up` axes;
- `on_ground`;
- per-wheel contact bit/mask;
- per-wheel hit point and hit normal;
- per-wheel suspension length and velocity;
- per-wheel contacted-body identity: world / ball / other car;
- chassis-world manifold point/normal/impulse;
- `has_jumped`, `is_jumping`;
- `has_double_jumped`;
- `has_flipped`, `is_flipping`;
- `jump_time`, `air_time`, `air_time_since_jump`, `flip_time`;
- actual flip-relative torque and actual flip onset.

### 2.2 Ball

Retain:

- position/orientation;
- linear/angular velocity;
- ball-world contact count and manifolds;
- world face/mesh/surface identity;
- contact normal/tangent/impulse.

### 2.3 Car-ball

Retain:

- unique contact onset and persistent contact state;
- car-side/ball-side contact points;
- contact normal/tangent;
- normal/tangent/push impulses;
- pre-contact car/ball linear and angular velocity;
- post-contact ball velocity;
- RocketSim extra-hit velocity;
- relative contact position on the ball.

## 3. Common derived quantities

### 3.1 Surface-relative tangent velocity

For unit surface normal `n` and velocity `v`:

`v_tangent = v - n * dot(v, n)`

`tangent_speed = length(v_tangent)`

Use this for floor, wall, ceiling, and curved dash outcomes instead of world-XY speed.

### 3.2 Contact-point velocity

At local/world contact offset `r` from the car center of mass:

`v_contact = v_car + cross(omega_car, r)`

`v_rot = cross(omega_car, r)`

This is required for scoop/flick, pogo, and some pinch analyses.

### 3.3 Relative contact closing velocity

Choose/contact-normal orientation consistently from car toward ball or support body toward moving body. Then derive:

`v_rel_contact = v_contact_moving - v_contact_other`

and the normal closing component:

`closing_n = max(0, -dot(v_rel_contact, n))`

The sign convention must be validated once against stored contact normals; individual detectors must not invent different conventions.

### 3.4 Ball transfer

At a discrete car-ball contact onset:

`delta_v_ball = v_ball_after - v_ball_before`

Retain:

- magnitude;
- direction;
- projection on contact normal;
- projection toward opponent goal/attack direction for telemetry only;
- solver impulses;
- extra-hit velocity.

### 3.5 Surface-normal rebound

For a car contact point approaching a world surface:

`normal_speed_before = dot(v_contact_before, n_surface)`

`normal_speed_after = dot(v_contact_after, n_surface)`

A genuine rebound requires the pre/post signs to show motion into then away from the surface after orienting `n_surface` consistently.

### 3.6 Dodge-resource state

Expose one discrete resource state per car:

- `GROUNDED_AVAILABLE`;
- `AIR_TIMED_AVAILABLE` — airborne after a real jump with an unconsumed dodge still inside its source timing semantics;
- `AIR_UNTIMED_AVAILABLE` — airborne without `has_jumped`, with flip/double-jump unconsumed;
- `AIR_CONSUMED`.

Transitions into `AIR_UNTIMED_AVAILABLE` from a previously worse resource state are the key reset-acquisition signal.

### 3.7 Possession epoch

Maintain one owner plus continuity state for the ball.

Epoch begins on a legitimate unique touch and ends on:

- opponent legitimate touch;
- goal/reset;
- possession owner demo/reset that destroys continuity;
- ball leaves a calibrated reachable/control envelope for longer than its allowed gap;
- time/separation exceeds the calibrated continuity window.

Persistent manifold chatter is not a new touch.

## 4. Dash/movement detector family

### 4.1 Successful wavedash

**START**

- zero wheel contact;
- actual directional dodge begins.

**PROGRESS**

- first wheel/surface contact arrives in the researched short landing relation while dodge motion remains relevant.

**COMPLETE**

- measured surface-tangent motion after the landing sample is usefully greater than the retained pre-dodge tangent motion;
- actual dodge, landing mask, normal, and pre/post velocities remain attached as evidence.

**LOCKOUT**

- until the car leaves the completed landing/dodge state and a new airborne dodge attempt begins.

Subtype from contacted surface:

- floor/landing wavedash;
- wall dash;
- curved-transition dash;
- ceiling/surface variant if observed.

One physical dash -> one payout.

### 4.2 Rival double dash

Use the project's observed definition, not an external naming restriction.

**START**: first successful dash completion.

**PROGRESS**:

- intervening wheel support/contact occurs;
- second successful dash begins inside the established short sequence window.

**COMPLETE**:

- second dash completes;
- post-second surface-tangent speed is greater than pre-first tangent speed;
- both component events are retained.

Fresh jump state between components is telemetry, not automatic rejection.

### 4.3 Zapdash

**START**:

- landing contact order reaches front wheels before rear wheels.

**PROGRESS**:

- car temporarily reaches source grounded jump availability while not yet flat/four-wheel settled;
- fresh jump occurs in that partial/angled support state;
- jump separates/pops the front;
- directional landing dodge/wavedash follows.

**COMPLETE**:

- final dash succeeds;
- sequence-level tangent speed increases.

Evidence must include wheel masks/order, suspension, jump onset, actual dodge onset, surface normal, and speed.

### 4.4 Speedflip

A speedflip detector must prove efficient dodge momentum **and** rapid orientation recovery.

**START**:

- forward-useful ground/support state;
- actual forward-diagonal dodge begins.

**PROGRESS**:

- pitch rotation is arrested/cancelled substantially earlier than an uncancelled diagonal flip reference;
- remaining roll/yaw/reorientation returns car forward axis toward travel;
- dodge translation is retained.

**COMPLETE**:

- forward axis and surface-tangent velocity become strongly aligned inside a short calibrated window;
- useful tangent/forward speed is preserved/increased rather than lost to the flip animation.

Do not require boost input. Retain boost force/use so calibration can distinguish speed gain caused purely by boost from efficient dodge execution.

### 4.5 Half-flip

**START**: actual backward dodge.

**PROGRESS**:

- backward pitch rotation is cancelled/arrested before normal backflip completion;
- roll/yaw continues a heading reversal.

**COMPLETE**:

- car forward axis is approximately opposite the pre-sequence forward axis;
- car is controllable/supported or imminently supported;
- velocity has useful projection along the new forward direction.

Do not reward merely rotating 180 degrees while stationary.

## 5. Flip-reset/resource detector family

### 5.1 Ball reset

This is source-exact enough to avoid a guessed visual classifier.

**START**:

- car is airborne;
- record pre-contact resource state.

**PROGRESS**:

- three or more wheel rays/contact states simultaneously support on the ball;
- contacted-body identity is the ball;
- source grounded logic clears consumed flip/double-jump state.

**COMPLETE**:

- car separates from the ball without initiating a fresh grounded jump;
- remains airborne;
- post-event resource state is `AIR_UNTIMED_AVAILABLE`;
- pre-event state was not already the same unchanged untimed resource.

This detects **resource acquisition**, not wheel proximity.

### 5.2 Chain reset

A second reset can begin only after the prior acquired resource has been consumed/lost or the first acquisition lockout has otherwise ended.

**COMPLETE** when a new distinct ball-wheel grounding event restores `AIR_UNTIMED_AVAILABLE` before world grounding.

### 5.3 Pre-flip reset

Subtype requirements:

- actual dodge/resource consumption before reset contact;
- valid later ball-reset completion.

The pre-flip subtype and base reset describe the same reacquisition event -> one reset-family payout.

### 5.4 Pop reset

Compound sequence:

1. valid reset acquisition;
2. car remains in a close controlled ball relation;
3. newly acquired dodge is consumed inside a short calibrated window;
4. that dodge produces a distinct controlled upward/outward ball transfer;
5. Rival remains able to continue the aerial play.

The resource acquisition and later pop are separate physical events. Future reward accounting may pay both if explicitly authorized and bounded.

### 5.5 Rapid reset

Subtype signature:

- consumed dodge drives a fast pre-flip/speedflip-like approach;
- valid reset reacquisition follows with short consumption-to-reacquisition latency.

Calibrate the latency/orientation threshold from traces rather than naming alone.

### 5.6 Pancake/wall-assisted reset

**START**:

- ball has active/near-active wall-like world manifold;
- Rival wheel support approaches from the opposite side.

**COMPLETE**:

- wheel-ball grounding restores untimed dodge resource while ball/world contact creates a compression state;
- retain ball speed/impulse change separately.

If the resource is not restored, it may still be a pinch but it is not a pancake reset.

### 5.7 Car reset

Same resource acquisition detector as ball reset except contacted-body identity is the other car.

### 5.8 Ceiling-retained dodge continuation

Leaving ceiling support without jumping naturally yields an untimed airborne dodge, so that departure alone does not pay.

Reward-candidate completion requires a meaningful use of that retained resource before ordinary world grounding, such as:

- retained-dodge ball contact;
- retained-dodge flick/redirect;
- another separately completed mechanic.

Keep ceiling origin as context telemetry.

## 6. Possession/repeated-contact detector family

### 6.1 Two-touch control

**START**: first legitimate unique Rival touch opens/continues Rival possession epoch.

**COMPLETE**:

- Rival produces a second distinct contact onset;
- opponent has not touched between them;
- possession continuity never crossed the calibrated loss condition.

This is one milestone payout.

### 6.2 Three-touch control

Same epoch reaches a third legitimate distinct contact.

After completion, lock the repeated-contact mechanics reward for the rest of that possession epoch. Touches 4+ are telemetry only unless a future contract explicitly adds another capped milestone.

### 6.3 Roof carry

**START**:

- ball occupies calibrated upper-car control region;
- car-ball support/manifold exists.

**PROGRESS**:

- low relative separation motion persists;
- no pinch-style world compression;
- Rival maintains world support appropriate to a ground carry.

**COMPLETE**:

- support/control persists through a meaningful time or meaningful change in car speed/direction.

One carry event -> one payout, never one reward per tick.

### 6.4 Bounce dribble

**START**: Rival possession plus ball-world ground bounce.

**PROGRESS**: Rival remains within the control envelope of the rebound.

**COMPLETE**: Rival takes the next legitimate controlled touch and continues the epoch; later bounce-touch cycles may build toward the three-touch milestone but cannot print unbounded reward.

## 7. Aerial-control detector family

### 7.1 Ground-to-air continuation

**START**:

- Rival ground possession/control;
- Rival contact produces a real positive upward ball transfer.

**PROGRESS**:

- Rival becomes airborne and pursues;
- opponent does not touch first.

**COMPLETE**:

- Rival obtains the next distinct aerial touch before Rival world landing.

The pop alone does not pay this event.

### 7.2 Two-touch air dribble

**START**: first distinct Rival touch while Rival is airborne, or an authorized ground-to-air transition.

**COMPLETE**:

- second distinct Rival touch while Rival remains airborne;
- opponent has not touched;
- aerial possession continuity is not broken.

### 7.3 Three-touch air dribble

Same aerial epoch reaches a third distinct Rival touch.

After third-touch completion, lock the repeated aerial-touch mechanics reward for that epoch. Additional touches remain valuable through ordinary gameplay and telemetry.

Boost use is telemetry. It is not required for the mechanic identity even if many real air dribbles use boost.

## 8. Redirect/rebound detector family

### 8.1 Redirect

**START**:

- ball has meaningful incoming velocity before Rival contact.

**COMPLETE**:

- Rival legitimate touch creates a material change in ball velocity direction;
- ball retains enough outgoing speed to be a redirect rather than a dead catch.

Compute:

- angle between normalized pre/post ball velocity;
- outgoing speed / incoming speed;
- projection toward attack/goal for telemetry only.

Thresholds must be calibrated.

### 8.2 Double tap / double touch

**START**:

- Rival touch sends/continues ball toward opponent backboard rebound region.

**PROGRESS**:

- ball makes authoritative world contact on opponent backboard region;
- Rival remains in pursuit;
- opponent does not take possession.

**COMPLETE**:

- Rival obtains the next legitimate touch after rebound;
- that touch materially redirects ball away from backboard.

Goal reward remains separate.

### 8.3 Extended rebound sequence

Further controlled touches/rebounds retain labels and counts, but the rebound-mechanics family payout is capped to prevent indefinite farming.

## 9. Pinch detector family

### 9.1 Compression topology

Orient ball-world normal `n_world` and car-ball normal `n_car` consistently relative to the ball.

**START**:

- ball-world manifold exists or begins inside the calibrated overlap window;
- Rival car-ball contact enters from the opposing side.

**PROGRESS**:

- car-ball and ball-world manifolds overlap;
- their normals are sufficiently opposing to represent compression rather than two unrelated contacts;
- relative contact velocities show closing/compression;
- retain both impulse streams.

A useful derived diagnostic is:

`opposition = -dot(unit(n_car), unit(n_world))`

with larger positive values meaning more directly opposed contacts under the chosen sign convention.

**COMPLETE**:

- compression releases;
- ball receives a material `delta_v` / speed or trajectory change attributable to the combined contact event.

Classify surface from authoritative geometry:

- ground;
- wall/Kuxir-style;
- ceiling;
- post;
- corner/transition.

One compression event -> one pinch payout regardless of surface label.

## 10. Pogo detector family

### 10.1 Contact geometry

Use chassis manifold contact point transformed to car-local coordinates. A pogo begins from a corner/edge support region, not an ordinary stable wheel landing.

Define a calibrated `cornerness`/edge-region test from the actual Octane collision half-extents and retained local contact point rather than world orientation.

### 10.2 Physical sequence

**START**:

- car airborne;
- chassis corner/edge approaches a world surface;
- fewer than ordinary grounded wheel-support conditions exist.

**PROGRESS**:

- chassis-world manifold proves impact;
- contact-point normal velocity is into the surface before impact;
- solver impulse reverses/redirects the normal component;
- car does not immediately settle into normal grounded support.

**COMPLETE**:

- car separates from the surface;
- contact-point/car center normal velocity is now away from the surface;
- rebound retains meaningful translational/control state.

Derived diagnostics:

- rebound normal speed;
- tangent speed before/after;
- angular velocity change;
- impacted local hitbox corner/edge;
- ball touch immediately before/after;
- defender-relative occlusion.

Lock out until a new airborne approach begins.

### 10.3 Pogo tactical variants

Label, do not separately stack reward:

- front-corner pogo;
- reverse/back-corner pogo;
- wall pogo;
- ceiling pogo;
- ball-synchronized/disguised pogo;
- Musty/catch-origin pogo when sequence context supports it.

## 11. Stall-assisted detector family

A bare stall should be observed but not automatically rewarded.

Identify a stall from an actual dodge consumption whose translational dodge vector/impulse is near zero and whose physical orientation transition matches the source state.

A stall becomes reward-relevant only when it is part of a separately completed useful sequence:

- stall -> reset;
- stall -> controlled aerial touch;
- stall -> another source-backed resource/control transition.

## 12. Anti-exploitation rules

### 12.1 Per-family lockout

A completed event cannot pay again until its physical sequence resets.

### 12.2 Possession caps

Repeated contact milestones stop paying after the authorized two/three-touch thresholds inside the same possession epoch.

### 12.3 Compound play accounting

Never pay two labels for one terminal transition. Do allow separate payments for genuinely separate earlier/later physical accomplishments when the future reward contract explicitly authorizes them.

### 12.4 Mechanics reward budget

A future reward contract should bound mechanics-tier contribution per player/episode or apply an equivalent frequency-control mechanism. This protects the game objective from high-frequency dash/touch mechanics.

The budget amount is not frozen here.

### 12.5 Gameplay remains dominant

No mechanic detector should require a goal, save, or opponent error to prove the physical capability. Those outcomes already carry their own much larger game reward and should teach when a capability is worth using.

## 13. Calibration matrix

### Source-exact / discrete

These can be implemented with little subjective thresholding:

- actual dodge onset;
- wheel contact masks;
- ball/other-car/world wheel-body identity;
- source grounded threshold;
- jump/flip/double-jump resource transitions;
- ball/world manifold identity;
- existing measured Rival dash/zap/double-dash sequence evidence.

### Continuous thresholds requiring trace calibration

Build compact positive/near-miss/negative corpora for:

- wavedash meaningful tangent gain if a nonzero tolerance beyond floating noise is required;
- speedflip cancel/reorientation/alignment;
- half-flip reversal and useful momentum;
- possession continuity/control envelope;
- roof carry duration/control;
- redirect angle/outgoing speed;
- pop-reset control/latency;
- rapid-reset latency;
- pinch normal opposition and compression window;
- pogo cornerness/rebound threshold;
- flick/Musty/Breezi/etc. thresholds from the flick appendix.

Threshold selection must be evidence-driven. Raw state around each classified event remains stored so false positives/negatives can be audited.

## 14. Source anchors

Primary physics authority:

- `ZealanL/RocketSim@c2baacb8f4b441dd8505e63c2aeb5a1679b60b02`;
- current accepted RivalSim resident physics implementation.

Project evidence:

- `results/rival2/gameplay_v1_nexto/dash_mechanics_contract.json`;
- `rivalsim/nexto_short_eval.py`.

Current Rocket League product confirmation for reset semantics:

- Rocket League v2.66 flip-reset indicator describes reset as regaining the dodge by touching the ball with three or more wheels.

This appendix is a detector-design authority only; any future training reward requires an explicit immutable reward contract and launch gate.