# Rival 2.0 Mechanics Reward — Calibration and Implementation V0.1

Status: **research/design specification**

Related working documents:

- `docs/RIVAL2_MECHANICS_REWARD_CONTRACT_V0_1.md`
- `docs/RIVAL2_MECHANICS_DETECTOR_PHYSICS_V0_1.md`
- `docs/RIVAL2_MECHANICS_FLICK_VARIANTS_V0_1.md`
- `docs/RIVAL2_MECHANICS_CATALOG_V0_1.md`

This document defines how the mechanics detectors should be calibrated and how a future mechanics reward should be implemented. It does **not** itself authorize a training run or mutate an existing reward identity.

The central rule is:

> Calibration decides whether the physical mechanic happened. It must not grade how well the mechanic was performed.

A weak but genuine Musty, reset, pogo, wavedash, flick, redirect, or other mechanic is still a completed capability. Execution quality and tactical selection are learned from ordinary gameplay return.

---

## 1. Reward scale decision

### 1.1 Common mechanics completion payout

Freeze the prospective common mechanics-tier payout at:

`MECHANICS_EVENT_REWARD = 0.005`

Use the same base completion value for every canonical rewardable mechanic event unless a later explicitly versioned contract changes it.

Current Rival reward scale for comparison:

- goal: `10.0`
- save: `0.75`
- demo: `0.10`
- ordinary legitimate touch: `0.05`
- full boost pad: `0.005`
- small boost pad: `0.001`
- max speed shaping per player decision: `0.00010`
- supersonic shaping per player decision: `0.00020`
- physical boost-use shaping per player decision: `0.00005`

Therefore one mechanics completion is:

- `1/2000` of a goal;
- `1/150` of a save;
- `1/20` of a demo;
- `1/10` of one ordinary touch;
- equal to one full boost-pad pickup.

This is intentionally large enough to create a local credit-assignment signal while remaining far below direct game accomplishments.

Rival PPO normalizes trainable rollout advantages before the policy loss. The mechanics event therefore does not need to approach the raw magnitude of a touch, save, or goal to remain visible to PPO.

### 1.2 Frequency guard

Per-player mechanics reward is capped at:

`MECHANICS_EPISODE_BUDGET = 0.05`

This is ten paid mechanics completions per player per short episode.

The cap is a safety/frequency guard, not a target. Even pathological mechanics farming can contribute no more than the raw value of **one ordinary touch** over the entire episode.

Track:

- `mechanics_reward_paid`;
- `mechanics_reward_suppressed_budget`;
- `mechanics_budget_exhausted`;
- event count by family and subtype.

If budget exhaustion becomes common, treat that as evidence of event spam, overbroad detection, or an unexpectedly high-frequency mechanic. Do **not** automatically increase the budget.

### 1.3 No quality scaling

Do not scale mechanics reward by:

- speed gained;
- shot power;
- height;
- accuracy;
- boost efficiency;
- recovery quality;
- defender reaction;
- goal probability;
- visual cleanliness;
- rarity/novelty.

Those are telemetry and downstream gameplay consequences.

Numerical thresholds below exist only to distinguish a real state transition from floating-point/contact noise or a physically different event.

### 1.4 No mechanics penalties

Do not add negative reward for failed attempts, ordinary jumps, flips, failed Musties, bad recoveries, missed resets, bad flicks, or other mechanics failures.

The normal opportunity cost and game outcome remain the negative feedback.

---

## 2. Reward accounting and de-duplication

### 2.1 Canonical event rule

One physical accomplishment emits at most one mechanics payout within its family even if it has several labels.

Examples:

- `wavedash + wall_dash + landing_wavedash` -> one movement-family payout;
- `flip_reset + preflip_reset + wall_origin` -> one reset-family payout;
- `generic_flick + 45_flick + mawkzy_signature` -> one flick-family payout;
- `musty + breezi` terminal release -> one flick-family payout, with `breezi` as the more specific completion label.

### 2.2 Distinct later physical accomplishments may pay

Do not collapse truly separate state transitions merely because humans describe the whole play with one name.

Examples:

- flip reset acquisition -> later Musty scoop: reset payout + flick payout;
- flip reset acquisition -> later pop created with the acquired dodge: reset payout + later ball-manipulation payout if that pop event is authorized;
- successful dash -> later controlled flick: dash payout + flick payout.

### 2.3 Future full mechanics contract versus current double-dash reward

A future general mechanics reward must **replace**, not stack on top of, the current standalone strict-double-dash `+0.005` component.

Once individual successful dash completions are rewardable:

- first successful dash may pay `0.005`;
- second distinct successful dash may pay `0.005`;
- `double_dash` remains the sequence label;
- the second dash must not receive an additional third `+0.005` merely because the two events together satisfy the double-dash label.

The current Gameplay V2 double-dash-only reward is a separate earlier contract and is not silently changed by this document.

---

## 3. Calibration doctrine

### 3.1 Two classes of detector

#### A. Source-exact/discrete detectors

Freeze directly from authoritative state transitions. No empirical tuning is required except parity tests.

Examples:

- actual jump/dodge onset;
- wheel contact masks;
- contacted-body identity;
- `>=3`-wheel grounded transition;
- flip/double-jump resource state;
- unique car-ball contact onset;
- ball-world manifold identity;
- episode/opponent-touch/world-grounding reset conditions.

#### B. Continuous-motion classifiers

Some named mechanics do not have engine event flags. Their identity depends on continuous geometry/motion.

Examples:

- speedflip cancellation/alignment;
- half-flip reorientation;
- possession continuity envelope;
- Musty rotational scoop contribution;
- Breezi orientation path;
- redirect direction change;
- pinch compression topology margins;
- pogo contact-region/rebound classification.

For these, do not invent a threshold from intuition or a tutorial. Derive it from a compact RivalSim calibration corpus.

### 3.2 Compact calibration corpus

For each continuous detector, construct approximately:

- `24` clear positive traces;
- `24` near-miss negative traces;
- `24` ordinary-action false-positive controls.

Total target: about `72` traces per continuous detector, not a broad simulator benchmark.

Where a mechanic has materially different origins, distribute positives across those origins rather than multiplying the corpus unnecessarily.

Examples:

- Musty: ground/catch, aerial, wall/ceiling, reset-origin;
- pogo: floor plus wall/ceiling variants;
- possession: carry, bounce control, near-ground control;
- redirect: multiple incoming angles/speeds.

### 3.3 Threshold selection rule

For each candidate scalar feature `f`:

1. calculate `f` on every positive and negative trace;
2. identify the nearest positive/negative boundary;
3. if the classes have a clear non-overlapping margin, choose the midpoint of the narrowest separating margin;
4. retain the raw extrema and selected threshold in a calibration artifact;
5. if positive and negative ranges overlap materially, **do not loosen the threshold until it accepts both**;
6. instead add the missing physical feature or change the state-machine topology until the event is separable.

The objective is event identity, not maximum recall.

### 3.4 Held-out shadow gate

After deriving thresholds, run the complete detector in **shadow mode with zero mechanics reward** on a compact held-out evaluation, target `256` short episodes.

Record:

- events/minute by mechanic and family;
- subtype distribution;
- raw evidence for a bounded sample of positive classifications;
- de-dup suppressions;
- family re-arm counts;
- any obviously impossible or repeated-contact-jitter classifications.

Do not train during this gate.

If the detector produces obviously pathological frequency, fix the detector before reward activation.

### 3.5 Numerical noise floors

Unless a source-exact event flag makes a numerical test unnecessary, use only tiny noise rejection floors initially:

- `EPS_LINEAR_SPEED_UU_S = 1.0`
- `EPS_BALL_DELTA_V_UU_S = 1.0`
- contact-normal squared-length guard: `1e-12`

These are **not performance thresholds**. They merely reject effectively zero changes relative to car/ball maxima of `2300` / `6000` uu/s.

If actual deterministic simulator traces show that float32/contact jitter requires a different numerical floor, derive the smallest separating value from the calibration corpus and document it.

---

## 4. Shared implementation architecture

### 4.1 Hot-path location

Mechanics tracking runs at the authoritative `120 Hz` physics boundary:

1. execute authoritative physics tick;
2. inspect resulting wheel/contact/resource/rigid-body state;
3. advance mechanics state machines and emit canonical completion events;
4. accumulate those events into the current `30 Hz` reward interval.

This matches the current architectural placement of the online double-dash tracker: after the physical world tick and before Rival reward accumulation.

### 4.2 GPU-resident state

Add a device-resident mechanics state object, conceptually:

`Rival2MechanicsState`

Per car/world retain only compact state needed by active detectors:

- previous resource state;
- previous wheel mask/contact-body masks;
- previous car pose/velocity/angular velocity;
- previous ball pose/velocity where required;
- active state-machine stage by family;
- timestamps/ticks for short sequence windows;
- possession owner/epoch state;
- current family lockouts;
- per-episode paid mechanics total;
- current 30 Hz canonical event counts;
- subtype/compound telemetry bitmask or bounded counters.

Do not run host-side trace classifiers in training.

Do not use a Python object per world.

Do not copy full world state CPU<->GPU each tick.

### 4.3 Event representation

Each completed mechanics event should expose at minimum:

- `family_id`;
- `canonical_event_id`;
- subtype bitmask/labels for telemetry;
- completion tick;
- rewardable boolean;
- paid/suppressed reason;
- optional compact scalar diagnostics used by its classifier.

Only the canonical event contributes to reward.

### 4.4 30 Hz accumulator

Across the four physics ticks in one Rival policy decision:

- sum new canonical completion events per player;
- de-duplicate same-family terminal labels;
- calculate requested mechanics reward as `event_count * 0.005`;
- pay only up to the player's remaining `0.05` episode budget;
- record the unpaid remainder as budget-suppressed telemetry.

For competitive zero-sum composition:

`BlueMechanics = paid_blue - paid_orange`

`OrangeMechanics = -BlueMechanics`

Reset family state and budget only at true episode/kickoff lifecycle reset.

---

## 5. Dash and movement calibration

### 5.1 Successful wavedash

Use the existing researched Rival classifier as the starting boundary:

- actual `has_flipped` onset from zero wheel contact;
- pre-flip air time `<= 42` physics ticks (`0.35 s`);
- first wheel contact within `24` physics ticks (`0.20 s`) after flip onset;
- compare velocity tangent to the contacted surface;
- require post-landing sampled tangent speed to exceed pre-dodge tangent speed by more than the numerical noise floor.

Completion:

`delta_tangent_speed > EPS_LINEAR_SPEED_UU_S`

Do not scale payout by `delta_tangent_speed`.

Surface subtype uses the existing normalized landing-normal classification unless later calibration proves it needs revision:

- floor/ceiling-like: `abs(n_z) >= 0.85`;
- wall-like: `abs(n_z) <= 0.25`;
- otherwise curved transition.

### 5.2 Zapdash

Use the existing physical sequence:

- front-wheel-first landing;
- non-flat three-wheel grounded first-jump onset within `12` ticks (`0.10 s`);
- directional landing dodge within `30` ticks (`0.25 s`) after that first jump;
- terminal dash satisfies the successful-dash event.

The sequence receives one movement-family payout for the terminal successful dash, with `zapdash` as a subtype label.

### 5.3 Rival double dash

Use the project definition:

- two successful dash completions;
- intervening wheel support/contact;
- second sequence begins/completes within the established `90`-tick (`0.75 s`) window;
- final sequence tangent speed exceeds pre-first-dash tangent speed by more than the numerical noise floor.

In the future general mechanics contract, the two distinct successful dash events may each receive their normal dash payout. The `double_dash` label itself does not create a third payout.

### 5.4 Speedflip

Do not freeze arbitrary cancel angles now.

Calibration corpus:

**Positive traces**

- controlled forward-diagonal dodge;
- early pitch cancellation;
- retained dodge translation;
- rapid forward-axis/velocity realignment.

**Negative traces**

- uncancelled diagonal flip;
- ordinary front flip;
- late cancel that behaves like ordinary flip;
- boost-only acceleration without qualifying dodge sequence.

Candidate features:

- time from actual dodge onset to pitch angular-velocity arrest/reversal;
- integrated pitch rotation before arrest;
- residual roll/yaw rotation;
- time until `dot(unit(forward), unit(surface_tangent_velocity))` recovers;
- retained tangent/forward translation versus pre-dodge state.

Select the narrowest trace-derived boundary that separates real speedflips from ordinary diagonal flips. Reward is binary `0.005`; do not reward a faster speedflip more.

### 5.5 Half-flip

Calibration positives:

- actual backward dodge;
- early pitch cancel;
- roll/yaw reorientation;
- approximately opposite final heading with retained motion.

Negatives:

- ordinary backflip;
- stationary air-roll 180;
- ordinary powerslide/ground turn;
- uncontrolled tumble that happens to face backward.

Features:

- actual backward dodge onset;
- pitch arrest/reversal timing;
- `dot(forward_final, forward_initial)`;
- velocity projection on new forward axis;
- wheel support/control state at sequence completion.

Thresholds are selected from the positive/negative trace margin.

---

## 6. Reset/resource calibration

### 6.1 Ball flip reset — source-exact

No quality threshold is required.

Start:

- car airborne;
- record pre-contact dodge-resource state.

Complete only when:

- `>=3` wheels simultaneously support on the ball through authoritative wheel contact/body identity;
- normal grounded logic restores the flip/double-jump state;
- car separates from the ball without a new grounded jump;
- car remains airborne;
- post-event resource state is `AIR_UNTIMED_AVAILABLE`;
- pre-event state was not already the same unchanged untimed resource.

Pay `0.005` on the acquisition event.

A weak/awkward reset still pays if the resource was genuinely restored.

### 6.2 Chain reset

Re-arm only after the previously acquired resource is actually consumed/lost or the first reset acquisition lockout otherwise ends.

A later distinct `AIR_UNTIMED_AVAILABLE` reacquisition before world grounding emits another reset-family event.

### 6.3 Pre-flip reset

Require:

- actual dodge/resource consumption before contact;
- later valid reset completion.

Subtype only; one reset-family payout for the reacquisition.

### 6.4 Pop reset

Treat as two possible physical events:

1. reset acquisition;
2. later controlled pop produced with the newly acquired dodge.

Avoid an arbitrary fixed `must use within X ms` requirement initially. Keep the acquired-reset token until one of:

- world grounding;
- opponent touch;
- acquired resource consumed for a different event;
- possession/aerial continuity clearly ends.

Pop completion requires:

- acquired dodge is consumed;
- a distinct same-player ball contact/impulse occurs;
- ball receives positive upward velocity change greater than the numerical floor;
- Rival remains in the aerial/control sequence.

Do not scale by pop power.

### 6.5 Car reset

Same resource transition as ball reset except the `>=3` wheel support body is the other car.

---

## 7. Possession and repeated-contact calibration

### 7.1 Unique touch authority

Use authoritative contact onset/latching. Persistent manifold chatter is never a new touch.

### 7.2 Possession continuity must be trace-calibrated

Do not define possession from a large guessed center-to-center radius alone.

Use a compact control-state classifier based on:

- latest legitimate touch owner;
- car-local ball position;
- car-ball relative position and relative velocity;
- whether ball trajectory remains reachable without a discontinuous chase;
- ball-world bounce relation where relevant;
- time since last legitimate owner touch;
- opponent touch as an exact termination.

Calibration positives:

- stable roof carry;
- controlled bounce dribble;
- soft catch into a second touch;
- low-speed controlled dribble with brief contact gaps.

Negatives:

- shot/clear followed by much later accidental recontact;
- kickoff/50 contact sequence;
- ball blasted outside control then chased down;
- manifold chatter;
- opponent interception.

Choose control envelope/gap thresholds from the separating trace margin.

### 7.3 Two-touch possession

Within one uninterrupted possession epoch:

- second distinct same-player legitimate touch -> `+0.005`.

### 7.4 Three-touch possession

Same epoch reaches third distinct same-player touch -> another `+0.005`.

Then lock further repeated-contact mechanics payout for that possession epoch.

Touch 4+ remains useful through ordinary touch/progress/game reward and telemetry.

### 7.5 Ground carry

Use car-local contact/support geometry, not merely ball-center distance.

Candidate positive evidence:

- grounded/supporting Rival;
- ball located in upper/roof control region;
- persistent or repeatedly renewed car-ball support;
- low relative slip/separation;
- no ball-world compression topology indicating a pinch.

Duration/control tolerance is trace-calibrated from real carry traces versus transient roof contacts.

One carry epoch pays once.

### 7.6 Bounce dribble

Require:

- active Rival possession epoch;
- authoritative ball-ground bounce;
- Rival remains inside the calibrated continuation envelope;
- next legitimate Rival touch occurs before opponent ownership.

Do not reward the bounce itself.

---

## 8. Aerial control calibration

### 8.1 Ground-to-air continuation

Start from controlled ground possession/touch.

Require:

- ball gets real upward `delta_v` above numerical noise;
- Rival becomes airborne;
- same player gets the next distinct aerial touch before opponent touch or Rival world grounding.

Pay on the aerial continuation touch, not the pop itself.

### 8.2 Two-touch air dribble

Within one aerial possession epoch:

- first distinct airborne Rival touch starts/continues epoch;
- second distinct airborne Rival touch before opponent touch/world grounding -> `+0.005`.

### 8.3 Three-touch air dribble

Third distinct same-player airborne touch in the same epoch -> another `+0.005`.

Then lock further aerial repeated-touch mechanics payout for that epoch.

### 8.4 Near-ground ambiguity

Do not use a large arbitrary height requirement. A legitimate low air-dribble touch can occur near the floor.

Use actual wheel/world grounding state as the primary boundary. If calibration traces expose one-tick wheel-loss/contact jitter, derive the smallest persistence window necessary to reject it.

Boost is never required for identity.

---

## 9. Flick calibration

### 9.1 Generic controlled flick

Prerequisite: calibrated possession/catch epoch.

Require:

- actual dodge onset;
- ball contact while the dodge/rotating hitbox is the active release mechanism;
- source contact impulse or `abs(delta_v_ball) > EPS_BALL_DELTA_V_UU_S` proving a real release;
- ball leaves the pre-release possession/support relation.

Pay one binary flick-family event.

### 9.2 Directional/angle variants

Classify from actual physical geometry:

- car forward/right/up at dodge onset;
- car-local ball position;
- actual dodge direction/flip torque;
- contact-point location;
- car orientation at contact;
- outgoing ball vector.

Do not create separate reward values for 45/90/180/etc. Their subtype identity remains available for tactical learning/telemetry.

### 9.3 Musty

A Musty must prove a genuine rotational scoop, not merely a backward dodge near the ball.

At the decisive contact compute:

`v_contact = v_car + cross(omega_car, r_contact)`

`v_rot = cross(omega_car, r_contact)`

Using a consistently oriented car->ball contact normal `n`, compute:

- total closing contribution along `n`;
- translational contribution along `n`;
- rotational contribution along `n`;
- contact-point sweep path over the short pre-contact window;
- actual backward-dodge state.

Calibration positives:

- successful Musty from ground/catch;
- aerial Musty;
- wall/ceiling-origin Musty;
- reset-origin Musty.

Near-miss negatives:

- ordinary backflip into loose ball;
- backward dodge where translation, not rotation, creates the hit;
- ordinary rear/roof collision without the scoop path;
- ball contact before the Musty sweep develops.

Do **not** invent a fixed rotational percentage now.

Derive the minimum rotational-scoop boundary from the positive/negative trace margin. If one scalar fraction overlaps, add sweep-path/contact geometry rather than grading power.

Any genuine Musty pays `0.005`, regardless of power or tactical success.

### 9.4 Breezi

A Breezi requires:

- controlled ball relation through setup;
- sustained combined roll/yaw orientation path consistent with the tornado-like setup;
- nose-up/inverted transition;
- later nose-down/roof-to-ball Musty release geometry;
- valid terminal Musty-class scoop.

Calibration positives:

- several timing/speed variations of true Breezi setup.

Negatives:

- ordinary Musty;
- random air-roll before Musty;
- Classy-style different inverted setup;
- generic aerial correction followed by backflip contact.

Features:

- integrated roll/yaw orientation path;
- orientation sequence/order rather than only final quaternion;
- setup duration;
- continuity of ball control;
- terminal Musty confirmation.

Select path/timing tolerances from the trace margin. Do not reward delay amount.

Most-specific terminal label is `breezi`; do not stack generic flick + Musty + Breezi payouts.

### 9.5 Other flick variants

Mawkzy/JZR/Classy/Suffo/Wizard/Bismillah/etc. remain subtype classifiers around the common flick/Musty primitives.

Each subtype should be trace-calibrated only if/when Rival actually produces enough candidate traces or the project intentionally creates a small positive corpus.

Absence of a subtype classifier must not block the base valid flick reward.

---

## 10. Redirect and rebound calibration

### 10.1 Redirect

At contact retain:

- incoming ball speed;
- incoming unit direction;
- outgoing ball speed;
- outgoing unit direction;
- angle between incoming and outgoing velocity.

Calibration corpus separates:

- real redirects;
- dead catches;
- tiny incidental deflections;
- ordinary dribble touches.

Derive minimum incoming speed, outgoing speed, and direction-change boundary from traces.

Do not reward more for a larger angle or harder shot.

### 10.2 Double tap / double touch

Use authoritative sequence topology rather than a quality threshold:

1. Rival touch;
2. opponent-backboard world contact;
3. same Rival next touch before opponent possession;
4. post-second-touch ball velocity has a component away from the contacted backboard normal greater than numerical noise.

A weak second touch still counts if the rebound sequence genuinely completes.

Goal reward remains separate.

---

## 11. Pinch calibration

### 11.1 Compression topology

Require overlapping/narrowly adjacent:

- car-ball manifold;
- ball-world manifold.

Orient both normals consistently relative to the ball and compute:

`opposition = -dot(unit(n_car_ball), unit(n_ball_world))`

Also require contact-point relative velocities to show the ball is being closed/compressed rather than merely touching two unrelated surfaces.

On release, require ball `delta_v` above numerical noise.

Do not initially demand a huge ball-speed increase; a weak real pinch still counts.

### 11.2 Threshold derivation

Positive corpus:

- ground pinch;
- wall/Kuxir pinch;
- ceiling/corner variants.

Negatives:

- ball rolling against wall while car taps same side;
- ordinary wall touch;
- ball/world contact shortly before unrelated car hit;
- non-opposed simultaneous contacts.

Select the narrow contact-overlap window and normal-opposition boundary from the trace margin.

Surface subtype is derived from authoritative world geometry.

---

## 12. Pogo calibration

### 12.1 Physical event

Require:

- car airborne approaching world surface;
- chassis-world contact in lower corner/edge region rather than ordinary wheel-supported landing;
- pre-contact contact-point normal velocity directed into surface;
- post-contact normal velocity directed away from surface by more than numerical noise;
- car does not immediately settle into source grounded `>=3` wheel support.

### 12.2 Contact-region calibration

Express chassis contact point in car-local hitbox coordinates.

Positive corpus:

- front/rear corner pogos;
- wall/ceiling variants if desired.

Negatives:

- ordinary four-wheel landing;
- one/two-wheel bounce;
- roof slam;
- uncontrolled chassis collision that immediately grounds.

Derive the lower-corner/edge region boundary and any short post-contact persistence window from traces.

Do not reward rebound height or quality.

---

## 13. Stall handling

Bare stall remains telemetry-only.

Detect source-backed dodge consumption with near-zero translational dodge effect / stall-like flip torque state.

A later separately completed reset, aerial-control, flick, or other mechanic may pay normally. Do not give an extra standalone `+0.005` simply because a stall occurred.

---

## 14. Calibration artifacts required before reward activation

For every continuous detector publish a compact machine-readable record containing:

- detector/version;
- positive trace IDs;
- negative trace IDs;
- features measured;
- positive extrema;
- negative extrema;
- selected threshold(s);
- separation margin;
- false positives/false negatives on held-out calibration cases;
- exact RivalSim source commit;
- exact physics/observation/action contract hashes where relevant.

A detector may enter the reward hot path only when:

- all intended positive calibration cases classify;
- all intended near-miss/ordinary negatives reject;
- held-out shadow evaluation shows no obvious exploit-frequency pathology;
- state-machine lockout/re-arm tests pass.

If clean separation is not possible, leave the mechanic telemetry-only until the detector is improved.

---

## 15. Implementation tests

Before a future mechanics reward is used for training, require focused tests for:

1. source-exact positive/negative reset transitions;
2. dash timing/contact boundary parity with the existing researched classifier;
3. continuous-detector calibration corpus classifications;
4. one payout per canonical same-family physical event;
5. subtype labels without duplicate payout;
6. distinct compound events allowed to pay separately;
7. family lockout and genuine re-arm;
8. exact `0.05` per-player episode budget cap;
9. budget reset only on episode reset;
10. exact zero-sum mechanics composition;
11. no event leakage across kickoff/reset;
12. exact reward-component reconstruction;
13. no per-world CPU classifier or full-state host transfer in the training hot path.

Do not broaden these into a full simulator acceptance suite.

---

## 16. Runtime telemetry and reward-scale acceptance

During any future mechanics-enabled training run, report per family:

- events/minute;
- paid events/minute;
- suppressed same-family duplicates;
- suppressed budget events;
- mechanics reward mean absolute per decision;
- mechanics reward as fraction of ordinary touch component;
- mechanics reward as fraction of V1 progress component;
- episode-budget hit fraction;
- subtype counts.

The hard `0.05` episode cap is the primary domination guard.

Additionally, if the observed mechanics component becomes unexpectedly comparable to ordinary touch/progress reward despite the cap, stop and inspect event frequency/accounting before extending training.

---

## 17. What is frozen versus what must still be measured

### Frozen now

- `MECHANICS_EVENT_REWARD = 0.005`;
- `MECHANICS_EPISODE_BUDGET = 0.05` per player;
- no quality scaling;
- no mechanics penalties;
- no novelty multiplier;
- 120 Hz detector / 30 Hz reward accumulation architecture;
- family de-duplication and compound-event rules;
- source-exact reset/resource semantics;
- existing dash windows: `42 / 24 / 12 / 30 / 90` ticks;
- existing surface-normal classes: `0.85 / 0.25`;
- tiny initial numerical noise floors: `1.0 uu/s` for linear/ball delta-velocity tests;
- two-touch/three-touch milestone structure and later lockout.

### Must be measured by the compact calibration gate

- speedflip cancel/alignment boundary;
- half-flip cancellation/reorientation boundary;
- possession/control envelope and allowed contact gap;
- carry persistence/control boundary;
- Musty rotational-scoop separator;
- Breezi orientation-path separator;
- redirect velocity/angle separator;
- pinch overlap/opposition separator;
- pogo local-contact-region/rebound separator.

Those measurements are implementation work, not unresolved conceptual design. The procedure above is the authority for obtaining them without turning execution quality into reward quality.
