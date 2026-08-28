# Gameplay V3 Implementation Specification

## 1. Reward identity

Add a new immutable reward version:

`RIVAL2_REWARD_GAMEPLAY_V3`

Add a canonical contract/hash entry without changing any prior contract object or hash.

Gameplay V3 uses:

- policy cadence: `30 Hz`;
- physics cadence: `120 Hz`;
- action hold: `4` physics ticks;
- episode lifecycle: unchanged `RIVAL2_EPISODE_V1`;
- zero-sum competitive reward composition.

## 2. Exact V3 reward arithmetic

At the 30 Hz decision boundary, compute Blue first and set Orange to its exact negation.

Conceptually:

```text
BlueCore =
    Goal
  + Progress
  + Demo

BlueGameplayExtras =
    Speed
  + Supersonic
  + BoostUse
  + BoostPickup
  + Save

BlueMechanics = BluePaidMechanics - OrangePaidMechanics

BlueBadFlip =
    -0.01 * BlueUnnecessaryFlipThroughContacts
    +0.01 * OrangeUnnecessaryFlipThroughContacts

BlueReward = BlueCore + BlueGameplayExtras + BlueMechanics + BlueBadFlip
OrangeReward = -BlueReward
```

Preserve Gameplay V1 arithmetic for the retained terms:

- goal: `+10/-10`;
- progress: `0.5 * (ball_y_after - ball_y_before) / 5120`;
- demo: `0.10 * (BlueDemoOnsets - OrangeDemoOnsets)`;
- speed: current Gameplay V1 formula;
- supersonic: current Gameplay V1 event/state value;
- physical boost use: current Gameplay V1 semantics;
- pad pickup: current Gameplay V1 semantics;
- save: current Gameplay V1 semantics.

### Touch correction

For Gameplay V3 only:

`unconditional_unique_touch = 0.0`

Do not remove touch onset tracking, lifecycle observations, save dependency, telemetry, or no-touch lifecycle accounting.

Do not mutate the historical V1/V2 unique-touch branch. V3 gets its own arithmetic path.

### Removed V2 standalone term

Gameplay V3 must not add `GAMEPLAY_STRICT_DOUBLE_DASH_REWARD` from the V2 reward branch.

The general mechanics system replaces that standalone V2 term. The historical V2 path remains untouched.

## 3. Native production architecture

### 3.1 Do not attach `MechanicsShadowObserver` in training

The calibration observer is evidence infrastructure. Its `attach()` monkeypatch and evidence arrays are not the production architecture.

Refactor/reuse the validated Warp functions and thresholds, but create native V3 state owned by `Rival2WorldSim`, conceptually:

`Rival2GameplayV3State`

Allocate it **only when Gameplay V3 is constructed**. Do not add the full V3 detector state to V1/V2 environments.

The state must exist before `Rival2TensorBridge` binds arrays.

### 3.2 Required per-tick order

For `REWARD_MODE_GAMEPLAY_V3`, the authoritative order is:

```text
CompleteWorldSim / super()._launch_tick()
    -> authoritative physics/contact/resource state complete
    -> V3 mechanics + flip-contact detector launch(es)
    -> rival2_accumulate_tick(...)
    -> on physics tick 4, V3 reward composition
```

Do not change the order for historical reward modes.

Do not run V3 detection before the physical tick.

### 3.3 Persistent vs interval state

Separate state by lifetime.

**Persistent episode/detector state** survives `begin_decision()`:

- previous car/ball pose/velocity/resource state needed by detectors;
- Musty nine-tick sweep history;
- Breezi setup state;
- dash/zap/double-dash sequence state;
- reset acquisition/re-arm state;
- Pogo pending state;
- flip-contact pending classification state;
- per-detector re-arm/lockout state;
- per-player mechanics paid-event count for current episode;
- cumulative telemetry counters if desired.

**30 Hz interval state** resets in `begin_decision()`:

- canonical mechanics completions emitted this decision;
- paid mechanics event count this decision;
- budget-suppressed mechanics this decision;
- bad-flip penalties resolved this decision;
- exemption counts resolved this decision;
- V3 reward component reconstruction arrays.

A pending contest/mechanic classification may cross a 30 Hz boundary. Its persistent state must not be cleared by `begin_decision()`. If it resolves during the next interval, its reward/penalty belongs to that next interval exactly once.

### 3.4 Episode reset

On `apply_interval_resets()` for a true reset world, clear all V3 persistent detector state and the mechanics episode budget in a dedicated V3 reset kernel.

Do not rely only on detecting `episode_ticks` wrapping inside a later tick. Explicit reset is safer and required for production.

The V3 reset launch must happen after the accepted physical kickoff reset has completed and before the next policy decision.

No detector history may bridge two episodes.

## 4. Production memory design

The current read-only calibration observer is intentionally diagnostic and is too large to copy verbatim into training.

At the published training scale (`131,072` worlds / `262,144` cars), the audited shadow-observer layout is approximately:

- ~`180 MiB` before bounded evidence arrays;
- ~`112 MiB` additional evidence arrays at capacity 16/car;
- ~`292 MiB` total logical diagnostic state.

Production requirements:

1. Do not allocate per-car evidence ring buffers in the training state.
2. Do not allocate `reward_contribution` from the shadow observer; V3 reward uses native interval counters.
3. Do not retain calibration-only possession/carry classifier state unless required by the new controlled-flick exemption. If required, keep only the minimal exemption state.
4. Reuse state between compatible detectors where doing so cannot alter semantics.
5. Expose a deterministic `logical_bytes`/state-memory report for the new V3 state.
6. Report added logical bytes at `131,072` worlds.

Diagnostic event evidence should be produced only by a bounded evaluation observer/export path, not by every training world.

## 5. Mechanics payout accounting

### 5.1 Common payout and budget

`MECHANICS_EVENT_REWARD = 0.005`

`MECHANICS_EPISODE_BUDGET = 0.05`

Enforce the budget as an integer event capacity:

`MECHANICS_MAX_PAID_EVENTS_PER_EPISODE = 10`

For each player:

```text
requested = canonical_rewardable_events_this_interval
remaining = max(0, 10 - mechanics_paid_events_episode)
paid = min(requested, remaining)
suppressed_budget = requested - paid
mechanics_paid_events_episode += paid
interval_mechanics_reward = 0.005 * paid
```

This representation is the authority for cap enforcement and avoids floating-point accumulation drift. The contract may still state the monetary/reward budget as `0.05`.

### 5.2 Canonical accomplishment / de-duplication

Do not blindly reuse `mechanics_calibration.FAMILY_NAMES` as production reward families. Those are calibration detector IDs, not a complete reward-accounting ontology.

A physical accomplishment emits one canonical payout. Subtypes do not stack.

Required examples:

- successful wavedash + wall/curve/ceiling subtype -> one payout;
- zapdash terminal successful dash -> one payout, subtype `zapdash`;
- first successful dash + later distinct second successful dash -> two payouts;
- the pair may be labeled `double_dash`, but that label adds no third payout;
- reset + preflip subtype -> one reset payout;
- Musty + Breezi classification for one terminal scoop -> one flick-family payout with most-specific `breezi` label;
- distinct reset acquisition followed later by a Musty -> two payouts because they are different physical accomplishments.

Implement event identity so de-duplication applies to the same terminal accomplishment, not a broad family lockout that suppresses later real events.

## 6. Reward-ready mechanics

### 6.1 Continuous calibrated detectors

Production behavior must match the committed final calibration thresholds/state topology for:

- speedflip;
- half-flip;
- Musty;
- Breezi;
- redirect;
- pinch;
- pogo.

Use `results/rival2/mechanics_calibration_v1/thresholds.json` and the targeted-correction evidence from source commit `0124cd7f29278702158c9cbba9c741c11a29f111` / published evidence commit `0228a833e90d2db2715f8b79b65f6cbdc59fefbc`.

Do not re-fit those thresholds during this implementation unless a concrete runtime parity test proves the production port differs from the validated observer. If that happens, stop and document the mismatch before changing calibration.

### 6.2 Wavedash / dash

The old Gameplay V2 online tracker is **not** sufficient for V3 payout. It recognizes the timing/contact pair used for the old double-dash reward but does not enforce the final calibrated useful tangent-speed result.

Keep `rival2_track_strict_double_dash` unchanged for historical Gameplay V2.

Create a separate V3 dash detector using the mechanics authority:

- actual flip onset from zero wheels / airborne;
- pre-flip air time <= 42 physics ticks;
- first wheel contact <= 24 ticks after flip onset;
- derive contacted surface normal;
- compare surface-tangent speed against pre-dodge tangent speed;
- require `delta_tangent_speed > 1.0 uu/s` numerical floor;
- one completion event per successful dash.

Surface subtype:

- floor/ceiling-like: `abs(n_z) >= 0.85`;
- wall-like: `abs(n_z) <= 0.25`;
- otherwise curve/transition.

Do not quality-scale payout by speed gain.

### 6.3 Zapdash

Subtype/sequence requirements:

- front-wheel-first landing;
- non-flat three-wheel grounded first-jump onset within 12 ticks;
- directional landing dodge within 30 ticks;
- terminal event must satisfy the successful V3 dash detector.

The terminal successful dash gets the one payout. `zapdash` is a subtype label, not an additional payout.

### 6.4 Rival double dash

Project semantics are frozen:

- two successful V3 dash completions;
- intervening wheel support/contact;
- second sequence within 90 ticks / 0.75 s;
- final sequence tangent speed exceeds pre-first-dash tangent speed by > 1.0 uu/s.

Do **not** add a no-fresh-jump restriction.

Both distinct dash completions may pay individually. `double_dash` itself adds zero additional reward.

### 6.5 Reset/resource

Implement source-exact reset acquisition based on real resource transition, not input buttons.

Ball reset completion requires:

- airborne car;
- >=3 wheels simultaneously support on ball;
- authoritative resource transition to `AIR_UNTIMED_AVAILABLE`;
- pre-event resource was not already the same unchanged untimed resource;
- separation without a new grounded jump;
- car remains airborne.

Car reset uses the same resource transition with the other car as support body.

Chain reset pays only on a distinct later resource reacquisition after the prior acquired resource is actually consumed/lost or the reset lockout genuinely re-arms.

Pre-flip reset is a subtype of the reacquisition and does not stack another reset payout.

Do not add pop-reset, generic ceiling departure, or stall payout in this task.

## 7. Unnecessary flip-through contact

### 7.1 Candidate definition

Create authoritative event:

`UNNECESSARY_FLIP_THROUGH_CONTACT`

Penalty to the offending player before zero-sum competitive composition:

`-0.01`

A candidate is created only on a **new legitimate car-ball contact onset** when the car is in an **active directional dodge at the contact**.

Use authoritative physical state. Prefer the conjunction:

- `is_flipping != 0` at the contact tick;
- valid flip/dodge resource state indicating an actual dodge is underway;
- non-zero directional `flip_rel_torque` consistent with a directional dodge;
- legitimate contact onset.

Do not use `has_flipped` alone; it can remain set after the active dodge motion.

Do not use jump/button input as authority.

### 7.2 Independent V3 contact latch

Do not repurpose or pre-update the historical `touch_contact_latched` array, because `rival2_accumulate_tick()` owns it for V1/V2 lifecycle compatibility.

V3 may maintain its own contact-onset latch/state for detector classification, and tests must prove its onset count agrees with historical legitimate touch-onset telemetry under V3.

### 7.3 Exactly-one outcome

Every V3 flip-touch candidate must eventually resolve exactly once to one primary outcome:

- `EXEMPT_CONTESTED_50`;
- `EXEMPT_POWER_CONTACT`;
- `EXEMPT_CONTROLLED_FLICK`;
- `EXEMPT_RECOGNIZED_MECHANIC`;
- `UNNECESSARY_FLIP_THROUGH_CONTACT`.

If several exemptions are true, retain all applicable telemetry flags but select one deterministic primary reason and emit no penalty.

Recommended precedence for primary telemetry reason:

1. recognized same-contact mechanic;
2. controlled flick;
3. contested 50/challenge;
4. dodge-powered shot/clear;
5. unnecessary flip-through.

This precedence changes telemetry labeling only; any exemption suppresses the penalty.

## 8. Pending classification window

Some contest/mechanic evidence may appear a few 120 Hz ticks after initial contact.

Implement a persistent bounded pending record per car containing only the captured state necessary to classify that one contact.

The maximum wait window must be derived from the focused contest/mechanic calibration in this task. Do not invent a large window.

Requirements:

- pending classification survives 30 Hz `begin_decision()` resets;
- pending classification is cancelled cleanly on true episode reset;
- it cannot resolve twice;
- it cannot attach a later unrelated mechanic/opponent touch to the old contact;
- if it resolves in the next 30 Hz interval, its penalty/exemption telemetry is booked in that interval;
- no unresolved candidate may leak across kickoff/reset.

## 9. Contest / 50 exemption

### 9.1 Source-exact evidence

Automatically exempt when the opponent has a legitimate ball contact in the same physical collision interval or within the calibrated narrowly adjacent contest window and the contacts are part of the same contest event.

### 9.2 Convergence evidence

A challenge can also be exempted before/without opponent contact only if both cars are physically committing to the ball.

Candidate features may include:

- opponent-ball separation;
- self/opponent closing speed toward ball;
- short-time intercept estimate;
- relative time-to-ball;
- self/opponent contact geometry.

A proximity radius alone is not valid.

### 9.3 Calibration

Run a small deterministic 24/24/24-style corpus (or smaller only if every boundary class remains well represented) with prospective held-out cases.

Required positives:

- simultaneous 50;
- opponent contact just before;
- opponent contact just after;
- genuine converging challenge.

Required negatives:

- distant opponent;
- opponent moving away;
- opponent behind play;
- opponent merely near ball but not contesting;
- uncontested loose-ball flip-through.

Freeze thresholds/window before held-out evaluation. If clean separation cannot be achieved, return `BLOCKED`.

## 10. Dodge-powered shot/clear exemption

A flip touch is exempt only when active dodge/rotating hitbox motion materially participates in the physical transfer.

At the authoritative contact use:

`v_contact = v_car + cross(omega, r_contact)`

Capture at minimum:

- total contact-point closing velocity;
- translational closing contribution;
- rotational/dodge contribution;
- actual ball delta-v / impulse;
- pre/post ball velocity;
- directional dodge state.

Do not exempt solely because ball speed rises or the car is fast.

Calibration positives must include offensive power contact, defensive clear, more than one dodge/contact geometry, and a weaker-but-genuine dodge-powered example.

Hard negatives must include weak flip touch, translation-dominated high-speed hit while flipping, normal drive-through contact, and an already-fast ball receiving a negligible dodge contribution.

Thresholds define physical identity only; no quality scaling.

If clean separation cannot be achieved, return `BLOCKED`.

## 11. Controlled-flick exemption only

Possession and ground-carry reward detectors remain `NOT_READY_FOR_REWARD`.

Do not promote them.

Build a narrow, high-confidence control/release classifier whose only purpose is to protect a real controlled flick from the bad-contact penalty.

Require a short pre-dodge continuous control history using features such as:

- car-local ball relation;
- car-ball relative velocity;
- recent contact/owner relation where available;
- support/catch geometry;
- short continuity duration;
- dodge-generated release contact;
- ball exits the pre-release control relation.

Calibrate against real flick positives and loose-ball flip-through hard negatives.

This detector emits `EXEMPT_CONTROLLED_FLICK`; it emits **zero mechanics reward**.

Musty/Breezi rewards remain governed by their own calibrated mechanic detectors.

If legitimate controlled flicks cannot be protected cleanly, return `BLOCKED` before V3 is declared ready.

## 12. Recognized-mechanic exemption

Do not let one physical contact receive both a mechanic reward and an unnecessary-flip penalty when the mechanic itself legitimately uses that same dodge/contact.

Use explicit same-contact association, not temporal proximity.

Automatic recognized-mechanic exemption allowlist for this V3 task:

- Musty terminal scoop;
- Breezi terminal scoop;
- pre-flip/reset continuation when the classified contact is the actual documented dodge-contact sequence;
- any future detector added in this implementation only if its canonical identity explicitly requires the same dodge/ball contact and is listed in the final V3 contract.

Do **not** blanket-exempt speedflip, half-flip, dash, reset acquisition, redirect, pinch, or pogo merely because one occurred nearby. Those mechanics do not universally prove that this particular flip-ball contact was necessary. A dodge-powered redirect/clear may instead qualify through the power-contact exemption.

## 13. Telemetry

Production telemetry must be device-resident counters, not per-event Python objects.

Expose bridge views for at least:

### Ball contact

- `legitimate_touch_onsets`;
- `flip_active_touch_onsets`;
- `unnecessary_flip_through_contacts`;
- `contested_50_exemptions`;
- `power_contact_exemptions`;
- `controlled_flick_exemptions`;
- `recognized_mechanic_exemptions`.

Derived evaluation ratios:

- flip-touch fraction of all touches;
- unnecessary fraction of flip-touches.

### Mechanics

For every production mechanic/event family:

- detected events;
- paid events;
- subtype counts/bitmask where compact;
- duplicate suppressions;
- re-arm count;
- budget-suppressed events.

Per player:

- episode paid mechanic event count (0..10);
- episode budget exhausted flag/count.

### Reward reconstruction

Per environment expose competitive Blue components:

- goal;
- progress;
- unconditional_touch (must be `0.0`);
- demo;
- speed;
- supersonic;
- boost_use;
- boost_pickup;
- save;
- mechanics;
- unnecessary_flip_through;
- total.

Do not change the observation schema to include these telemetry values.

## 14. Reward mode integration

Add a new `REWARD_MODE_GAMEPLAY_V3` without renumbering historical modes.

V3 detector launches must occur only in V3 mode.

V1/V2 runtime state and launch paths must remain intact. In particular:

- V2 still launches `rival2_track_strict_double_dash` exactly as before;
- V3 uses the new general mechanics detector instead;
- V3 reward branch must not fall through to the old generic Gameplay branch if that would include the historical touch reward or V2 double-dash term.

## 15. Checkpoint/reward transition semantics

Do not use a live in-place `transition_reward_curriculum()` from an existing Gameplay V2 process for the production campaign transition.

Gameplay V3 starts in a **freshly constructed V3 environment** from an accepted checkpoint boundary.

Use/extend the explicit checkpoint curriculum-transition path so it:

- validates the source checkpoint's Gameplay V2 contract exactly;
- restores model, optimizer, RNG, counters, historical pool, mixed PPO state, retention corpus, curriculum generator, family assignments, and side assignments;
- creates fresh simulator/V3 detector episode state;
- resets frozen-opponent temporal adapters to the fresh episode/kickoff state consistent with their restored family/side assignments rather than carrying temporal history from a different world state;
- records exactly what was preserved vs reinitialized.

Do not alter opponent probabilities or adapter implementation.

A test must prove the reward transition cannot accidentally use strict `load_checkpoint()` with mismatched V2/V3 contract hashes.

## 16. Training is not part of this task

No call to `trainer.update()`, `train_iteration()`, `ppo_update()`, or mixed PPO update may run under Gameplay V3 during this implementation task.

The only allowed policy execution is frozen inference for tests, smoke checks, and no-learning shadow evaluation.
