# Rival 2.0 Mechanics Calibration V1 — Execution Handoff

Status: **authorized calibration work; no training and no reward activation**

## 1. Mission

Convert the mechanics research contracts into measured, implementation-grade detector boundaries for the continuous mechanics that still require calibration.

This handoff is **calibration only**. It must not:

- train or fine-tune Rival;
- change the policy, observation, action, PPO, physics, episode, or opponent contracts;
- activate the general mechanics reward in training;
- change the frozen `MECHANICS_EVENT_REWARD = 0.005` or `MECHANICS_EPISODE_BUDGET = 0.05` design values;
- retune source-exact reset semantics or the already-researched dash timing windows unless a direct parity defect is discovered;
- broaden into a simulator acceptance benchmark.

The output is a calibrated detector/evidence package that can be reviewed before a later, separately authorized reward implementation.

## 2. Repository source

Handoff prepared from `main` at:

`1da8557f32a94e6a8e96d1acbb0103656e203e27`

Implementation may begin from a later `main` descendant, but **must not reset or discard newer repository work**. Verify the handoff source is an ancestor of the implementation head before changing code.

## 3. Design authority

Read these files completely before implementation:

1. `docs/RIVAL2_MECHANICS_REWARD_CONTRACT_V0_1.md`
2. `docs/RIVAL2_MECHANICS_DETECTOR_PHYSICS_V0_1.md`
3. `docs/RIVAL2_MECHANICS_FLICK_VARIANTS_V0_1.md`
4. `docs/RIVAL2_MECHANICS_CATALOG_V0_1.md`
5. `docs/RIVAL2_MECHANICS_CALIBRATION_IMPLEMENTATION_V0_1.md`

The calibration doctrine in those documents is authoritative:

> thresholds decide whether the physical mechanic occurred; they do not grade execution quality.

A weak but genuine mechanic must classify. A physically different near-miss must reject.

## 4. What is already frozen — do not recalibrate

Preserve unless a direct implementation/parity defect is proven:

- physics cadence: 120 Hz;
- Rival policy cadence: 30 Hz, four physics ticks per decision;
- source-exact jump/dodge/resource transitions;
- ball/car/world contact authority;
- ball flip reset and car reset resource semantics;
- chain/pre-flip reset state topology;
- existing dash timing windows: `42 / 24 / 12 / 30 / 90` physics ticks;
- existing dash surface classes: `abs(n_z) >= 0.85` floor/ceiling-like, `abs(n_z) <= 0.25` wall-like, otherwise curve/transition;
- initial numerical noise floors: `1.0 uu/s` linear speed and ball delta-velocity, subject only to deterministic noise-floor evidence;
- two-touch and three-touch milestone topology;
- no quality scaling and no mechanics failure penalties.

Source-exact/discrete mechanics need focused positive/negative tests and parity only, not empirical threshold fitting.

## 5. Continuous detectors to calibrate

Calibrate exactly these nine boundaries:

1. speedflip cancel/alignment;
2. half-flip cancellation/reorientation;
3. possession/control continuity envelope and allowed gap;
4. ground carry persistence/control boundary;
5. Musty rotational-scoop separator;
6. Breezi orientation-path separator;
7. redirect incoming/outgoing velocity and direction-change separator;
8. pinch contact-overlap / normal-opposition separator;
9. pogo local-contact-region / rebound separator.

Do not add unrelated named mechanics merely because the calibration framework makes it convenient.

## 6. Calibration corpus

For each continuous detector target **72 deterministic traces**:

- 24 clear positives;
- 24 near-miss negatives;
- 24 ordinary false-positive controls.

Split each class deterministically into:

- 16 derivation cases;
- 8 held-out cases.

So each detector has 48 derivation traces + 24 held-out traces.

Use fixed case IDs and seeds. Store the scenario parameters and measured features for every trace.

### 6.1 Positive cases are physical positives, not desired outcomes

A case may be labeled positive only because its scripted/state-injected sequence actually instantiates the mechanic's physical invariant. Do not label a case positive because it is fast, scores, looks clean, or is expected to be useful.

### 6.2 Near misses must attack the decision boundary

Examples:

- speedflip: uncancelled / late-cancel diagonal flips;
- half-flip: normal backflip, tumble, stationary air-roll reversal;
- possession: ball leaves control and is later chased down;
- carry: transient roof contact / ball passing across roof;
- Musty: backward dodge contact dominated by translation rather than rotational scoop;
- Breezi: ordinary Musty with unrelated air roll; Classy-like alternate setup; random aerial correction before Musty;
- redirect: dead catch / tiny deflection / dribble tap;
- pinch: unrelated near-simultaneous wall + car contact with no opposing compression;
- pogo: wheel bounce, roof slam, chassis collision that immediately settles grounded.

## 7. Trace generation

Prefer deterministic RivalSim-native state injection plus scripted controller sequences over requiring the trained policy to discover the mechanic.

Requirements:

- run authoritative 120 Hz physics;
- use real jump/dodge/contact/vehicle physics, not analytic mock transitions;
- capture the exact pre/event/post state needed by the detector;
- keep state generation physically plausible enough that the event uses the same engine path as gameplay;
- parameterize origins where the invariant should be origin-independent (for example Musty ground/catch, air, wall/ceiling, reset-origin);
- do not require a goal or good tactical result for a positive case.

If a scripted case intended as positive fails to physically instantiate the invariant, it is not a positive trace. Fix the case instead of weakening the detector.

## 8. Threshold derivation

For each detector:

1. compute the candidate physical features on the 48 derivation traces;
2. find the closest separating positive/negative margin;
3. for a clean scalar separation, choose the midpoint of the narrowest separating margin;
4. if one scalar overlaps, add the next physically justified feature/state-machine condition from the design documents;
5. do not tune for maximum F1/recall by accepting physically different negatives;
6. do not grade event quality;
7. freeze selected thresholds only after all derivation cases classify correctly;
8. evaluate the frozen threshold on the 24 held-out traces without further fitting.

If held-out cases reveal a genuine overlap that cannot be resolved with physically justified features, mark that detector `NOT_READY_FOR_REWARD` and keep it telemetry-only. Do not invent a permissive threshold to force completion.

## 9. Detector-specific feature requirements

### Speedflip

Measure at minimum:

- actual forward-diagonal dodge onset;
- dodge-to-pitch-arrest/reversal ticks;
- integrated pitch rotation before arrest;
- residual roll/yaw path;
- time to forward-axis / surface-tangent-velocity realignment;
- pre/post surface-tangent and forward translation.

Boost may be present but cannot define the mechanic.

### Half-flip

Measure:

- actual backward dodge onset;
- pitch arrest/reversal timing;
- orientation path;
- `dot(forward_final, forward_initial)`;
- velocity projection on the new forward axis;
- supported/controllable completion state.

### Possession

Measure:

- latest legitimate touch owner;
- car-local ball position;
- relative position and velocity;
- ball-world bounce relation;
- touch/contact gap;
- reachability/continuity before the next owner touch;
- exact opponent-touch termination.

Derive an actual control envelope; do not use one guessed center-distance radius.

### Ground carry

Measure:

- car-local ball support/contact location;
- relative slip/separation velocity;
- support persistence;
- car velocity/direction change survived while retaining ball support;
- absence of pinch compression topology.

### Musty

At decisive contact retain:

- actual backward-dodge state;
- `v_contact = v_car + cross(omega_car, r_contact)`;
- `v_rot = cross(omega_car, r_contact)`;
- translation and rotation contributions to car->ball closing velocity;
- contact-point sweep path over the short pre-contact window;
- car-local ball/contact geometry;
- actual car-ball impulse / ball delta-v.

Do not freeze an arbitrary rotational percentage. Derive the separator from traces; if a scalar rotational fraction overlaps, combine it with sweep/contact geometry rather than grading power.

### Breezi

Require terminal Musty confirmation and measure:

- controlled ball relation through setup;
- full orientation sequence, not only final quaternion;
- integrated roll/yaw path;
- nose-up / inverted / nose-down ordering;
- setup duration;
- continuity of possession/control.

Delay amount and defender reaction are telemetry, not the classifier reward boundary.

### Redirect

Measure:

- incoming ball speed/direction;
- outgoing ball speed/direction;
- direction-change angle;
- contact type/context sufficient to reject ordinary dribble/catch taps.

### Pinch

Measure:

- car-ball and ball-world manifold overlap window;
- consistently oriented contact normals;
- `opposition = -dot(unit(n_car_ball), unit(n_ball_world))`;
- relative closing/compression velocities;
- impulse streams and post-release ball delta-v.

### Pogo

Measure:

- chassis-world contact point in car-local hitbox coordinates;
- wheel-support mask at impact and immediately after;
- contact-point normal velocity before/after;
- local corner/edge region;
- whether the car separates instead of immediately settling into `>=3` wheel grounding.

## 10. Read-only detector implementation

Implement enough of the detector state machines to run calibration and shadow evaluation, but **do not add mechanics reward to training**.

Preferred shape:

- compact GPU-resident 120 Hz detector state;
- canonical family/event IDs;
- subtype bitmasks/labels;
- raw classifier diagnostics for bounded evidence capture;
- family lockout/re-arm state;
- no per-world Python object in the training/eval hot path;
- no full world-state host transfer each tick.

It is acceptable for the deterministic calibration harness itself to export bounded trace evidence to host after each small case/batch. The future runtime detector must remain GPU-native.

## 11. Source-exact regression checks

Alongside continuous calibration, add focused tests proving that the read-only detector preserves:

- ball reset resource acquisition;
- car reset body identity;
- chain/pre-flip re-arm semantics;
- existing wavedash / zapdash / Rival double-dash timing/contact definitions;
- same-family subtype de-duplication;
- genuine compound events remaining separately observable.

Do not build a broad RocketSim acceptance suite.

## 12. Shadow gate — no reward

After thresholds are frozen, run **256 short episodes with mechanics reward disabled**.

Use the healthy frozen Gameplay V1 +239 Rival checkpoint for reproducibility:

`checkpoints/rival2/gameplay_v1/rival2_gameplay_resume.pt`

Expected checkpoint SHA-256:

`77BF257131FB71DDEAEAE49D668C5E25AB1D06EE26149AB0D0AE303573CA5F21`

Use stochastic Rival actions so natural behavior has some breadth.

Recommended split:

- 128 episodes versus frozen deterministic Nexto;
- 128 episodes versus frozen deterministic Wisp 75B;
- balance Rival Blue/Orange equally inside each opponent block;
- use paired/reproducible seeds where practical.

This is detector-frequency validation only. No opponent learns and Rival does not learn.

Report:

- event counts and events/minute by family/canonical event/subtype;
- bounded raw evidence samples for every family that fires;
- family lockout/re-arm counts;
- duplicate suppressions;
- impossible/pathological classification checks;
- event frequency by opponent and Rival side;
- no reward contribution (must remain exactly zero during shadow gate).

If a detector fires at obviously pathological frequency or on impossible states, stop and fix the detector before considering reward activation.

## 13. Required artifacts

Write results under:

`results/rival2/mechanics_calibration_v1/`

At minimum:

- `calibration_manifest.json`
- `thresholds.json`
- `case_results.jsonl` or compact equivalent
- `heldout_summary.json`
- `source_exact_regression.json`
- `shadow_gate_summary.json`
- bounded event-evidence samples by family

Write human-readable summary:

`docs/RIVAL2_MECHANICS_CALIBRATION_V1_RESULTS.md`

Each continuous detector record must include:

- status: `CALIBRATED` or `NOT_READY_FOR_REWARD`;
- positive/negative/control case IDs;
- features used;
- derivation extrema;
- selected threshold(s);
- separation margin;
- held-out false positives/false negatives;
- exact source commit;
- physics/observation/action contract identities/hashes where relevant.

## 14. Acceptance

Calibration is complete only if:

- source-exact regression checks pass;
- each `CALIBRATED` detector classifies every derivation trace correctly;
- each `CALIBRATED` detector passes its held-out corpus with zero false positives and zero false negatives;
- any detector that cannot cleanly separate is explicitly left `NOT_READY_FOR_REWARD` rather than force-fit;
- shadow gate shows no obvious event-spam/pathological classifier behavior;
- mechanics reward remains disabled throughout;
- no training occurs;
- no policy/PPO/obs/action/physics/lifecycle contract changes occur.

A detector failing calibration does **not** fail the whole handoff. Return the evidence and keep that family out of future reward activation until its physical classifier is improved.

## 15. Return to reviewer

Return:

1. implementation commit SHA;
2. exact source head used;
3. files changed;
4. calibration harness commands;
5. source-exact regression result;
6. a table of all nine continuous detectors with status and final measured thresholds/features;
7. held-out confusion counts;
8. shadow-gate event frequencies and any suspicious cases;
9. threshold/evidence artifact paths;
10. any `NOT_READY_FOR_REWARD` detector with the exact overlapping evidence that blocked a clean classifier;
11. confirmation that no training or reward activation occurred.
