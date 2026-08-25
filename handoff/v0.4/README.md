# RivalSim v0.4 — Complete Standard 1v1 Game Transition

Status: **AUTHORIZED / NOT YET IMPLEMENTED**

This handoff authorizes the bounded game-lifecycle milestone on top of the completed v0.3 dynamic-physics release.

## Frozen starting boundary

Release commit:

`d6ca3912418a3dd7ca8979415142cd861e0c0ddb`

Implementation commit:

`a63d317b0de0522e6d3cbe243bf282c6b93a9d58`

v0.3 is `PASS_GREEN` and is a mandatory regression baseline:

- Phase A ball/world: 31,216 / 31,216;
- Phase B car/ball: 8,192 / 8,192;
- Phase C car/car: 8,192 / 8,192 against both complete native-valid visitation branches;
- Phase D integrated dynamic contacts: 512 / 512 across eight simultaneous-contact families and both branches;
- v0.2.2 static-world acceptance: 39,236 / 39,236;
- v0.1 live RocketSim: 27 / 27;
- complete dynamic throughput: 196,614.39 aggregate simulated game-seconds/s at 131,072 worlds;
- zero timed hot-loop transfers and deterministic mixed-contact stress.

Do not rewrite, delete, regenerate, or weaken published `results/v0.1/` through `results/v0.3/` evidence.

## Mission

Turn the accepted two-Octane/one-ball physics engine into a complete **headless standard Soccar 1v1 game transition engine**.

v0.4 adds only lifecycle/rules/state needed to run complete standard 1v1 worlds:

1. complete integration of the existing 34 standard Soccar boost pads;
2. goal detection, goal attribution, and scoring state;
3. source-correct kickoff/reset poses and reset behavior;
4. demolition disable/removal-from-physics, respawn timer/state, and respawn transition;
5. bounded match/reset state and explicit lifecycle events;
6. generic terminal/truncation outputs sufficient for the later training integration layer;
7. deterministic per-world reset/lifecycle state suitable for GPU batching.

The milestone succeeds when RivalSim can advance a standard two-Octane/one-ball 1v1 world through repeated ordinary play, pad pickups/recharge, goals, kickoffs, demolitions, respawns, and resets without CPU-side game-state intervention.

## Preserve the accepted physics core

Treat v0.3 physics as frozen unless new evidence proves a regression is caused there.

Do not redesign:

- ball/static collision;
- car/ball collision;
- car/car collision;
- bump/demo **classification** physics;
- dynamic wheel rays;
- static or dynamic broadphase pair ordering;
- connected-island ordering;
- shared constraint solver;
- rigid-body integration;
- per-world car `_PreTickUpdate` visitation-order lifecycle.

v0.4 completes what happens **after** game events already produced by those systems.

## Source and authority hierarchy

For behavior RocketSim defines, primary authority remains the exact pinned source/build already used by v0.3:

- RocketSim primary commit `c2baacb8f4b441dd8505e63c2aeb5a1679b60b02`;
- Python-binding lineage `2da51b1dac7b8127127613a5ff30e490bdd70dd8`;
- package/build identity `rocketsim==2.2.1` plus the installed extension hash recorded by evidence.

Before implementing each rule family, map the exact source/lifecycle path and record the relevant constants/order.

Do not assume that a physically intuitive rule matches RocketSim. Validate source behavior for:

- pad pickup/deactivation/recharge/reset;
- goal/scored-ball detection and team attribution;
- kickoff/reset car and ball states;
- demolition disable state and collision participation;
- respawn timing/pose/state restoration;
- whether resets reconstruct objects or mutate existing objects;
- effects on broadphase lifecycle and car visitation order.

If RocketSim does **not** define a training-facing terminal/truncation policy, do not silently invent one. v0.4 may expose generic source events (`goal_scored`, `match_time_expired`, `reset_required`, etc.) and a frozen bounded RivalSim lifecycle contract; v0.5 owns training-policy composition.

## Critical lifecycle invariants from v0.3

The native-order experiment is controlling evidence:

- construction or actual car-container membership change establishes/may change car visitation order;
- ordinary ticks preserve that order;
- physical state/kickoff-style resets preserve it;
- demolition and respawn preserve it when `_cars` membership does not change.

Therefore a demo must **not** be modeled as removal/reinsertion into the logical car container merely because the car is temporarily absent from physics. Preserve the same per-world visitation-order state unless the exact pinned source performs a real membership mutation.

Likewise, broadphase disable/re-enable and object activity must follow source lifecycle rather than being approximated as world reconstruction.

## Phase A — lifecycle foundation + boost pads

Integrate the existing source-backed 34-pad state into the complete v0.3 world.

Required behavior includes:

- small/full pad identity and boost grant;
- pickup eligibility;
- active/inactive state;
- cooldown/recharge timing;
- simultaneous/near-simultaneous pickup semantics;
- pad state under kickoff/reset/full-world reset;
- deterministic per-world pad lifecycle with no host polling.

Do not rederive already-green pad arithmetic unless source evidence shows the integrated dynamic world changes it.

Freeze an authority corpus that covers all 34 pads, cooldown boundaries, overlapping pickup attempts, both cars, and reset transitions.

## Phase B — goals, scoring, kickoff/reset

Implement source-correct standard Soccar goal lifecycle.

Required behavior:

- scored-ball detection;
- scoring-team attribution;
- score counters/event state;
- exact transition boundary at the scoring tick;
- standard kickoff ball state;
- standard 1v1 kickoff car poses/orientations/state;
- all source-valid kickoff layouts needed by the standard two-car path;
- boost/pad/demo/contact state after kickoff/reset as defined by source;
- preservation or recreation of internal lifecycle state exactly where the source does so.

Do not add replay cameras, goal explosions, presentation countdowns, rendering, or other client/UI behavior.

Prefer an explicit deterministic kickoff-layout selector for validation. If random selection is supported, its RNG state/seed must be explicit per world and included in authority identity. Never rely on host-global randomness.

## Phase C — demolition lifecycle and respawn

v0.3 already hard-gates bump/demo classification. v0.4 must implement the lifecycle that follows a demolition.

Required behavior:

- transition into demoed/disabled state;
- removal from active collision/solver participation as the pinned source does it;
- timer progression;
- controls/physics behavior while demoed;
- respawn timing;
- source-valid respawn location/orientation selection;
- restored velocity/angular/boost/jump/contact/control state as defined by source;
- re-entry to broadphase/physics;
- preservation of car object/container identity and visitation order when membership is unchanged.

The v0.3 post-demo numeric masks may be removed only after the source-correct demo lifecycle makes those frames authoritative. The bump/demo predicate and ordered event stream remain hard regressions.

If respawn selection has multiple legitimate outcomes, represent the causal RNG/layout state generically; do not create per-case expected-output tables.

## Phase D — complete match/reset transition

Compose the accepted physics and Phase A/B/C rules into complete headless standard 1v1 worlds.

At minimum support and validate:

- ordinary play through repeated pad cycles;
- goal -> score event -> kickoff/reset -> resumed play;
- demo -> disabled interval -> respawn -> resumed play;
- goals while pads or demo timers are active;
- resets while cars/ball have active contact/manifold state;
- repeated goals/kickoffs without arena reconstruction unless source requires it;
- deterministic explicit full-world reset;
- score/match/tick/event state surviving or clearing at the correct lifecycle boundaries;
- world independence across large GPU batches.

Add generic lifecycle outputs needed by v0.5 to decide episode termination/truncation. Keep them policy-neutral unless an exact standard contract is frozen in v0.4.

## Match clock / terminal boundary

Do not smuggle training-policy choices into physics.

If the standard headless 1v1 contract includes a match clock, overtime, score limit, or timeout in the chosen authority, implement and validate that exact contract.

If RocketSim has no authoritative equivalent for a requested terminal/truncation rule, expose enough deterministic state/events for v0.5 to implement the training condition later, or stop and document the missing policy decision.

No reward function belongs in v0.4.

## Randomness and reset policy

Any stochastic lifecycle decision must be explicit and reproducible:

- per-world RNG state or an explicit selected-layout value;
- no Python/global host RNG inside the hot loop;
- no dependence on pointer addresses, heap layout, process order, or expected outputs;
- RNG/configuration inputs included in authority identity;
- same seed/state => identical transition.

Batching must not alter lifecycle outcomes.

## Performance architecture

Game logic must remain GPU native.

Target path remains:

`GPU world state -> physics -> game events/rules -> next GPU world state`

Do not introduce routine host round trips for:

- scoring;
- pad timers;
- demo timers;
- kickoff/reset;
- match clocks;
- terminal/truncation flags.

A reset-heavy workload matters as much as steady-state play because v0.5 will run many parallel episodes.

## Explicitly out of scope

Do **not** implement in v0.4:

- observation construction;
- action parsing beyond the existing controller path;
- reward functions;
- rollout buffers;
- GAE/PPO;
- PyTorch integration;
- Rival policy inference;
- curriculum/mutator sampling beyond the minimum explicit lifecycle/reset selector required here;
- arbitrary car/ball counts;
- Hoops, Dropshot, Snowday, Heatseeker, Rumble, or other modes;
- rendering, replay, client countdown/UI;
- generic Bullet or generic Rocket League client compatibility.

## Completion boundary

v0.4 may be published `PASS_GREEN` only if:

1. all authorized lifecycle families pass their frozen authority/spec gates;
2. complete standard two-Octane/one-ball episode transitions run headlessly on GPU;
3. v0.3 Phase A/B/C/D remains fully green;
4. v0.2.2 static and v0.1 live regressions remain fully green;
5. deterministic repeated reset/demo/goal stress is finite and reproducible;
6. routine hot-loop game transitions require zero timed H2D/D2H transfers;
7. complete game-transition throughput remains >=100,000 aggregate simulated game-seconds/s at a stable practical batch;
8. reset-heavy throughput is measured and reported rather than hidden;
9. compact evidence/manifest is committed and remotely verified;
10. v0.5 has not begun.

If a source semantics decision or performance boundary prevents completion, stop there with explicit evidence. Do not weaken physics/rules or begin training integration to work around it.
