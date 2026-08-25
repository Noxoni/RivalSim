# RivalSim v0.4 — Lifecycle and Rule Implementation Policy

This policy exists to keep v0.4 from turning accepted physics into a hand-fitted game script.

## 1. Source-first, event-first

For every rule family, identify the exact source/lifecycle behavior before implementing GPU code.

Prefer direct translation of the pinned RocketSim operation/state-machine path where it exists.

Parity failures are evidence that the translation or authority is incomplete; they are not permission to invent a special-case rule.

## 2. Keep physics and game rules separated

v0.3 owns accepted physical interaction:

- ball/world contact;
- car/ball contact;
- car/car contact;
- bump/demo physical classification;
- wheel/suspension interaction;
- shared solver and integration.

v0.4 owns lifecycle state that consumes those events:

- pad pickup/recharge/reset;
- goal/scoring state;
- kickoff/full reset;
- demo disable/respawn;
- score/match/event state;
- generic terminal/truncation outputs.

Do not compensate for a game-rule mismatch by perturbing accepted collision impulses, contact normals, timing tolerances, or solver arithmetic.

## 3. Demolition is not car-container deletion unless source says so

The v0.3 native lifecycle experiment proved that ordinary RocketSim demolition/respawn preserves the existing `Car` and `_cars` membership.

Therefore:

- preserve the car's logical identity;
- preserve its per-world visitation-order state;
- disable/remove it from active physics exactly as source does;
- re-enable/respawn it exactly as source does;
- do not erase/reinsert the car merely to make GPU bookkeeping convenient.

Only a real source membership mutation may establish a new visitation order.

## 4. Reset semantics must be explicit

Distinguish:

- physical state reset on an existing arena;
- kickoff reset on existing objects;
- full world reset;
- arena reconstruction;
- true insertion/removal.

These operations are not interchangeable. They may differ in:

- car visitation order;
- broadphase lifecycle;
- pad state;
- demo state;
- score/match state;
- RNG/layout selection.

Implement separate source-backed paths where semantics differ.

## 5. Randomness is state, not ambient behavior

Any kickoff/respawn/random lifecycle selection must use explicit per-world state.

Allowed:

- deterministic layout index;
- explicit seed + per-world RNG state;
- source-backed discrete selector stored in world state.

Prohibited:

- Python/global random calls in the hot loop;
- process/launch order as hidden state;
- pointer/allocator state;
- expected-output-guided selection;
- case IDs used to choose behavior.

Authority must record the causal selector/RNG state.

## 6. Native multi-outcome behavior must remain relational

v0.3 established two complete native-valid car-visitation branches.

If v0.4 lifecycle does not change car membership, preserve the branch and compare one complete labeled trajectory against its corresponding native-valid outcome.

Never:

- mix a score/event from one branch with physical state from the other;
- choose the branch after seeing expected output;
- collapse the branches with a tolerance;
- create a per-case branch lookup table.

If a source lifecycle event genuinely reconstructs membership and establishes a new order, model that lifecycle causally rather than preserving a stale branch.

## 7. Hard discrete rules are exact

These are exact/hard semantics, not numeric tolerances:

- which pad was picked up;
- which car received boost;
- whether a goal occurred;
- scoring team;
- score counter;
- kickoff layout/selector;
- demoed/disabled state;
- respawn event/tick;
- respawn selector/pose identity;
- match/reset/terminal event flags;
- car membership/visitation lifecycle changes.

A wrong discrete event blocks even if positions/velocities remain numerically close.

## 8. Do not use presentation behavior as authority

RivalSim is a headless training transition engine.

Do not add unless proven necessary for physical/lifecycle state:

- goal explosions;
- replay delay;
- cameras;
- HUD/countdown rendering;
- sound;
- client animation;
- menu/matchmaking state.

If Rocket League presentation has a delay but RocketSim's headless lifecycle does not, follow the chosen headless authority contract.

## 9. Terminal/truncation policy must be explicit

RocketSim may not own every training-facing episode decision.

Do not embed an arbitrary reward/training convention into core world physics.

Preferred v0.4 output is deterministic raw lifecycle state/events from which v0.5 can build:

- goal terminal;
- timeout truncation;
- match-over conditions;
- curriculum-specific termination.

If v0.4 itself freezes a standard match terminal contract, document its authority and make it configurable rather than disguising it as physics.

## 10. Avoid host-side lifecycle escape hatches

Do not solve reset/rule complexity by making Python or CPU code mutate thousands of worlds every event.

The steady-state and reset-heavy paths should remain device resident. Host interaction is for:

- configuration;
- evidence/metrics;
- checkpoints;
- explicit external control outside timed hot loops.

## 11. Reuse accepted implementation where possible

The existing 34-pad code and v0.3 event classifications already have source-backed behavior.

Integrate and validate them before rewriting them.

The default question should be:

> what lifecycle state is missing around this accepted mechanism?

not:

> how can we redesign the mechanism while adding game rules?

## 12. Stop at genuine semantics boundaries

If source inspection shows an event depends on hidden state absent from the current authority/input, stop and characterize it exactly before altering schema or runtime semantics.

The v0.3 car-order investigation is the model:

1. prove the missing causal state;
2. measure its lifecycle;
3. represent the minimum generic state/event required;
4. prohibit expected-output fitting;
5. resume only after the semantics decision is explicit.

## 13. No v0.5 creep

Do not add:

- observations;
- reward shaping;
- action masks/parsers beyond current controls;
- PPO/GAE;
- rollout buffers;
- PyTorch policy inference;
- learner synchronization;
- curriculum scheduling.

v0.4 ends when the complete headless game transition is accepted and benchmarked.
