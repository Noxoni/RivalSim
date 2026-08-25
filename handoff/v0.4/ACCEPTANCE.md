# RivalSim v0.4 — Authority, Acceptance, and Evidence Rules

This document defines the validation protocol for the authorized complete standard-1v1 game-transition milestone.

## Governing principle

v0.4 is a lifecycle/state-machine milestone built on accepted v0.3 physics.

For source-defined behavior:

> freeze the native event/state transition, compare the earliest differing lifecycle operation, and fix the source translation rather than fitting downstream outcomes.

For behavior RocketSim does not define:

> freeze an explicit bounded RivalSim contract or stop for a semantics decision; do not silently invent training policy.

## Preserve prior authority

Do not regenerate prior accepted physics authority merely because v0.4 begins.

The following remain frozen mandatory regressions:

- v0.3 Phase A/B/C/D native caches and results;
- v0.2.2 static-world 39,236-case authority;
- v0.1 live RocketSim 27-scenario corpus;
- 4,608-ray correctness corpus for both arena-query backends.

Large new v0.4 authority artifacts belong under `.tools/v0.4/` and remain untracked.

## New v0.4 authority identity

Build content-addressed authority/spec identities for each lifecycle phase.

An identity must change if any input capable of changing the expected lifecycle changes, including:

- RocketSim primary/binding/package/build identity;
- installed native extension hash;
- collision assets when relevant;
- lifecycle corpus/generator source;
- generator configuration/schema;
- exact generated cases;
- seed/RNG state or explicit layout selector;
- kickoff/respawn configuration;
- game-mode/rule configuration;
- source authority settings;
- frozen non-RocketSim lifecycle contract, if one is required.

It must not change because of:

- RivalSim implementation edits;
- Warp/CUDA recompilation;
- reporting changes;
- representative-selection changes;
- benchmark configuration;
- comparison tooling changes.

Once a native cache is frozen, cached runners must have no live-RocketSim fallback. Missing/corrupt source data is an error.

## Event-driven horizons

Do **not** force v0.4 into the v0.3-only 1/4/8/12-tick shape where longer lifecycle timing is inherently part of the rule.

Use two complementary gates:

### Physics regression gate

All existing v0.3 local blocking horizons remain unchanged at ticks 1, 4, 8, and 12.

### Lifecycle gate

Validate exact event timing across the source-defined duration needed to reach the event, for example:

- pad pickup and exact reactivation tick;
- goal detection tick and scoring attribution;
- kickoff/reset transition tick;
- demolition start/disabled-state progression;
- exact respawn transition tick;
- match clock expiration/terminal state where applicable.

Long lifecycle tests must compare discrete/event/state-machine truth. Do not turn them into a requirement for chaotic long-open-loop float identity between unrelated contacts.

## Required cached/source state

Cache only the state needed to make lifecycle truth explicit.

At minimum, where relevant:

### World/match

- world tick/lifecycle counters;
- score per team;
- goal-scored event + scoring team;
- kickoff/reset state or selected layout;
- terminal/truncation/reset-required flags defined by the frozen contract;
- per-world RNG/selector state if used.

### Boost pads

- all 34 pad active states;
- cooldown/recharge timers or source-equivalent state;
- pickup event and receiving car;
- boost before/after pickup where relevant.

### Cars

- normal v0.3 physical state at event boundaries;
- demoed/disabled state;
- demo/respawn timer;
- collision/broadphase participation state where source exposes it;
- respawn pose/state;
- car visitation-order lifecycle state;
- membership epoch/state if a true insertion/removal occurs.

### Ball

- normal physical state at reset/goal boundaries;
- goal-detection state/event;
- kickoff/reset state.

Use compact ordinary trajectory/event records. Deep native internals belong in selected diagnostic traces.

## Initial and reset-state custody

For every deterministic reset/kickoff/respawn case, store:

1. exact requested source state/selector/RNG input;
2. immediate native readback after the lifecycle operation;
3. any internal lifecycle state that causally affects subsequent transitions.

Do not reconstruct authoritative matrix/quaternion, timer, container-order, RNG, or pad state through lossy public round trips if a more authoritative representation exists.

## Phase A gate — boost pads and lifecycle foundation

Freeze a deterministic corpus covering:

- all 34 pads;
- both cars;
- small/full pad classes;
- pickup while eligible;
- pickup while inactive;
- exact cooldown boundary before/at/after recharge;
- simultaneous/near-simultaneous contenders;
- car entering/leaving pickup radius around boundary ticks;
- full-world reset;
- kickoff/reset interaction with pad state;
- repeated pickup/recharge cycles.

Hard discrete comparisons must include:

- correct pad identity;
- correct receiving car;
- correct active/inactive transition;
- correct boost grant;
- exact recharge event tick;
- correct reset behavior.

The already-source-backed v0.2.x pad implementation is a regression baseline, not permission to skip integrated v0.4 testing.

## Phase B gate — goals/scoring/kickoff

Freeze cases spanning:

- both goals;
- multiple approach directions/speeds;
- near-goal but not-scored boundary states;
- wall/goal-interior contacts;
- scoring while car/ball contacts are active;
- scoring while pad/demo timers are nontrivial where valid;
- every standard 1v1 kickoff layout used by the chosen source contract;
- repeated goal -> reset -> goal cycles.

Hard comparisons:

- goal present/absent;
- scoring team;
- exact scoring event tick;
- score counter transition;
- ball reset state;
- both car reset states;
- lifecycle timers/events cleared or preserved exactly as source defines;
- boost-pad reset/preservation state;
- car visitation order preserved unless true membership reconstruction occurs.

If random kickoff selection is supported, compare the explicit selector/RNG state and complete selected layout. Never infer a layout from expected output.

## Phase C gate — demolition disable/respawn

Freeze cases spanning:

- both cars as victim/attacker;
- source-valid demolition predicates already proven by v0.3;
- grounded/aerial demos;
- static/dynamic contact context;
- exact timer boundaries;
- controls during demo state;
- collision/broadphase inactivity while demoed;
- source-valid respawn locations/orientations;
- repeated demos and respawns;
- demo across goal/kickoff/reset lifecycle boundaries where source behavior is defined.

Hard comparisons:

- demolition event classification remains exact;
- victim disabled state begins at the correct tick;
- physical/collision participation while disabled is correct;
- respawn occurs at the correct tick;
- respawn pose/state is correct;
- car identity/container visitation semantics are preserved where membership is unchanged;
- v0.3 post-demo numeric masks are removed only when the new lifecycle makes those frames authoritative.

## Phase D gate — complete game transition

Build a bounded integrated corpus and longer deterministic episode traces covering at minimum:

- ordinary play + repeated pad cycles;
- goal -> scoring -> kickoff/reset -> resumed physics;
- demo -> disabled interval -> respawn -> resumed physics;
- multiple goals and kickoffs in one world lifetime;
- goal during active pad cooldowns;
- goal/demo ordering neighborhoods;
- respawn followed by immediate ordinary play;
- reset with active contacts/manifolds;
- explicit full-world reset;
- world independence across a GPU batch;
- both v0.3 car-visitation-order branches where the lifecycle does not collapse them.

The integrated gate must compare one coherent native/spec trajectory at a time. Do not mix events or metrics from alternate valid branches.

## Terminal/truncation contract

v0.4 may expose generic terminal/truncation lifecycle outputs but must not implement reward or PPO policy.

If the chosen standard 1v1 authority defines regulation time, overtime, score limit, or reset termination, freeze it and test exact discrete transitions.

If no authoritative source exists, either:

- expose raw deterministic events/state for v0.5 to compose; or
- freeze a clearly documented v0.4 standard-lifecycle contract after an explicit semantics decision.

Do not silently copy arbitrary RLGym defaults into the simulator core.

## Deep native traces

On a failing representative/full lifecycle gate, rank by:

1. earliest wrong event tick;
2. hard semantic failure before numeric drift;
3. mechanism diversity;
4. largest normalized physical error only after event semantics.

Trace only the active rule path needed to find the first causal mismatch. Candidate fields:

- pad pickup query/order/timer state;
- scored-ball test and attribution;
- reset/kickoff function entry/selected spawn/state writes;
- demo lifecycle state machine;
- collision-body disable/re-enable/broadphase state;
- respawn selector/timer/state writes;
- car-container membership/visitation state;
- score/match clock/event state transitions;
- per-world RNG/selector state.

Build machine-readable comparators instead of repeatedly interpreting logs manually.

## Failure policy

### Blocking

- wrong goal/no-goal result;
- wrong scoring team or tick;
- wrong score;
- wrong kickoff/respawn pose or selection;
- wrong pad pickup/recharge state or tick;
- wrong demo disable/respawn state or tick;
- car participating in physics while source says disabled, or vice versa;
- wrong visitation-order lifecycle caused by reset/demo handling;
- cross-world state leakage;
- non-deterministic same-seed reset/lifecycle behavior;
- stale authority/cache reuse;
- any v0.3/v0.2.2/v0.1 regression;
- systematic physical tolerance failures introduced by rules.

### Nonblocking only after semantics are exact

Tiny physical float residuals may remain nonblocking only under the already-frozen physics tolerances when they do not alter an event, branch, reset, or later 12-tick physics gate.

Do not add timing tolerance, event hysteresis, case rules, or expected-output selection to hide lifecycle failures.

## Determinism and batch invariance

Prove that:

- same source state + same lifecycle/RNG state => same output;
- batch position/world count does not alter a world's lifecycle;
- independent repeated runs produce identical full-state/event hashes under deterministic configuration;
- no authority case shares mutable native state with another case in a way that changes truth.

Stress should include repeated pads, goals, kickoffs, demos, respawns, and full resets.

## Performance gates

Do performance work only after fidelity/regression gates are green.

Measure at minimum:

### Complete game-transition benchmark

The accepted v0.3 physics + v0.4 lifecycle/rules at practical large world counts.

Report:

- world ticks/s;
- aggregate simulated game-seconds/s;
- coefficient of variation;
- peak/device VRAM;
- timed H2D/D2H bytes;
- comparison to v0.3 complete-dynamic 196,614.39 sim-s/s.

Required viability floor: **100,000 aggregate simulated game-seconds/s** on the complete v0.4 path.

### Reset-heavy benchmark

Use a deterministic workload with intentionally frequent goals/kickoffs, demos/respawns, and/or explicit resets so reset throughput and allocation behavior are visible.

Report:

- resets or lifecycle transitions/s;
- world ticks/s / sim-s/s;
- whether any host synchronization or allocation enters the timed path;
- variance and VRAM behavior.

There is no separate arbitrary reset-heavy PASS number unless evidence justifies one, but a pathological reset bottleneck is a genuine v0.4 boundary and must be reported before v0.5.

Do not weaken source behavior to preserve v0.3 throughput.

## Regression gates

Before v0.4 can complete:

1. v0.3 Phase A: 31,216 / 31,216;
2. v0.3 Phase B: 8,192 / 8,192;
3. v0.3 Phase C: 8,192 / 8,192 against both native-valid branches;
4. v0.3 Phase D: 512 / 512 across both branches;
5. v0.2.2 static: 39,236 / 39,236;
6. v0.1 live RocketSim: 27 / 27;
7. both arena ray backends remain green;
8. repository tests/lint/compile/diff checks pass;
9. deterministic mixed physics + lifecycle stress passes;
10. prior published evidence remains byte-for-byte unchanged.

## Final evidence package

Large v0.4 authority, event traces, diagnostics, and benchmark chunks remain ignored under `.tools/v0.4/`.

Commit compact evidence sufficient to verify custody and outcome. Suggested layout:

- `results/v0.4/boost_pads.json`;
- `results/v0.4/goals_kickoff.json`;
- `results/v0.4/demolition_respawn.json`;
- `results/v0.4/match_lifecycle.json`;
- `results/v0.4/oracle_data.json`;
- `results/v0.4/rules_source.json`;
- `results/v0.4/regression.json`;
- `results/v0.4/benchmark.json`;
- `results/v0.4/manifest.json`;
- `docs/V0_4_RESULTS.md`;
- `docs/REPRODUCING_V0_4.md`;
- `docs/V0_4_AUTHORITY.md`.

The final manifest must identify:

- v0.3 release and implementation baseline;
- v0.4 implementation commit;
- all authority/spec identities;
- lifecycle corpus hashes;
- source/tool/config hashes;
- compact evidence hashes;
- prior evidence hashes;
- final lifecycle/regression/performance verdicts;
- explicit `v0_5_begun: false` (or equivalent).

Do not publish `PASS_GREEN` unless every v0.4 completion gate passes.
