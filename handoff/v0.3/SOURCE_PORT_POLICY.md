# RivalSim v0.3 — Source-Port Policy

This policy is binding for v0.3 dynamic contacts.

## Primary rule

If the pinned RocketSim/Bullet source defines the behavior, **port that behavior before inventing an approximation**.

Parity is a validator of the port. It is not the primary method for reverse-engineering known source behavior.

## Required workflow for each new interaction

For ball-world, car-ball, and car-car independently:

1. Identify the exact native call chain used by the pinned build.
2. Record the reachable shapes, margins, constants, solver settings, callbacks, and RocketSim-specific logic.
3. Identify which existing v0.2.2 GPU mechanisms are truly shared and which are not.
4. Translate the bounded source behavior into fixed-size GPU data structures/functions.
5. Preserve source operation order where float32 branch behavior can depend on it.
6. Compare against frozen native authority at ticks 1/4/8/12.
7. On failure, compare operation-level traces and locate the **first** divergent operation.
8. Correct the translation at that operation.
9. Rerun the representative cached gate.
10. Advance only after the representative gate is clean.

## What "source port" means here

The GPU implementation does **not** need to preserve Bullet's CPU object graph, virtual dispatch, general-purpose containers, or API surface.

It may restructure:

- memory into SoA arrays;
- pair/contact buffers into fixed-capacity GPU storage;
- loops into kernels;
- source recursion into bounded iterative storage;
- object-owned state into per-world arrays;
- general shape dispatch into the exact fixed shape pairs RivalSim supports.

It must preserve the physics semantics that affect the native result:

- constants and units;
- support/witness math;
- margins;
- branch conditions;
- candidate/pair ordering when authoritative;
- manifold reduction/refresh;
- callback behavior;
- solver row construction;
- iteration ordering;
- restitution/friction calculations;
- force/torque ordering;
- rigid-body integration ordering;
- strict comparisons and tie behavior;
- RocketSim-specific pre/post-processing.

## Float32 discipline

Do not assume mathematical equivalence implies float32 equivalence.

The v0.2.2 source port found real downstream branch changes from differences as small as one ULP. Examples included:

- reassociated weighted vector sums;
- matrix/quaternion reconstruction order;
- UU ↔ Bullet unit round trips;
- SSE reciprocal-square-root behavior;
- matrix/vector multiplication orientation and reduction order;
- direct Bullet-unit coefficient calculation versus algebraically equivalent scaling;
- strict first-maximum / first-minimum lane ordering.

For v0.3:

- default to ordinary GPU arithmetic where it produces equivalent local transitions;
- when a cached trace proves instruction/operation order changes a meaningful branch, translate the relevant source order narrowly;
- do not globally emulate x86/SSE semantics without evidence that the reachable path requires it.

## Prohibited behavioral fitting

Do not fix parity by adding behavior that is absent from the pinned authority source, including:

- face IDs or triangle-specific branches;
- case IDs;
- tick-specific branches;
- hand-authored reference outputs;
- hidden lookup tables keyed by authority results;
- contact epsilons introduced only to force the expected edge;
- near-tie tolerances replacing strict source comparisons;
- hysteresis absent from the source;
- downstream wheel/solver compensation for an upstream collision error;
- suppressing a source-correct force merely because it magnifies a prior divergence;
- widening acceptance tolerances to hide a systematic error.

A generic optimization/approximation is allowed only when it is proven equivalent for the authoritative local transition space or when the user explicitly accepts the deviation.

## No scenario-name magic

The runtime simulator must never know or care which validation case is running.

Diagnostics may select a case ID for tracing, but production physics may not branch on it.

## Preserve accepted subsystems

Do not reopen a v0.2.2 subsystem merely because a later trajectory differs.

Before modifying an accepted subsystem, prove with identical incoming native/GPU state that its first operation differs.

If the subsystem is exact at identical inputs, move upstream and find why the inputs differ.

This rule prevented repeated false fixes during v0.2.2 and remains mandatory.

## Source-map artifact

Before implementing each v0.3 phase, create a compact local source map containing at least:

- pinned source file;
- function/class;
- caller/callee chain;
- relevant constants;
- shape types;
- margin configuration;
- solver/contact path;
- RocketSim-specific modifications;
- GPU function/kernel intended to implement it;
- known operation-order hazards;
- validation family that exercises the path.

The final committed `source_port.json` should summarize this provenance without committing upstream source files.

## Native diagnostic policy

The existing diagnostic executable may be extended to expose values from the pinned source.

Allowed:

- logging internal state;
- emitting candidate/pair order;
- recording GJK/EPA/simplex iterations;
- recording contact/manifold history;
- recording solver rows/impulses;
- recording RocketSim-specific collision classification;
- emitting body transforms and integration state.

Not allowed:

- changing authority constants;
- changing branch behavior;
- changing solver iterations;
- patching contacts to make them easier to compare;
- changing world semantics for convenience.

Record diagnostic source hashes and rebuild provenance in the v0.3 evidence.

## Cached-truth policy

Once a phase's native authority cache is frozen, normal GPU iterations must not relaunch RocketSim.

Use:

`cached native truth -> changed GPU implementation -> automatic comparison -> first divergent operation`

If deeper native information is needed for a failing case, generate a new content-addressed deep trace from the same authority identity once, validate it, then reuse it.

Do not repeatedly pay for the same native calculation.

## Stopping boundaries

Stop and report rather than silently broaden scope if clearing a phase requires:

- a generic Bullet port beyond the shape/body paths required by standard 1v1;
- game-rule implementation belonging to v0.4;
- training integration belonging to v0.5;
- changing the pinned RocketSim authority;
- weakening frozen parity tolerances;
- abandoning GPU residency;
- adding behavioral fitting.

A clear negative/blocked boundary is preferable to disguising an unsupported physics path.

## Final standard

The desired end state is:

> RivalSim uses a GPU-specialized implementation, but for the bounded standard-Soccar dynamic-contact paths it follows the same source-defined local physics decisions as the pinned RocketSim/Bullet authority.

That is the v0.3 fidelity target.
