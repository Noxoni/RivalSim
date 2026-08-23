# RivalSim v0.2.1 Gates

## Gate 0 — custody and regression

Before changing solver behavior:

- verify `origin/main` starts from the published v0.2 boundary;
- preserve `results/v0.1/` and `results/v0.2/` byte-for-byte;
- rerun v0.1 regression and require 27/27 passing;
- verify collision asset hashes and mesh-query gate remain unchanged unless a diagnosed cause requires otherwise.

## Gate 1 — divergence index

Required artifact:

`results/v0.2.1/divergence_index.json`

It must rank the frozen v0.2 failures by earliest causal divergence and identify representative cases for steering, powerslide, surface transition and chassis impact.

Pass condition: the first differing wheel/contact/impulse stage is known for each representative family, or a concrete instrumentation blocker is documented.

## Gate 2 — internal oracle

A reproducible diagnostic RocketSim reference must expose enough per-tick state to explain the selected divergences.

Pass condition:

- exact upstream revision recorded;
- diagnostic build/wrapper reproducible;
- logging does not change physics semantics;
- trace fields and stage boundaries are documented;
- representative traces can be aligned tick-by-tick with RivalSim.

## Gate 3 — representative parity

The selected representative scenarios must pass all existing hard checks and frozen numeric tolerances at all existing horizons before running the full corpus.

Do not widen tolerances.

## Gate 4 — full frozen static-world parity

Run the exact existing 35 scenarios × eight horizons.

Required:

- hard mismatch records: **0**;
- hard mismatch fields: **0**;
- numeric tolerance failures: **0**;
- all previously frozen tolerances unchanged.

If this gate fails, v0.2.1 fails. Do not proceed to final performance qualification.

## Gate 5 — regression and stress

After full parity passes:

- v0.1 corpus: 27/27 pass;
- two independent contact-rich 2,400-tick stress runs: full-state deterministic equality;
- no NaNs/nonfinite state;
- no timed H2D/D2H traffic;
- no new tracked collision assets.

## Gate 6 — corrected B3 performance

Only after Gates 0–5 pass, benchmark the corrected solver.

Required corrected complete B3 throughput:

**>=100,000 aggregate simulated game-seconds/s**

Report the measured optimum rather than assuming the previous 262,144-world optimum still applies.

Classification:

- `PASS_GREEN`: full parity + >=500,000 sim-s/s;
- `PASS`: full parity + 100,000–499,999 sim-s/s;
- `PAUSE_PERF`: full parity but <100,000 sim-s/s;
- `PAUSE_FIDELITY`: any frozen parity failure remains.

Correctness always overrides throughput.

## Stop boundary

Even on `PASS_GREEN`, stop. v0.3 dynamic contacts require a separate handoff.
