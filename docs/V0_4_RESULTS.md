# RivalSim v0.4 Results

## Verdict

RivalSim v0.4 is complete with **`PASS_GREEN`**.

The fixed standard-Soccar world containing exactly two Octanes and one ball now performs the
accepted v0.3 physics plus standard boost-pad, goal/scoring, kickoff, demolition, respawn, event,
clock, and reset transitions on the GPU. Every v0.4 lifecycle, inherited fidelity, regression,
determinism, residency, ray, repository, and performance gate passed. Work stopped at the v0.4
boundary; no v0.5 training-integration work began.

## Implemented boundary

The public `CompleteWorldSim` adds:

- persistent state for all 34 standard Soccar boost pads, with source grants, pickup geometry,
  prior-lock behavior, float32 cooldown/recharge, pickup/reactivation events, and reset state;
- strict scored-ball detection, source team attribution, score counters, first-entry goal events,
  and an explicit choice to compose the binding callback with an immediate deterministic kickoff;
- all five source-valid standard two-car kickoff layouts, exact car/ball readback state, and
  source-defined pad/demo/contact cleanup;
- demolition transition, disabled collision/solver participation, frozen public car state,
  exact timer progression, all four source-valid respawn locations for both teams, clean re-entry,
  and preservation of car membership/visitation order;
- monotonic world tick, resettable episode tick, raw goal/pad/demo/respawn/kickoff/full-reset
  events, score state, reset-required state, and policy-neutral terminal/truncation fields;
- deterministic host full reset and a CUDA-graph-compatible resident interval-reset path.

The existing `IntegratedWorldSim` and all accepted v0.3 collision, manifold, solver, integration,
and multi-outcome visitation behavior remain the physics substrate. v0.4 did not change the
frozen v0.3 authority or tolerances.

## Native lifecycle authority

The final content-addressed authority identity is:

`33AA0BA3BC35BC4300E2D2B84A3813CB0AD776479546A50AC3BBC6CE3D3E2562`

It binds RocketSim primary commit `c2baacb8f4b441dd8505e63c2aeb5a1679b60b02`, binding commit
`2da51b1dac7b8127127613a5ff30e490bdd70dd8`, installed `rocketsim==2.2.1` extension SHA-256
`E3EE24CA82445B4BFCC754583F6778D7B0D8B7A7F7D64F872BE8C65E621A63D0`, all 16 external
Soccar CMFs and combined SHA-256
`2239556BDC74D205CAA6E46A0F6E91FA2C6E4257E84D4F608BA775958B0A5538`, generator
source/config/seed, authority settings, and the bounded RivalSim selector/event contract.

| Phase | Authority coverage | Result |
| --- | --- | --- |
| A — pads | 34 pads × 2 cars; large/small cooldown boundaries; both contention orders | 68/68 pickup cases; `PASS_GREEN` |
| B — goals/kickoff | both goal directions; inside/equal/outside strict boundary; five layouts; reset composition | 6/6 boundary cases and 5/5 layouts; `PASS_GREEN` |
| C — demolition/respawn | two teams × four locations; timer ticks 0/1/358/359/360/361 | 8/8 poses; exact tick-360 respawn; `PASS_GREEN` |
| D — complete lifecycle | 64 worlds × 400 ticks; goals, pads, demos, interval resets, both car orders | identical repeat hashes and zero timed transfers; `PASS_GREEN` |

Large native records remain ignored under `.tools/v0.4/oracle/<identity>/`. The frozen cache is
complete, verifies its authority JSON hash, and has no live fallback from an acceptance runner.

## Exact discrete results

- Large pads reactivate on tick 1,201 from a 10-second float32 cooldown.
- Small pads reactivate on tick 480 from a four-second float32 cooldown.
- With both cars overlapping one active pad, `a_then_b` gives the pickup to B and `b_then_a`
  gives it to A, matching the source's last-lock behavior under persistent visitation order.
- Goal detection is strict: `abs(ball_y) > 5124.25 + 91.25`. Equality is not a score.
- Positive goal Y attributes the score to blue/team 0; negative goal Y to orange/team 1.
- Kickoff layouts are explicit per-world state and advance modulo five.
- Demolition begins at 3.0 seconds, reaches `0.008321664296090603` at relative tick 359, and
  respawns on relative tick 360.
- Respawn selection is explicit internal state and advances modulo four.
- Kickoff and respawn do not mutate car-container membership, so the v0.3 per-world visitation
  state is preserved.
- RocketSim defines no training episode termination policy. `terminated` and `truncated` remain
  zero while raw events, scores, clocks, and reset-required state are exposed for v0.5.

## Regression and integrity

| Gate | Result |
| --- | ---: |
| v0.3 Phase A ball/world | 31,216 / 31,216 |
| v0.3 Phase B car/ball | 8,192 / 8,192 |
| v0.3 Phase C car/car | 8,192 / 8,192 against both complete branches |
| v0.3 Phase D integrated | 512 / 512 across both complete branches |
| v0.2.2 static-world breadth | 39,236 / 39,236 |
| v0.1 live RocketSim | 27 / 27 |
| default Warp ray backend | 4,608 / 4,608 gate pass |
| cuBQL ray backend | 4,608 / 4,608 gate pass |
| configured repository tests | 70 / 70 |

Both ray backends record zero hit mismatches, zero unambiguous face mismatches, maximum nearest
distance error 0.001953125 uu, and minimum checked normal dot 0.9999998807907104. Ruff, Python
bytecode compilation, and `git diff --check` pass. Published `results/v0.1/` through
`results/v0.3/` have zero byte differences from authorized baseline
`b5875c4b853a8ce844d0904e989b1d2a3854d0ac`; aggregate tree-listing hashes are in
`results/v0.4/regression.json`.

Two independent 64-world, 400-tick mixed physics/lifecycle executions produced the same complete
state SHA-256:

`B650F689BA630523A6E834EA96E53D73EF2144AED23DF849BA2B138BD1E2669F`

## Performance

Measured environment:

- NVIDIA GeForce RTX 5090, driver 610.62, 32 GiB;
- AMD64 Family 26 Model 68, 8 physical / 16 logical CPU cores;
- Windows 11, Python 3.14.3, NumPy 2.5.2, Warp 1.16.0;
- 120 Hz, device-resident 64-entry action tape, CUDA graph blocks of eight ticks;
- five repeats after warmup; 5% CV stability gate.

The best stable complete-game-transition point is:

| Worlds | World ticks/s | Aggregate simulated game-seconds/s | CV | Peak observed VRAM | Timed transfers |
| ---: | ---: | ---: | ---: | ---: | --- |
| 131,072 | 23,009,771.59 | **191,748.10** | 0.856% | 11,092,152,320 bytes | 0 H2D / 0 D2H |

This is 1.92× the required 100,000 sim-s/s viability floor and retains 97.52% of the v0.3
complete-dynamic reference of 196,614.39 sim-s/s.

The reset-heavy point performs a deterministic full reset every eight ticks:

| Worlds | World ticks/s | Aggregate simulated game-seconds/s | Reset transitions/s | CV | Peak observed VRAM | Timed transfers |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 131,072 | 27,000,607.06 | **225,005.06** | **3,375,075.88** | 0.723% | 11,104,145,408 bytes | 0 H2D / 0 D2H |

No host synchronization, host/device transfer, or allocation enters either timed graph-replay
path. Readbacks and telemetry are outside timing.

## Evidence and boundary

Compact machine-readable evidence is in `results/v0.4/`. Reproduction and cache-invalidation
rules are in `docs/REPRODUCING_V0_4.md` and `docs/V0_4_AUTHORITY.md`.

Not implemented: observations, rewards, training-specific action parsing, tensor interop, rollout
buffers, GAE/PPO, Rival policy inference/training, arbitrary body counts, other modes, rendering,
or a generic Bullet API. **v0.5 was not begun.**
