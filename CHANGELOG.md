# Changelog

## Rival 2.0 overnight curriculum — acquisition, base reward, and six-hour continuation (2026-08-26)

- Resumed the exact Campaign 04 checkpoint SHA-256
  `DB5AA09B2CAD40D4C1F5DB1014FDE245C58994A6948458212751724F782BE6B0` at update 120 /
  1,006,632,960 samples with optimizer, RNG, counters, opponent assignments, historical
  policies, and live state intact.
- Continued unchanged Reward V2 until the first two consecutive 4,096-world held-out
  evaluations below the 1% no-touch threshold: updates 390 and 420 at `0.007324` and
  `0.006104`. Acquisition completed at update 420 / 3,523,215,360 samples.
- Added an explicit reward-curriculum checkpoint transition that permits only the authorized
  `RIVAL2_REWARD_V2` -> `RIVAL2_REWARD_V1` contract change. Exact comparisons proved that model,
  optimizer, RNG, counters, assignments, historical policies, and live runtime state did not
  reset or drift.
- Corrected bounded historical-pool eviction lifecycle by deterministically returning live
  assignments to the current policy when their evicted snapshot leaves the 16-policy pool.
  The official clean lineage then passed all historical-version eligibility checks.
- Completed exactly 239 Reward V1 PPO updates / 2,004,877,312 additional Phase B samples. The
  held-out curve was honestly non-monotonic, including no-touch regression to `0.152588` and
  `0.139893` at offsets +180 and +239; no setting was retuned in response.
- Applied the user's prospective extension of Phase C from three to six real elapsed hours.
  Stopped at the first completed update crossing 21,600 seconds: update 5,403 /
  45,323,649,024 cumulative samples at 21,601.926 seconds.
- Published hourly held-out touch rates `66.022572`, `75.795357`, `79.884180`, `75.911922`,
  `83.108105`, and `85.483708` per simulated minute. The final evaluation recorded `2.496403`
  goals/minute and `0.003418` no-touch truncation.
- Passed integrity for all 5,283 continuation updates and exact final-checkpoint reload. Published
  the 48,720,678-byte full resumable checkpoint with SHA-256
  `4DC158DC2A9D16B79FB5FE7D868E3B50928AB113B55DFCC753F3734F8D87372E`.
- Did not run preflight, smoke, parity/regression, pytest, Ruff, compileall, viewer work, or v0.6
  work.

## Rival 2.0 Campaign 04 — Reward V2 continuation to 1B (2026-08-25)

- Verified and loaded the exact Campaign 03 checkpoint SHA-256
  `A0F2E554448B31A373BD73254125AC0ADFDB541EE3B695AD9D040B2CCFA68991`, preserving optimizer,
  Torch/CUDA and policy/opponent RNG state, update/sample counters, opponent assignments, and
  historical policy versions `[0, 2, 3, 6, 12]`.
- Continued the unchanged Reward V2 / entropy-off 131,072-world, horizon-32 training line for
  108 updates. All updates 13–120 passed integrity; the run stopped at exactly 1,006,632,960
  cumulative samples and did not run update 121.
- Saved and evaluated only the authorized update-30/60/90/120 checkpoints. All four 4,096-world
  stochastic self-play evaluations passed with no extra baseline or checkpoint evaluation.
- Recorded the five-point touch curve `1.308672 -> 3.202896 -> 6.453265 -> 8.712013 ->
  16.661451` per simulated minute from 100M through 1B, while no-touch truncation changed
  `0.936279 -> 0.867676 -> 0.869873 -> 0.770752 -> 0.550293`.
- Classified the frozen touch/no-touch 750M-to-1B trend as `CONTINUING`. Reported the
  non-monotonic secondary goal rate honestly: `0.426426` at 750M to `0.311649` at 1B.
- Published the 31,159,541-byte final resumable checkpoint with SHA-256
  `DB5AA09B2CAD40D4C1F5DB1014FDE245C58994A6948458212751724F782BE6B0` and passed exact reload.
- Did not run preflight, reward smoke, regression/parity, post-run test/lint/compile ceremony,
  extra evaluation, viewer work, reward/PPO/model/simulator changes, or v0.6 work.

## Rival 2.0 Campaign 03 — direct Reward V2 training (2026-08-25)

- Preserved `RIVAL2_REWARD_V1` and added `RIVAL2_REWARD_V2` with exactly one per-agent dense
  approach term: true 3D car-ball distance decrease over the final pre-reset four-tick transition,
  divided by 4096. The active Reward V2 contract hash is
  `54CD5AC582133D9BA77CF7DF7976C549B3E659920BA407C9ACCE8A9FD5F50B32`.
- Passed the sole authorized targeted CUDA smoke for positive closing reward, negative opening
  reward, zero unchanged reward, reset-leakage exclusion, finite tensors, and device residence.
- Immediately trained from scratch with the unchanged Campaign 02 entropy-off PPO/model/self-play
  baseline at 131,072 worlds / horizon 32. Stopped at update 12 / 100,663,296 samples, the first
  completed update crossing 100M; all 12 update-integrity records passed.
- Saved resumable checkpoints at the first 25M/50M/100M crossings. Published the final
  21,126,388-byte checkpoint with SHA-256
  `A0F2E554448B31A373BD73254125AC0ADFDB541EE3B695AD9D040B2CCFA68991` and passed exact reload.
- Ran exactly one final 4,096-world stochastic self-play evaluation at seed `920260826`.
  Touches/minute increased `0.291182 -> 1.308672`, goals/minute increased
  `0.040362 -> 0.243800`, and no-touch truncation decreased `0.989746 -> 0.936279` versus
  Campaign 02 final.
- Did not run the old preflight/regression/evaluation ceremony, add a curriculum or another
  reward/PPO/model change, continue beyond the bounded stop, or begin v0.6.

## Rival 2.0 Campaign 02 — controlled entropy-off rerun (2026-08-25)

- Reproduced Campaign 01's seed-`20260826` initialization model SHA-256 and every substantive
  4,096-world initialization-evaluation metric exactly.
- Changed exactly one learning value at the campaign layer: PPO entropy coefficient `0.01 -> 0.0`.
  The diagnostic entropy metric remained logged but contributed zero to every optimization loss.
- Trained at the unchanged 131,072 worlds / horizon 32 for exactly 12 updates and stopped at
  100,663,296 samples. Preserved initialization and first-crossing 10M/25M/50M/100M checkpoints
  and evaluations.
- Passed every finite-state, loss/gradient/optimizer, action-bound, button, done/reset,
  historical-policy, version/sample-age, zero-transfer, checkpoint, and exact-continuation gate.
- Applied the prospective classification unchanged and obtained `IMPROVED`: ordinary self-play
  touches/minute was `0.291182` versus `0.272091` initially and `0.175624` at Campaign 01 final;
  stochastic touch differential was `+35` versus `+15` initially and `-46` at C01 final. Final
  stochastic goal differential `-3` was slightly worse than initialization `-2` but better than
  C01 final `-16`.
- Prevented the diagnosed standard-deviation escalation under the controlled A/B: final
  representative analog standard deviation was approximately `1.015`, maximum KL was `0.008194`,
  maximum clip fraction was `0.087534`, and no diagnostic instability threshold was crossed.
- Published the full 21,126,324-byte final resume checkpoint with SHA-256
  `4A9B366CD3A04222D639252EB2E3EBAD194AF2154D9DBFF213B1AF89A3909FA0`.
- Left v0.1-v0.5 results, Campaign 01 artifacts, the four frozen contracts, and v0.5 training
  implementation byte-for-byte unchanged. No reward/curriculum/second-parameter/v0.6 work began.

## Rival 2.0 Campaign 01 — bounded first training run (2026-08-25)

- Froze campaign seed `20260826`, the normal v0.5 PPO/self-play defaults, and one 4,096-world
  held-out evaluation protocol before the first training update.
- Passed the ordered capacity preflight at the preferred 131,072 worlds with horizon 32: one real
  finite rollout/GAE/PPO update, zero hot-path H2D/D2H state traffic, successful checkpoint and
  inference allocation, and 4,515,647,488 bytes of observed VRAM margin.
- Trained a fresh policy for exactly 12 completed PPO updates and stopped at 100,663,296 agent
  decision samples, the first update crossing 100M. Preserved initialization and first-crossing
  10M/25M/50M/100M resume checkpoints and fixed evaluations.
- Passed every per-update finite-state, bounded/binary-action, selective-done/reset,
  frozen-historical-policy, version/sample-age, parameter/gradient/optimizer, and zero-transfer
  integrity condition. The final checkpoint reload reproduces the next stochastic sample exactly.
- Published execution status `COMPLETE` separately from behavioral result `DEGRADED`. Final
  ordinary self-play touch rate fell from 0.272091 to 0.175624 per simulated minute; the final
  stochastic checkpoint lost 7–23 to initialization with a -46 touch differential, deterministic
  play lost 0–819, and analog policy standard deviations rose near the frozen ceiling.
- Committed the exact 21,126,324-byte full resumable final checkpoint with SHA-256
  `704F2B887BF50E767C86B7080C1E881644480D41A3302D245E833BDE65752B4A`.
- Left the frozen v0.5 `PASS_GREEN` verdict and all `results/v0.1/` through `results/v0.5/` bytes
  unchanged. No reward/model/contract/curriculum setting was altered and no v0.6 work began.

## v0.5 — Rival 2.0 GPU-native training (2026-08-25)

- Added 48 persistent zero-copy Warp/PyTorch CUDA aliases and direct `[world,agent,182]`
  `RIVAL2_OBS_V1` construction with proper orange 180-degree Z rotation, exact canonical boost-pad
  remap, fixed physical normalization, previous action, and accepted lifecycle/mechanic state.
- Froze `RIVAL2_ACTION_V1`: five pre-tanh Gaussian analog controls plus three Bernoulli buttons,
  correct tanh Jacobian log probability, deterministic native controls, configurable `[-5,+1]`
  log-standard-deviation clamp, and exactly four 120-Hz ticks per 30-Hz decision. No lookup table
  or legacy Rival/Wisp action vocabulary participates.
- Added `RIVAL2_REWARD_V1` and `RIVAL2_EPISODE_V1`: exactly zero-sum goal, canonical ball
  progress, unique contact-entry touch, and unique demolition rewards; goal termination;
  15-second no-touch and 45-second hard truncation; final-state truncation bootstrap; and accepted
  selective standard-kickoff reset.
- Added bounded CUDA rollout storage, mixed terminal/truncation GAE, clipped hybrid PPO,
  advantage normalization, GPU shuffling/gathers, entropy/value terms, gradient clipping, and a
  shared 3x512 SiLU actor/critic with 626,190 parameters.
- Added exact checkpoints for weights, optimizer, counters, configs and contract hashes, sampling
  RNG, assignments, and historical metadata/weights. The next stochastic sample reproduces
  exactly and incompatible contract hashes are refused.
- Added current-policy two-sided self-play and reset-only bounded historical-opponent selection.
  Frozen opponents remain GPU resident, receive no gradients, and are bounded to 16 snapshots
  with 20% default eligibility after snapshots exist.
- Passed deterministic observation/action/reward/cadence/rollout gates, independent float64
  hybrid-action and PPO objectives, independent GAE, finite optimizer stress, exact checkpoint
  resume, and the fixed-seed learning sanity gate. The held-out clipped PPO objective improved by
  `5.304016e-4`, or 4.226 standard errors, after one official update.
- Preserved two failed development return metrics as negative evidence and did not change the
  reward contract after observing them. The successful bounded gate is an integration sanity
  result, not a learned-skill or external-transfer claim.
- Swept five practical points with five repeats each. Selected 131,072 worlds at **2,233,901.63
  complete agent samples/s**, **89,505.78 simulated game-seconds/s**, **0.588% CV**,
  14,414,032,896 peak observed VRAM bytes, and zero timed hot-loop H2D/D2H.
- Reran all inherited v0.4 lifecycle/ray/performance gates, v0.3 Phase A/B/C/D, all 39,236 v0.2.2
  cases, all 27 v0.1 scenarios, and repository quality checks. All remain green and published
  v0.1–v0.4 evidence is byte-for-byte unchanged.
- Stopped at the v0.5 boundary. RLBot/CPU RocketSim deployment, Rocket League transfer,
  curricula, legacy Rival compatibility, other modes, and v0.6 work remain unstarted.

## v0.4 — complete standard 1v1 game transition (2026-08-25)

- Added `CompleteWorldSim`, a bounded GPU-resident composition of the accepted v0.3 physics with
  standard-Soccar boost-pad, goal/scoring, kickoff, demolition, respawn, clock, event, and reset
  lifecycle state.
- Integrated all 34 source-backed boost pads into the complete path, including both-car pickup,
  large/small grants, exact float32 cooldown boundaries, reset state, and contention order driven
  by the persistent per-world car visitation lifecycle.
- Ported RocketSim's strict scored-ball boundary and team attribution, the binding's first-entry
  score callback semantics, all five source-valid standard 1v1 kickoff layouts, and source-backed
  goal-callback-to-kickoff reset composition.
- Added demolition disable/frozen-public-state behavior, exact three-second float32 timer,
  tick-360 respawn, all four source-valid respawn poses for both teams, physics re-entry, and
  preservation of car identity and visitation order because membership does not change.
- Added deterministic explicit per-world kickoff and respawn selector state. No ambient RNG,
  pointer/allocator state, case ID, expected output, best-match selection, or hidden table enters
  the runtime.
- Exposed world/episode clocks, score counters, goal/pad/demo/respawn/reset events, and
  policy-neutral terminal/truncation outputs. RocketSim defines no training termination policy,
  so v0.4 keeps `terminated=truncated=0` and leaves policy to the unstarted v0.5 layer.
- Froze native lifecycle authority identity
  `33AA0BA3BC35BC4300E2D2B84A3813CB0AD776479546A50AC3BBC6CE3D3E2562`, binding the pinned
  RocketSim/binding commits, installed extension, all 16 CMFs and combined asset SHA, generator
  source/config/seed, authority settings, and bounded selector/event contract. The frozen cache
  has no live fallback.
- Passed 68/68 pad pickup cases, six goal-boundary cases, five kickoff layouts, eight team/respawn
  poses, exact demolition timing, deterministic mixed lifecycle/reset stress, all inherited v0.3
  A/B/C/D gates, the 39,236-case v0.2.2 gate, the 27-case v0.1 gate, both ray backends, 70/70
  configured tests, Ruff, compile, and diff checks. All prior evidence bytes remain unchanged.
- Measured the complete path at **191,748.10 aggregate simulated game-seconds/s** at 131,072
  worlds with **0.856% CV** and zero timed transfers. The reset-heavy path reached **225,005.06
  sim-s/s**, **3,375,075.88 reset transitions/s**, **0.723% CV**, and zero timed transfers.
- Stopped at the v0.4 boundary. Observations, rewards, training action parsing, tensor interop,
  rollout buffers, PPO, Rival policy training, other modes, arbitrary bodies, and generic Bullet
  work remain unstarted v0.5+ scope.

## v0.3 — dynamic contacts implemented (2026-08-25)

- Added source-ordered Soccar ball/world sphere collision, friction, restitution, spin,
  persistent contacts, analytic planes, and internal-edge adjustment. The complete 31,216-case
  Phase A corpus passes all 124,864 blocking checkpoints.
- Added Octane/ball compound box-sphere GJK/Voronoi/EPA contact, persistent manifold, solver rows,
  and RocketSim-specific hit callback behavior. The 8,192-case Phase B corpus passes all 32,768
  checkpoints with exact callback presence counts.
- Added Octane/Octane compound box-box contact, ordered bump/demo physical classification, queued
  bump impulse, and generic per-world car visitation lifecycle state. Both source-valid native
  visitation branches pass all 8,192 Phase C cases without metric mixing or runtime best-match
  selection.
- Added the fixed two-car/one-ball integrated world: dynamic suspension-ray candidates,
  source-lifecycle broadphase pair order, equal-island manifold ordering, one shared three-body
  constraint solve, split impulse, writeback, and transform integration. All 512 Phase D cases
  pass across eight simultaneous-contact families and both native-valid branches.
- Froze isolated content-addressed native authority for all four phases, including every tick
  1–12, exact source state, immediate post-`SetState` readback, RocketSim/binding revisions,
  extension and CMF hashes, generator source/config/seed, authority settings, and source-valid
  branch custody. Cached gates have no live native fallback.
- Preserved the complete v0.2.2 static gate at 39,236/39,236 and the v0.1 live gate at 27/27;
  both ray backends, 63/63 tests, lint, compile, deterministic stress, and all prior published
  evidence remain green.
- Measured the complete dynamic path at **196,614.39 aggregate simulated game-seconds/s** at
  131,072 worlds, **1.313% CV**, and zero timed host/device transfers, satisfying the 100,000
  sim-s/s v0.3 viability floor with `PASS_GREEN`.
- Stopped at the v0.3 boundary. Demolition removal/respawn, scoring, kickoff/match rules,
  training integration, arbitrary body counts, other modes, and all v0.4 work remain excluded.

### Handoff authority

- Authorized the bounded standard-Soccar dynamic-contact milestone on top of v0.2.2 `PASS_GREEN`.
- Froze v0.2.2 release `6dfd44ad9afeb3d1164da7e0e38c097fb74d07b8` and all prior published evidence as mandatory regressions.
- Sequenced v0.3 as ball-world, car-ball, car-car/bounded bump-demo physical semantics, then integrated static + dynamic multi-contact validation.
- Required source-first translation of the exact pinned RocketSim/Bullet paths before implementation rather than approximate physics followed by behavioral fitting.
- Required a new content-addressed native authority cache per phase, isolated native case semantics, complete tick-1-through-12 cached frames, and no live-RocketSim fallback after freeze.
- Retained ticks 1/4/8/12 as the only blocking local parity horizons; long synchronized open-loop identity remains diagnostic only.
- Required cached operation-level traces and automatic first-divergence comparison for failures; prohibited face/case exceptions, tie epsilons, hysteresis, tolerance broadening, and downstream compensation.
- Kept the complete 39,236-case v0.2.2 static gate and 27/27 v0.1 live-RocketSim scenarios as mandatory regressions.
- Deferred performance work until fidelity is green; set the complete v0.3 dynamic-contact viability floor at 100,000 aggregate simulated game-seconds/s while treating v0.2.2's 511,886.15 sim-s/s only as a comparison baseline.
- Explicitly excluded scoring, kickoff/reset/respawn game rules, RLGym/PPO/training integration, rendering, other game modes, and v0.4+ work.
- Added the controlling package under `handoff/v0.3/` and activated `CODEX_START_PROMPT.md`.

## v0.2.2 — static-world source-parity breadth gate (2026-08-24)

- Froze a deterministic 39,236-case Octane/Soccar authority corpus, hashed against the pinned
  RocketSim/binding revisions, installed extension, CMFs, generator source/config, seed, and
  authority settings. Cached 470,832 native frames with no live fallback in the GPU gate.
- Added complete chassis/wheel states for all 8,020 DFH triangles, all 23,176 shared directed
  edges, and 20 analytic-plane states, with generated coverage separated from actual paired
  target contact.
- Translated the bounded pinned Bullet operation path for box-versus-static-triangle
  GJK/Voronoi/EPA witnesses, persistent-manifold refresh/four-point reduction, internal-edge
  adjustment, contact rows, split impulse, and rigid-body integration.
- Ported RocketSim's exact wheel ray, suspension, friction coefficient/impulse, force/torque,
  and brake-force float32 order. Kept source-correct internal edges and added no hysteresis,
  edge/tie tolerance, face-specific rule, or behavioral fitting.
- Corrected `btGjkPairDetector` internal-valid versus callback-report control flow, preventing
  false EPA fallback for valid shallow witnesses outside the callback distance.
- Passed the cached 1,043-case representative gate and complete 39,236-case gate with **0 hard
  mismatch events, 0 numeric failures, and 0 failed cases** across 156,944 checkpoints.
- Preserved the v0.1 live RocketSim regression at 27/27 and passed 46/46 repository tests.
  Deterministic stress, both GPU query backends, and hot-loop residency are green.
- Measured corrected B3 at **511,886.15 aggregate simulated game-seconds/s** at 262,144 worlds,
  **0.0913% CV**, zero timed transfers, and `PASS_GREEN`.
- Added compact authority, source-trace, parity, regression, benchmark, and manifest evidence;
  retained large oracle/tracing artifacts locally under `.tools/v0.2.2/`.
- Stopped at the v0.2.2 boundary. Dynamic bodies, ball physics, car-car physics, game rules,
  training integration, and all v0.3 work remain unstarted.

## v0.2.1 — static-world fidelity redesign implemented (2026-08-23)

- Replaced v0.2's approximate wheel/contact response with source-backed RocketSim/Bullet
  operation ordering: two-phase wheel preparation/application, exact suspension/friction rows,
  shared solver prestate, ten-iteration velocity and split-impulse PGS, and deferred caps.
- Added Bullet-equivalent Octane box margins/inertia, CMF-local quantized-BVH ordering,
  triangle shared-edge metadata, SAT/GJK closest-feature handling, contact thresholds, manifold
  ordering, tangent/RHS construction, persistence fields, and callback-normal semantics.
- Corrected grounded boost acceleration, air-control suppression, auto-roll state, wheel state,
  and GPU-resident standard Soccar boost-pad pickup/cooldown behavior from pinned source.
- Built a source-only native diagnostic executable against unmodified RocketSimPython commit
  `2da51b1dac7b8127127613a5ff30e490bdd70dd8` and its pinned RocketSim/Bullet sources. It exposes
  pre/post rigid state, wheel rows, manifolds, solver impulses, triangle identities, and GJK
  features without entering the benchmark path.
- Adopted the immediate 2026-08-23 validation-policy adjustment: unchanged meaningful
  tolerances and hard semantic checks now gate authoritative local transitions at 1/4/8/12
  ticks (up to 100 ms); 30–600-tick synchronized open-loop identity is diagnostic only.
- Passed all 140 local checkpoints across the 35 existing scenarios with **0 hard mismatches**
  and **0 numeric failures**. Maximum errors were 0.0009785 uu position, 0.002752 uu/s linear
  velocity, 0.0003908 rad orientation, and 0.00005585 rad/s angular velocity.
- Preserved v0.1 at 27/27 passing and passed 38/38 repository tests. Two 64-world,
  2,400-tick stress runs produced the identical full-state SHA-256, finite/bounded state, and
  zero hot-loop H2D/D2H bytes.
- Measured corrected B3 at **822,480.77 aggregate simulated game-seconds/s** at 262,144 worlds,
  **0.403% CV**, zero timed transfers, and stable scaling. This is 60.89% of v0.2 throughput
  at the same batch and satisfies the v0.2.1 **`PASS_GREEN`** threshold.
- Added a bounded breadth prototype that audits shared-edge topology across all 8,020 DFH
  triangles and reports observed transition coverage without claiming exhaustiveness. The
  existing corpus exercises 2 mesh triangles; per-triangle authoritative generation remains
  a later, separately authorized breadth milestone.
- Preserved all published `results/v0.1/` and `results/v0.2/` bytes and stopped at the v0.2.1
  boundary. Dynamic ball-world, car-ball, and car-car contacts and training integration remain
  unstarted v0.3+ work.

## v0.2.0 — arena + ground-contact proof implemented (2026-08-23)

- Added a strict little-endian RocketSim `.cmf` parser and deterministic Soccar loader with
  structural validation, SHA-256 custody, RocketSim internal hashes, bounds, and combined
  4,468-vertex / 8,020-triangle metadata. Extracted assets remain external and untracked.
- Added one shared normal Warp mesh for chassis AABB queries and a separately measured cuBQL
  mesh over the same geometry for suspension rays. Both passed the 4,608-ray independent CPU
  query corpus; cuBQL delivered the better B1 throughput.
- Added four explicit Octane-compatible wheel states per car and RocketSim-derived suspension,
  throttle, reverse, coast, brake, steering, boost-ground interaction, handbrake, powerslide,
  friction, sticky-force, and extra-pushback behavior.
- Added conservative OBB bounds, Warp mesh triangle candidate enumeration, 13-axis
  triangle-vs-OBB SAT, up to four chassis contacts, and bounded impulse/friction/restitution,
  positional-correction, and angular-response handling.
- Added a deterministic device-resident action tape, contact-rich state generator, and B0/B1/
  B2/B3 CUDA-graph benchmark decomposition with explicit variance, transfer, utilization,
  VRAM, candidate, contact, penetration, and NaN/error accounting.
- Measured a best stable B3 result of 1,350,748.16 aggregate simulated game-seconds/s at
  262,144 worlds, with 0.998% CV and zero timed H2D/D2H traffic. All 44 benchmark points were
  stable below 5% CV.
- Added a measurement-first RocketSim parity protocol, then froze the tolerance table before
  a clean 35-scenario gate run across eight horizons. The gate recorded 85 records with hard
  mismatches and 617 numeric failures; tolerances were not widened to conceal the divergence.
- Added two identical-hash 64-world, 2,400-tick stress runs with finite state and no hot-loop
  transfer.
- Classified v0.2 as `PAUSE_RED`: the standalone performance threshold is green, but required
  RocketSim transfer fidelity fails. Stopped without beginning v0.3.

## v0.2 — arena + ground-contact proof handoff (2026-08-22)

- Authorized the next bounded milestone after v0.1's decisive GPU continuation pass.
- Preserved frozen v0.1 result boundary `1f7a36cc6165273fb658ba07a8458e8d8e60628a` and prohibited rewriting `results/v0.1/` evidence.
- Defined a three-gate v0.2 implementation: stadium mesh/query engine; wheels/suspension/ground driving; chassis-vs-static-world contact.
- Selected one shared immutable DFH/Stadium_P GPU triangle mesh rather than per-world geometry.
- Added explicit collision-asset custody rules: prefer the exact local RocketSim `.cmf` assets when available, otherwise use `RLArenaCollisionDumper` or the RLBot extraction path; extracted game assets remain ignored and are never committed to the public repo.
- Kept NVIDIA Warp as the primary backend and required measured comparison of normal Warp mesh BVH vs the Warp 1.16 cuBQL ray backend for suspension rays where supported.
- Required RocketSim-derived `btVehicleRL` wheel transforms, raycasts, suspension, friction, steering and handbrake behavior rather than a generic bicycle/turn-radius approximation.
- Defined GPU OBB-vs-triangle car-world contact using mesh AABB candidate queries plus a measured narrow-phase/impulse solver.
- Added decomposed B0/B1/B2/B3 benchmarks to isolate the cost of contact-free motion, stadium rays, wheel mechanics and complete static-world contact.
- Added contact-rich parity scenarios across floor, braking/acceleration, steering, powerslide, ramps, walls, ceiling, landings and chassis impacts through horizons up to 600 ticks.
- Added verdict classes: `PASS_GREEN` at >=100k full static-world sim-s/s with parity; `PASS_YELLOW` at >=20k with parity and no architectural dead end; otherwise `PAUSE_RED`.
- Explicitly excluded ball-world, car-ball, car-car, boost pads, game rules and training integration until a separate v0.3+ authorization.

## v0.1.0 — GPU physics proof implemented (2026-08-22)

- Added a flattened, GPU-resident two-car/one-ball world state and fused NVIDIA Warp 120 Hz
  contact-free transition kernel.
- Implemented source-backed gravity, rigid-body integration, caps, airborne throttle/boost,
  jump/sticky/hold/double-jump, dodge/flip, aerial torque/damping and free-ball behavior.
- Added the vectorized NumPy same-equation CPU reference and live `rocketsim==2.2.1`
  `GameMode.THE_VOID` oracle.
- Added 27 deterministic parity scenarios and horizon-specific tolerances selected only after
  a corrected measurement-only run. Same-equation, live RocketSim and axis/sign/state parity
  all passed at 1/4/8/30/60/120 ticks.
- Added automated allocation/reset/control/mechanics/stress/parity/evidence tests.
- Added an adaptive, repeated CPU/GPU benchmark with separate untimed telemetry and explicit
  transfer/variance/NaN accounting. The GPU hot path uses eight-tick CUDA graph blocks.
- Measured a best stable RTX 5090 result of 40,919,361.97 aggregate simulated game-seconds/s
  at 131,072 worlds, versus 11,125.38 sim-s/s for the best same-equation CPU point: 3,678.02x.
- Passed every v0.1 continuation condition. The 203,934.02x ratio to the 200.65 sim-s/s full
  RocketSim/RLGym system reference is recorded only as a non-apples-to-apples comparison.
- Added compact benchmark/parity evidence, resolved dependency/source custody, third-party
  notices and exact reproduction commands.
- Final validation: 20 tests passed; Ruff, `compileall`, JSON parsing and `git diff --check`
  passed. `pip check` retained a documented upstream RocketSim wheel-tag metadata warning;
  the extension imported and all live-oracle tests passed.
- Stopped at the v0.1 boundary; no arena, suspension, ground-contact or other v0.2 work began.

## v0.1 — GPU physics proof handoff

- Established RivalSim as a separate GPU-simulation research repository.
- Defined Soccar 1v1-only scope for the initial architecture.
- Selected NVIDIA Warp for the first GPU proof.
- Defined GPU-resident batched state and contact-free 120 Hz mechanics scope.
- Added RocketSim/RLBot/RLGym physics references and source hierarchy.
- Added CPU/GPU benchmark and RocketSim parity gates.
- Added staged roadmap through arena collision, dynamic contacts and tensor-native training integration.
- Added Codex implementation prompt.

This section records the pre-implementation handoff. The implemented result is preserved above
as v0.1.0 rather than rewriting the historical boundary.
