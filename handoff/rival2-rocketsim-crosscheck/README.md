# Active Handoff — Rival 2.0 RocketSim Reciprocal Cross-Validation

## Purpose

This milestone is an **inverse-environment validation** of RivalSim, not a training milestone.

Rival and public Nexto have already been run inside RivalSim. The published RivalSim full-match benchmark at commit `15d6119f2fc860a81c64c81f9eec722b6b99f1ad` showed final-45B Rival overwhelmingly beating pinned public Nexto, but most Rival scoring came directly from kickoff.

The user now wants the reciprocal experiment:

> Keep Nexto in the RocketSim/RLGym-style environment it was built and trained around, adapt only frozen Rival into that environment, and compare the matchup behavior/results against the already-published RivalSim benchmark.

The goal is to answer two separate questions:

1. **Simulator cross-validation:** Does the same frozen Rival-vs-Nexto matchup produce broadly similar performance in RocketSim and RivalSim?
2. **Open-play skill:** When kickoff is removed inside RocketSim, how does frozen Rival perform against frozen Nexto in ordinary 1v1 play?

This milestone supersedes the previously active RivalSim-only kickoff-free open-play handoff for now. Do **not** execute `handoff/rival2-nexto-open-play/README.md` during this milestone unless a later explicit authorization reactivates it.

No learning is authorized.

---

## Frozen identities

### Rival

Use the final Rival 2.0 checkpoint exactly:

`checkpoints/rival2/overnight/rival2_overnight_final_6h_resume.pt`

Expected SHA-256:

`4DC158DC2A9D16B79FB5FE7D868E3B50928AB113B55DFCC753F3734F8D87372E`

Expected policy version / cumulative samples:

`5403 / 45,323,649,024`

Rival remains:

- `RIVAL2_OBS_V1`;
- native hybrid 8-controller action;
- deterministic deployment action for canonical testing;
- ordinary stochastic hybrid sampling only where the robustness protocol explicitly requests it;
- policy cadence: 30 Hz / one action every 4 physics ticks.

Do not change or retrain the policy.

### Nexto

Use the exact already-pinned public Nexto:

`Rolv-Arild/Necto@2e6ed7d6ed2b352e8ff529d4a12a0c9c70c28cca`

Expected public model SHA-256:

`BF5343B5EEACAC6BF7CDB75DAC4A5C14BA0F94D820EAE75F00A211B6119D69FA`

Use the upstream/source Nexto semantics in the RocketSim harness rather than depending on the RivalSim GPU port for the official reciprocal result:

- upstream `nexto_obs.py` observation semantics;
- upstream TorchScript model;
- exact 90-action lookup table;
- deterministic `beta=1` policy selection;
- 15 Hz / 8-tick neural cadence;
- exact stock hard-coded kickoff controls at 120 Hz.

The already-published GPU-port fidelity evidence may be used as supporting evidence, but the official RocketSim result should exercise the source/reference Nexto path as directly as practical.

Preserve upstream provenance and CC BY-NC-SA 4.0 attribution.

### RocketSim reference physics

Use the exact RocketSim physics lineage RivalSim was ported against:

`ZealanL/RocketSim@c2baacb8f4b441dd8505e63c2aeb5a1679b60b02`

Use the accepted Python/binding path already used by the RivalSim project if available (historically `rocketsim==2.2.1`), but record the exact runtime package/version/source identity in the evidence.

Do not silently substitute another RocketSim physics revision. If the available Python binding cannot be demonstrated to correspond to the accepted reference physics, build/use an appropriate binding from the pinned source or stop with a clear identity failure rather than comparing against an unknown physics build.

Physics rate: 120 Hz.

---

## Phase 1 — Build a RocketSim -> Rival adapter

Do **not** port RivalSim into RocketSim. Build only the policy adapter needed to run frozen Rival from RocketSim/RLGym state.

Required flow:

`RocketSim state -> RIVAL2_OBS_V1 adapter -> frozen Rival policy -> native 8 controls -> RocketSim`

The adapter must reconstruct Rival's existing observation contract without redefining it.

At minimum, faithfully derive or maintain:

- ball position, linear velocity, angular velocity;
- self/opponent car position, velocity, quaternion/orientation basis, angular velocity;
- boost amount;
- grounded/jump/double-jump/flip state and derived jump/dodge availability;
- demo state/timer semantics available from the RocketSim/RLGym state;
- wheel/on-ground information required by the frozen observation contract;
- all 34 boost-pad availability/cooldown fields with the exact Blue/Orange pad ordering/remap used by `RIVAL2_OBS_V1`;
- relative ball/self/opponent features;
- exact 180-degree Orange canonicalization;
- previous Rival action state;
- kickoff indicator;
- episode-age and no-touch-age fields maintained by the adapter/runtime with the same scaling expected by the frozen policy.

Do not change `RIVAL2_OBS_V1` to make RocketSim integration easier.

If a RocketSim API does not expose an observation field literally, derive it from authoritative RocketSim state/lifecycle data when the semantics are equivalent. Document any field whose semantics cannot be represented exactly and quantify the difference rather than hiding it.

### Adapter parity gate

Before any official matchup, prove that the RocketSim adapter reproduces Rival's accepted observation semantics.

Construct at least **2,048 broad physically valid reference states** covering:

- both teams;
- all five kickoff layouts;
- ground and airborne cars;
- jump/flip availability combinations;
- demos/respawn states if representable;
- broad boost values and pad cooldown patterns;
- broad ball positions/heights/velocities;
- wall/corner/backboard-adjacent states.

For each state, compare the RocketSim-built `RIVAL2_OBS_V1` against the accepted RivalSim observation builder for the same physical/lifecycle state.

Publish:

- max absolute error overall;
- max error by observation block;
- count of exact/non-exact fields;
- deterministic Rival action agreement;
- any known semantic mismatch.

Target: observation differences should be limited to unavoidable numeric representation noise. Deterministic action agreement should be 100% across the parity set. If deterministic actions disagree, do not proceed to the official benchmark until the adapter is corrected or the mismatch is explicitly shown to be an unavoidable environment-semantic difference and reported for review.

---

## Phase 2 — Reciprocal full-match benchmark in RocketSim

Reproduce the already-published RivalSim matchup protocol as closely as the reference RocketSim environment permits.

### Match semantics

- 120 Hz RocketSim physics;
- 5:00 active regulation = exactly 36,000 physics ticks;
- goals preserve score and reset to kickoff;
- tied regulation enters fresh-kickoff, next-goal-wins overtime;
- preserve the same authorized simplification as the RivalSim benchmark: no Rocket League zero-second airborne continuation;
- Rival native 30 Hz cadence;
- Nexto native 15 Hz neural cadence;
- Nexto exact stock hard-coded kickoff controller at 120 Hz;
- standard Soccar boost, demos and respawns;
- no Rival training-specific 15-second no-touch or 45-second episode truncation;
- no reward affects match outcome.

### Canonical deterministic suite

Run exactly the same canonical matrix as the RivalSim benchmark:

- all five starting kickoff layouts;
- Rival as Blue and Rival as Orange;
- 10 deterministic full matches total.

Use frozen deterministic Rival and deterministic Nexto.

Publish every exact scoreline.

### Stochastic Rival robustness suite

Run **4,096 full matches** if practical under RocketSim throughput:

- 2,048 Rival as Blue;
- 2,048 Rival as Orange;
- Rival sampled from its ordinary stochastic hybrid action distribution using a fixed published seed;
- Nexto remains deterministic;
- kickoff layouts distributed evenly/deterministically.

If 4,096 is materially impractical in the reference CPU RocketSim runtime, do not silently shrink the suite. Publish measured throughput from a small targeted probe and choose the largest power-of-two count that is practical while preserving exact 50/50 Rival side balance and even kickoff distribution. The reason and final count must be explicit.

### Required full-match metrics

At minimum, publish separately for Rival as Blue and Rival as Orange:

- wins/losses;
- regulation/OT wins;
- win rate;
- goals for/against;
- goals per match;
- mean/median goal differential;
- exact deterministic scorelines;
- touch count/share if available faithfully;
- kickoff first-touch count;
- direct kickoff goals using the same definition as the RivalSim benchmark (goal before more than one accepted touch after kickoff);
- possession next-touch retention/opponent handoff if available;
- demos;
- goal-entry placement if practical.

Do not hide Blue/Orange results inside only an aggregate.

---

## Phase 3 — Cross-simulator comparison

Compare the RocketSim result directly against the already-published RivalSim evidence at:

- `docs/RIVAL2_NEXTO_RESULTS.md`;
- `results/rival2/nexto/summary.json`;
- `results/rival2/nexto/canonical_deterministic.json`;
- `results/rival2/nexto/stochastic_robustness.json`.

The purpose is not to require bit-identical trajectories. Small physics differences and chaotic divergence make that unrealistic.

Instead, compare **behavioral/performance distributions**.

Publish a cross-simulator table containing at least:

- Rival win rate by side;
- Rival goals/match by side;
- Nexto goals/match by side;
- mean goal differential by side;
- kickoff first-touch rate;
- kickoff-goal rate;
- touch share if available;
- physical Blue-vs-Orange scoring totals;
- deterministic 10-match result direction and score-range comparison.

For every comparable metric report:

- RivalSim value;
- RocketSim value;
- absolute delta;
- relative delta where meaningful.

### Cross-validation interpretation

Do not manufacture a single pass/fail tolerance after seeing the results.

Classify the evidence descriptively:

- **STRONG_AGREEMENT** — same qualitative matchup dominance and similar major rates/distributions;
- **PARTIAL_AGREEMENT** — same winner/qualitative ordering but material rate differences;
- **DISAGREEMENT** — matchup ordering or core behavior changes materially between simulators.

Explain which metrics drive the classification.

Do not tune RivalSim or the policies during this milestone.

---

## Phase 4 — Kickoff-free open-play benchmark in RocketSim

After the reciprocal full-match benchmark is complete, use RocketSim to answer the separate open-play question.

This phase is **inside RocketSim only**. It does not require the deferred RivalSim-only open-play handoff.

### Open-play state bank

Harvest exactly **4,096 physically continuous RocketSim open-play states**:

- 2,048 from frozen Rival stochastic self-play in RocketSim;
- 2,048 from deterministic pinned-Nexto self-play in RocketSim.

Eligibility:

- at least 5.0 active simulated seconds since kickoff/reset;
- at least one accepted ball touch since kickoff;
- no goal/reset pending;
- ball not beyond scoring plane;
- both cars active/not demolished at capture;
- ordinary continuous play, not kickoff control.

Preserve all RocketSim state necessary for faithful continuation, including car/ball state, boost/pad state, and relevant lifecycle state.

At restored starts initialize policy previous-action memory neutrally to zero for both policies while preserving real physical/lifecycle state.

### Four-way paired replay

For each of 4,096 base states run four deterministic duels:

1. original state — Rival Blue / Nexto Orange;
2. original state — Nexto Blue / Rival Orange;
3. exact 180-degree/team-swapped mirror — Rival Blue / Nexto Orange;
4. exact mirror — Nexto Blue / Rival Orange.

Total: **16,384 kickoff-free open-play duels**.

Each duel:

- starts directly from restored open play;
- has no kickoff at start;
- first goal wins;
- no goal reset/kickoff after scoring;
- maximum 60 simulated seconds;
- unresolved at 60 seconds = draw;
- standard RocketSim physics/boost/demos/respawns continue.

### Required open-play output

Publish overall and stratified:

- Rival wins;
- Nexto wins;
- draws;
- decisive-duel Rival win rate;
- all-duel Rival win fraction;
- time-to-goal by winner;
- Rival-as-Blue vs Rival-as-Orange;
- source state distribution;
- original vs mirrored;
- closest-to-ball at start;
- initial ball third/height;
- paired-family result: Rival wins 4/4, 3/4, 2/4, 1/4, 0/4, plus draw-incomplete families.

Reuse touch/possession/trajectory telemetry where practical, but do not let telemetry complexity block the core outcome benchmark.

This phase answers whether Rival's apparent Nexto dominance exists without kickoff inside the reference RocketSim environment.

---

## Integrity checks

This is a targeted validation milestone, not general release ceremony.

Required checks only:

- frozen Rival checkpoint identity exact;
- pinned Nexto source/model identity exact;
- pinned RocketSim reference identity exact;
- Rival adapter parity gate;
- deterministic Rival action agreement on parity states;
- Nexto source path identity/semantics preserved;
- exact match counts/side assignment;
- full-match and open-play termination semantics correct;
- no policy/reward/PPO/simulator tuning during the run;
- no telemetry buffer overflow if telemetry is used;
- state capture/restore integrity for the RocketSim open-play bank;
- mirror involution check for open-play state transforms.

Do not run unrelated Ruff/pytest/compileall/regression/parity suites unless a changed implementation path actually requires a small targeted check to establish correctness.

---

## Published evidence

Publish at minimum:

- human-readable report: `docs/RIVAL2_ROCKETSIM_CROSSCHECK.md`;
- machine summary: `results/rival2/rocketsim_crosscheck/summary.json`;
- Rival adapter parity evidence;
- RocketSim runtime/provenance identity;
- canonical full-match ledger;
- stochastic full-match summary/ledger;
- cross-simulator metric comparison;
- RocketSim open-play state-bank description;
- RocketSim open-play per-duel ledger;
- RocketSim open-play paired-family summary.

Record implementation entrypoints and exact artifact hashes where appropriate.

---

## Explicitly deferred work

Do **not** train Rival against Nexto during this milestone.

Do **not** begin fake-kickoff curriculum training yet. The user's planned future fake-kickoff curriculum remains recorded: opponents may intentionally backflip/retreat to boost, concede first contact, and receive Rival's kickoff hit. That becomes useful after the simulator/open-play validation is understood.

Do not change Rival's reward, PPO, architecture, observation/action contracts, or physics.

Do not build the viewer or begin v0.6.

---

## Stop boundary

When:

1. the RocketSim Rival adapter passes its targeted parity gate;
2. the reciprocal RocketSim full-match benchmark is complete;
3. the cross-simulator RivalSim-vs-RocketSim comparison is published; and
4. the RocketSim kickoff-free open-play benchmark is complete,

commit and push all implementation/evidence to `origin/main` and stop for review.
