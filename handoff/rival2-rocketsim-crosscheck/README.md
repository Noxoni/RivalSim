# Active Handoff v2 — Rival 2.0 RocketSim Reciprocal Cross-Validation

## Authority and version boundary

This is the controlling `v2` handoff for the next RivalSim objective.

The prior RocketSim handoff remains recoverable in Git history. This version updates it after completion of the RivalSim kickoff-free benchmark at:

`9807da8b3c404beb63a5426959132de549332128`

Do **not** repeat the completed RivalSim open-play benchmark. Its result is now reference evidence.

The immediate implementation objective is to build and validate the adapter required to run frozen final-45B Rival inside pinned RocketSim. Once the adapter passes the gates below, continue through both the **normal full-match benchmark with kickoffs** and the **kickoff-free open-play benchmark**, then publish the cross-simulator comparison.

No learning is authorized.

---

## 1. Purpose

Keep public Nexto in the RocketSim/RLGym-style environment and semantics it was built around. Adapt only frozen Rival into that environment:

`RocketSim state -> RIVAL2_OBS_V1 adapter -> frozen Rival policy -> native 8 controls -> RocketSim`

The milestone must answer three questions:

1. **Adapter fidelity:** Can RocketSim state reproduce frozen Rival's accepted `RIVAL2_OBS_V1` semantics and deterministic action behavior without redefining the policy contract?
2. **Normal-match transfer:** In ordinary 5-minute RocketSim 1v1 matches with real kickoffs and goal resets, how does frozen Rival perform against native/source public Nexto?
3. **Side-asymmetry localization:** Does the large Rival Blue/Orange split observed in RivalSim persist in RocketSim, both in normal matches and after kickoff is removed?

### Completed RivalSim reference evidence

Full-match reference:

- `docs/RIVAL2_NEXTO_RESULTS.md`;
- `results/rival2/nexto/`;
- completion commit `15d6119f2fc860a81c64c81f9eec722b6b99f1ad`.

Kickoff-free reference:

- `docs/RIVAL2_NEXTO_OPEN_PLAY_RESULTS.md`;
- `results/rival2/nexto_open_play/`;
- completion commit `9807da8b3c404beb63a5426959132de549332128`.

Completed RivalSim kickoff-free result:

- overall decisive Rival win rate: `54.772%`;
- Rival as Blue: `46.948%`;
- Rival as Orange: `62.545%`;
- Orange-minus-Blue difference: approximately `15.597` percentage points;
- original vs exact mirrored states: `55.158% / 54.384%`;
- Rival inheriting original Blue vs original Orange physical car: `54.678% / 54.866%`.

Blue/Orange must remain an explicit diagnostic dimension throughout this milestone.

---

## 2. Frozen identities

### Rival

Use exactly:

`checkpoints/rival2/overnight/rival2_overnight_final_6h_resume.pt`

Expected SHA-256:

`4DC158DC2A9D16B79FB5FE7D868E3B50928AB113B55DFCC753F3734F8D87372E`

Expected policy version / cumulative samples:

`5403 / 45,323,649,024`

Preserve:

- `RIVAL2_OBS_V1` unchanged;
- `RIVAL2_ACTION_V1` / native hybrid 8-controller output unchanged;
- deterministic deployment action for canonical tests;
- stochastic hybrid sampling only where the robustness protocol explicitly requests it;
- 30 Hz policy cadence / one action every four 120-Hz physics ticks.

Rival's normal policy handles its own kickoff play. Do not add or substitute a scripted Rival kickoff controller.

### Nexto

Use the exact pinned public Nexto:

`Rolv-Arild/Necto@2e6ed7d6ed2b352e8ff529d4a12a0c9c70c28cca`

Expected model SHA-256:

`BF5343B5EEACAC6BF7CDB75DAC4A5C14BA0F94D820EAE75F00A211B6119D69FA`

For the official RocketSim result, exercise source/reference Nexto semantics as directly as practical:

- upstream `nexto_obs.py` semantics;
- upstream TorchScript model;
- exact 90-action lookup table;
- deterministic `beta=1` selection;
- 15 Hz / one neural decision every eight physics ticks;
- exact stock hard-coded kickoff controls at 120 Hz.

The RivalSim GPU Nexto port is prior/supporting fidelity evidence only, not the official RocketSim Nexto implementation.

Preserve upstream provenance and CC BY-NC-SA 4.0 attribution.

### RocketSim

Reference physics lineage:

`ZealanL/RocketSim@c2baacb8f4b441dd8505e63c2aeb5a1679b60b02`

Use the accepted Python/binding path corresponding to that lineage if demonstrable. Historically `rocketsim==2.2.1` was used by the project, but record the exact package/source identity actually executed.

Do not silently substitute an unknown physics revision. If the installed binding cannot be tied to the accepted reference lineage, resolve the identity from the pinned source or stop with a clear identity failure.

Physics rate: 120 Hz.

---

## 3. Phase 1 — Build the RocketSim -> Rival adapter

Do **not** port RivalSim into RocketSim and do not change `RIVAL2_OBS_V1` to make integration easier.

Build only the state/lifecycle adapter required to reconstruct Rival's existing observation contract and maintain its runtime memory fields.

At minimum, faithfully derive or maintain:

- ball position, linear velocity and angular velocity;
- self/opponent car position, velocity, quaternion/orientation basis and angular velocity;
- boost amount;
- grounded/wheel state;
- jump, double-jump, dodge/flip availability and lifecycle semantics;
- demo state and timer/respawn semantics available from RocketSim;
- all 34 boost-pad availability/cooldown fields with exact `RIVAL2_OBS_V1` Blue/Orange ordering/remap;
- relative ball/self/opponent features;
- exact Orange 180-degree canonicalization;
- Rival previous-action state;
- kickoff indicator;
- episode age and no-touch age with the scaling expected by the frozen policy.

Where RocketSim does not expose a field literally, derive it only from authoritative RocketSim state/lifecycle data when semantics are equivalent. Name and quantify any unavoidable semantic mismatch.

### 3.1 Adapter parity corpus

Before official gameplay, construct at least **2,048 broad physically valid reference states** spanning both teams, all five kickoff layouts, ground/air states, jump/flip combinations, demos/respawns if representable, broad boost/pad states, broad ball states, and wall/corner/backboard-adjacent situations.

Compare RocketSim-built `RIVAL2_OBS_V1` against the accepted RivalSim observation builder for equivalent physical/lifecycle states.

Publish:

- max absolute error overall and by observation block;
- exact/non-exact field counts;
- deterministic Rival action agreement;
- all metrics separately for Rival-as-Blue and Rival-as-Orange;
- every known semantic mismatch.

Target: only unavoidable numeric representation noise and **100% deterministic action agreement**.

### 3.2 Mandatory team/mirror symmetry diagnostic

Create broad exact 180-degree/team-swapped physical-state pairs. For every pair, build both canonical observations and compare every block expected to be team-symmetric, then run frozen deterministic Rival on both.

Publish observation error by block and deterministic native-action agreement. Expected action agreement is **100%** for canonical-equivalent pairs.

If this fails, do not bury it in gameplay averages. Correct an adapter implementation error if one exists; otherwise stop before official benchmarking with the unresolved semantic difference documented.

---

## 4. Phase 2 — Normal 5-minute RocketSim matches with kickoffs

This is the **primary gameplay benchmark**. These are ordinary matches, not kickoff-free duels.

### Match lifecycle

- standard 1v1 Soccar;
- 120 Hz RocketSim physics;
- exactly 5:00 active regulation / 36,000 physics ticks;
- every match begins from a standard kickoff;
- every goal updates score and resets both players/ball to a standard kickoff;
- kickoff layouts follow the normal five-layout set and are controlled/balanced for comparison;
- tied regulation enters a fresh standard kickoff and next-goal-wins overtime;
- Rival uses its frozen policy during kickoff and open play;
- Nexto uses its exact stock source kickoff controller during kickoff and its neural policy during open play;
- standard boost pads, demolition and respawn behavior continue throughout;
- no Rival training no-touch/hard episode truncations;
- no reward affects match outcome.

For apples-to-apples comparison with the already-published RivalSim full-match benchmark, retain its previously authorized omission of Rocket League's zero-second airborne continuation rule. Do not otherwise simplify kickoff or match lifecycle.

### 4.1 Canonical deterministic suite

Run exactly all five starting kickoff layouts with Rival as Blue and Rival as Orange: **10 deterministic full matches**.

Publish every exact scoreline.

### 4.2 Stochastic Rival robustness suite

Target **4,096 complete 5-minute matches**:

- 2,048 Rival Blue;
- 2,048 Rival Orange;
- fixed published Rival stochastic seed;
- Nexto deterministic;
- kickoff layouts evenly/deterministically distributed.

If reference CPU RocketSim throughput makes 4,096 materially impractical, first publish a targeted throughput probe and then choose the largest practical power-of-two count while preserving exact 50/50 side balance and even kickoff distribution. Never silently shrink the suite.

### 4.3 Required normal-match output

Report separately for Rival Blue and Rival Orange:

- wins/losses;
- regulation/OT wins;
- win rate;
- goals for/against and goals per match;
- mean/median goal differential;
- exact canonical deterministic scorelines;
- total kickoffs;
- kickoff first-touch count/rate;
- direct kickoff goals under the same definition used by the RivalSim benchmark;
- touch count/share where faithful;
- same-next-touch retention/opponent handoff where faithful;
- demos;
- goal-entry placement where practical.

Also report physical Blue-team and Orange-team scoring/win totals regardless of policy assignment.

---

## 5. Phase 3 — Cross-simulator normal-match comparison

Compare RocketSim directly against the completed RivalSim full-match evidence:

- `docs/RIVAL2_NEXTO_RESULTS.md`;
- `results/rival2/nexto/summary.json`;
- `results/rival2/nexto/canonical_deterministic.json`;
- `results/rival2/nexto/stochastic_robustness.json`.

Do not require bit-identical trajectories. Compare distributions and qualitative ordering.

For every comparable metric publish RivalSim value, RocketSim value, absolute delta, and relative delta where meaningful.

At minimum compare by side:

- Rival win rate;
- Rival/Nexto goals per match;
- mean goal differential;
- kickoff first-touch rate;
- direct kickoff-goal rate;
- touch share if available;
- physical Blue-vs-Orange scoring totals;
- deterministic result direction and score range.

Classify the normal-match evidence as `STRONG_AGREEMENT`, `PARTIAL_AGREEMENT`, or `DISAGREEMENT` and explain which metrics drive it. Do not tune either policy or simulator after seeing results.

---

## 6. Phase 4 — Kickoff-free open-play benchmark in RocketSim

After the normal-match comparison is complete, reproduce the controlled open-play question inside RocketSim.

Harvest exactly **4,096 physically continuous RocketSim open-play states**:

- 2,048 from frozen Rival stochastic self-play;
- 2,048 from deterministic native/source Nexto self-play.

Capture only ordinary continuous play at least 600 physics ticks after kickoff/reset, after at least one accepted touch, with both cars active, no pending goal/reset, and the ball inside the scoring plane. Preserve all RocketSim physical/lifecycle state necessary for faithful continuation.

Initialize only policy previous-action memory neutrally at restored start; do not erase real physical timers/state.

For each base state run four deterministic duels:

1. original — Rival Blue / Nexto Orange;
2. original — Nexto Blue / Rival Orange;
3. exact 180-degree/team-swapped mirror — Rival Blue / Nexto Orange;
4. exact mirror — Nexto Blue / Rival Orange.

Total: **16,384 duels**.

Each duel begins directly in open play, has no kickoff at start, ends on first goal, has no goal reset, and draws at exactly 60 simulated seconds if unresolved.

Publish overall and side-separated wins/losses/draws, decisive win rate, all-duel win fraction, time-to-goal, source distribution, original/mirror split, inherited physical role, closest-to-ball, field third/height, boost advantage, and four-duel family outcomes.

Where practical, reuse touch/possession/trajectory telemetry from the RivalSim protocol.

---

## 7. Phase 5 — Side-asymmetry localization

Compare the RocketSim Blue/Orange split against both completed RivalSim protocols.

Explicitly answer:

- Does normal RocketSim play show the same Orange advantage seen in RivalSim full matches?
- Does RocketSim kickoff-free play reproduce the RivalSim `46.948%` Blue vs `62.545%` Orange split?
- Are original/mirror and inherited-physical-role controls close while team assignment remains separated?
- Did the adapter parity/symmetry corpus show any team-dependent observation or action discrepancy?

Interpretation guide:

- asymmetry present in both simulators with clean adapter symmetry -> policy/contract/game-side semantics become more likely than RivalSim-specific physics;
- asymmetry large in RivalSim but absent/reduced in RocketSim with clean adapter symmetry -> RivalSim simulator/lifecycle asymmetry becomes more likely;
- asymmetry appears only after adapter introduction or parity differs by team -> adapter/canonicalization defect becomes the first suspect.

Do not claim causality beyond what the evidence supports.

---

## 8. Integrity checks

Required targeted checks:

- frozen Rival checkpoint identity exact;
- pinned Nexto source/model identity exact;
- pinned RocketSim reference identity exact;
- adapter parity corpus complete;
- deterministic Rival action agreement reported overall and by side;
- team/mirror symmetry diagnostic complete;
- Nexto source semantics preserved;
- normal full-match counts, side balance, kickoff lifecycle and score/reset behavior exact;
- RocketSim open-play state capture/restore integrity;
- open-play mirror involution;
- exact duel assignment counts and termination semantics;
- no training/tuning during evaluation;
- no telemetry overflow if telemetry is used.

Avoid unrelated release/lint/regression ceremony unless a changed implementation path requires a small targeted correctness check.

---

## 9. Published evidence

Publish at minimum:

- `docs/RIVAL2_ROCKETSIM_CROSSCHECK.md`;
- `results/rival2/rocketsim_crosscheck/summary.json`;
- RocketSim runtime/provenance identity;
- Rival adapter parity + team/mirror symmetry evidence;
- canonical normal-match ledger;
- stochastic normal-match summary/ledger;
- normal-match RivalSim-vs-RocketSim comparison;
- RocketSim kickoff-free state-bank description;
- RocketSim kickoff-free per-duel ledger;
- paired-family summary;
- side-asymmetry localization summary;
- relevant implementation entrypoints and artifact hashes.

---

## 10. Explicitly deferred work

Do **not** train Rival against Nexto during this milestone.

Do **not** begin fake-kickoff curriculum training yet. Retreat/backflip-to-boost opponents that intentionally concede first contact remain future curriculum work.

Do not change Rival's reward, PPO, architecture, observation/action contracts, controller semantics, or either simulator's physics.

Do not build the viewer or begin v0.6.

---

## Stop boundary

When all of the following are complete:

1. RocketSim -> Rival adapter passes targeted parity and team/mirror symmetry gates;
2. normal 5-minute RocketSim Rival-vs-Nexto matches with kickoffs are complete;
3. normal-match RivalSim-vs-RocketSim comparison is published;
4. RocketSim kickoff-free open-play benchmark is complete;
5. Blue/Orange asymmetry localization is published;

commit and push all implementation/evidence to `origin/main` and stop for review.
