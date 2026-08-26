# Active Handoff — Rival 2.0 vs Nexto Kickoff-Free Open-Play Benchmark

## Authority and purpose

Start from the completed Nexto-port/full-match benchmark lineage at commit:

`15d6119f2fc860a81c64c81f9eec722b6b99f1ad`

Frozen Rival checkpoint:

`checkpoints/rival2/overnight/rival2_overnight_final_6h_resume.pt`

Expected SHA-256:

`4DC158DC2A9D16B79FB5FE7D868E3B50928AB113B55DFCC753F3734F8D87372E`

Pinned Nexto remains exactly the already-validated public port:

`Rolv-Arild/Necto@2e6ed7d6ed2b352e8ff529d4a12a0c9c70c28cca`

with model SHA-256:

`BF5343B5EEACAC6BF7CDB75DAC4A5C14BA0F94D820EAE75F00A211B6119D69FA`

The prior full-match result was dominated by direct kickoff goals. This handoff exists to answer one question only:

> When kickoff is removed from the experiment, which frozen policy is stronger in ordinary 1v1 open play: final-45B Rival or pinned public Nexto?

This is evaluation only. Do not train either policy and do not change any policy, reward, PPO setting, observation/action contract, controller semantics, or physics.

## 1. Preserve the accepted policy ports

Reuse the already-published GPU-native Nexto adapter and Rival runtime. Do not reimplement or retune them.

- Rival: deterministic deployment action, native 30 Hz cadence / 4 physics ticks.
- Nexto: deterministic `beta=1` action, native 15 Hz cadence / 8 physics ticks.
- Physics: RivalSim at 120 Hz.
- Nexto stock hard-coded kickoff controller is irrelevant in the actual open-play duels because no duel begins from a kickoff and no kickoff reset is allowed.

If code changes are required only to support state capture/restore or this evaluator, keep them policy-neutral and evaluation-only.

## 2. Build a realistic open-play state bank

Do not generate arbitrary independent random car/ball teleports as the primary benchmark distribution.

Harvest exactly **4,096 base states** from physically continuous RivalSim trajectories:

- **2,048 states from final-45B Rival stochastic self-play**;
- **2,048 states from deterministic pinned-Nexto self-play**.

The source policy is used only to produce physically plausible open-play states. Source-policy identity must be retained in the state-bank evidence so results can be reported separately by source distribution.

### Eligibility for capture

A state may be captured only when all of the following are true:

- at least **5.0 active simulated seconds / 600 physics ticks** have elapsed since the most recent kickoff/reset;
- at least one accepted car-ball touch has occurred since that kickoff/reset;
- no goal/reset is pending and the ball is not already beyond the scoring plane;
- both cars are active/not demolished at the capture instant;
- the world is in ordinary continuous play, not kickoff control;
- the full simulator state needed for faithful continuation is captured, including car/ball rigid state, boost amounts and pad cooldowns, jump/flip/supersonic/demo/lifecycle state needed by physics, and any other state required to continue the accepted simulator exactly.

Sample states deterministically from seeded source rollouts without selecting them based on which policy appears advantaged. Publish the capture seed/rule and the state-bank distribution: ball position/height/speed, car-ball distances, car boosts, field thirds, and source policy.

### Neutral policy-memory initialization

The benchmark must not give either policy another policy's previous controller action.

At each restored open-play start:

- simulator/physics state is restored from the harvested state;
- Rival previous-action observation state is initialized to all zeros;
- Nexto previous-action input is initialized to all zeros;
- Nexto's required boost/demo timer semantics must remain consistent with the restored simulator state; do not zero physical/lifecycle timers merely for convenience.

Record this one-decision neutral-memory boundary explicitly in the evidence.

## 3. Four-way paired replay for every base state

Each of the 4,096 base states must produce **four deterministic open-play duels**, for **16,384 total duels**.

For every base state S:

1. original S — Rival controls Blue, Nexto controls Orange;
2. original S — Nexto controls Blue, Rival controls Orange;
3. exact 180-degree field-plane/team-swapped mirror of S — Rival controls Blue, Nexto controls Orange;
4. exact mirror of S — Nexto controls Blue, Rival controls Orange.

The mirror must transform all side-dependent physical state consistently, including cars, ball, orientations/velocities, boost-pad indexing/cooldowns, team-specific lifecycle state, and any simulator state whose meaning changes under the 180-degree team swap.

This four-way pairing is mandatory. It balances:

- which policy inherits each physical car/role;
- initial possession/field advantage;
- Blue versus Orange;
- the already-observed physical/team scoring asymmetry.

Do not collapse away the paired identity in the raw evidence.

## 4. Duel semantics — no kickoff contamination

Each duel begins directly from its restored open-play state.

- no kickoff countdown or hard-coded kickoff controller at duel start;
- first goal wins the duel;
- **no goal reset and no subsequent kickoff**;
- maximum active-play duration: **60.0 simulated seconds / 7,200 physics ticks**;
- if no goal occurs by the limit, classify the duel as a draw/unresolved open-play duel;
- normal boost pickup/cooldown, ball/world contacts, car-car contacts, demos and 3-second demo respawns continue normally;
- no Rival training-specific 15-second no-touch or 45-second truncation rule;
- no reward is needed to decide the duel.

There must be no way for a kickoff event to enter the outcome after the duel starts.

## 5. Required headline results

Publish at minimum:

### Overall

- Rival wins;
- Nexto wins;
- draws;
- decisive-duel Rival win rate;
- all-duel Rival win fraction;
- mean and median time-to-goal by winner.

### Mandatory stratification

Report the same outcome metrics separately by:

- Rival as Blue vs Rival as Orange;
- original vs mirrored states;
- Rival-self-play-source states vs Nexto-self-play-source states;
- initial physical car/role assignment before policy swap;
- initial ball field third;
- initial ball-height bins;
- initial closest-to-ball policy;
- initial boost-advantage bins if practical from the captured state bank.

### Paired-state analysis

For each 4-duel base-state family, retain all four outcomes and summarize how often:

- Rival wins all four;
- Rival wins 3/4;
- split 2/2;
- Rival wins 1/4;
- Rival wins 0/4;
- draws prevent a complete four-way decision.

This is the strongest control against initial-state advantage and side asymmetry and must not be replaced by an aggregate-only win rate.

## 6. Open-play behavioral telemetry

Reuse the existing match/behavioral telemetry where practical. Publish policy-separated:

- touches and touch share;
- first accepted touch after the restored start;
- same-player next-touch retention / opponent handoff;
- immediate forward / neutral / backward touch direction;
- net ball displacement before next touch/goal;
- wall/backboard continuations;
- demos;
- goal-entry X/Z placement;
- time from final touch to goal;
- goal scorer vs last toucher where available.

Also report these metrics by Rival side because the prior benchmark showed a material Blue/Orange scoring difference.

Do not label backward/lateral/wall/backboard play as inherently bad.

## 7. Integrity and evidence

This is not a release-certification milestone. Do not run unrelated pytest/Ruff/compileall/parity/regression ceremony.

Do perform only the targeted checks needed to trust this evaluation:

- frozen Rival checkpoint identity exact;
- pinned Nexto identity exact;
- existing validated policy adapters unchanged or, if touched, targeted fidelity recheck only for the changed path;
- state capture -> restore reproduces the captured simulator state to the precision required by the existing device representation;
- mirrored state transform is involutive within accepted numeric tolerance: mirror(mirror(S)) == S;
- four-way assignment counts exact;
- no duel begins in kickoff state;
- no kickoff/reset occurs after duel start;
- every duel ends by Rival goal, Nexto goal, or 60-second draw;
- telemetry capacity has zero overflow;
- timed hot path remains free of host state transfers.

Publish:

- a human-readable report under `docs/`;
- machine-readable summary and state-bank description under `results/rival2/nexto_open_play/`;
- per-duel ledger with base-state ID, mirror flag, source distribution, policy assignment, outcome and time-to-goal;
- paired-family summary;
- relevant implementation entrypoint(s).

## 8. Explicit future work — not authorized now

The user intends to train Rival later against **fake-kickoff strategies**, including opponents that immediately backflip/retreat to boost and intentionally concede first contact so Rival's kickoff hit is received by the defender.

Record that as future curriculum work only. Do **not** begin fake-kickoff training, add scripted fake-kickoff opponents to training, or change Rival's kickoff behavior during this open-play benchmark.

## Stop boundary

When the 16,384-duel open-play benchmark and evidence are complete, commit and push to `origin/main` and stop for review.

Do not train Rival against Nexto yet. Do not change rewards/PPO/model/physics. Do not build the viewer and do not begin v0.6.
