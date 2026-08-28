# RivalSim

RivalSim is an experimental **GPU-native Rocket League 1v1 transition engine** intended to accelerate training for [Noxoni/Rival](https://github.com/Noxoni/Rival).

## Native human demonstration recorder

The repository includes a production-oriented, read-only BakkesMod plugin that records the exact
native Rocket League `ControllerInput` and synchronized game state at the physics input-application
cadence. It supports continuous local/RLBot match demonstrations and Freeplay mechanic sessions,
uses a crash-recoverable lossless chunk format, and includes deterministic Python validation,
current 182-field observation-coverage reporting, direct native-120-Hz to Rival-120-Hz action
alignment metadata, and retained four-tick variation diagnostics for historical comparison. It
does not train Rival, modify rewards or physics, inject controls, or automate gameplay. See
[the complete recorder guide](docs/RIVAL2_HUMAN_DEMO_RECORDER.md).

The no-learning 120 Hz control transition is documented in
[the Rival 2.0 120 Hz transition report](docs/RIVAL2_120HZ_TRANSITION_V1.md). The active V2 line
uses one policy decision per 120 Hz physics tick; historical V1 policies and evidence remain bound
to their original 30 Hz/four-tick contract.

The project is deliberately narrower than RocketSim. The target is standard Soccar 1v1 only:

- two cars;
- one ball;
- standard DFH/Stadium_P arena;
- standard static-world boost-pad pickup/cooldown state;
- fixed 120 Hz physics;
- no rendering in the training benchmark path.

## RivalVis checkpoint viewer

RivalVis is a separate one-world Panda3D spectator for watching a Rival 2.0
checkpoint play an authoritative five-minute RivalSim match. It is intentionally
outside the trainer and never copies state from the 131,072-world training hot path.

Install the optional dependency and launch the latest acquisition checkpoint:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[viewer]"
.\.venv\Scripts\python.exe -m rivalsim.viewer --checkpoint checkpoints\rival2\acquisition_v1\rival2_acquisition_resume.pt
```

RivalVis discovers the standard sibling `RLBot-Rival/bot/collision_meshes`
checkout automatically. On another layout, pass `--collision-dir PATH` or set
`RIVALSIM_COLLISION_DIR`. Stochastic policy behavior is the default; add
`--deterministic` for reproducible debugging. See [RivalVis documentation](docs/RIVALVIS.md)
for cameras, playback controls, seeds, and architecture.

## Current boundary — v0.5 / Rival 2.0 overnight curriculum complete

The uninterrupted overnight curriculum resumed Campaign 04's exact update-120 Reward V2
checkpoint. Acquisition completed at update 420 after two consecutive 4,096-world held-out
evaluations reached <=1% no-touch truncation. The explicit reward-only transition to the base
`RIVAL2_REWARD_V1` preserved learned weights, optimizer, RNG, counters, opponent assignments,
historical policies, and live runtime state exactly.

Phase B then completed exactly 239 PPO updates / 2,004,877,312 additional Reward V1 samples.
The user prospectively extended the final timed continuation from three to six hours; Phase C
stopped at the first completed update crossing 21,600 seconds: update 5,403 /
**45,323,649,024 cumulative samples** at 21,601.926 seconds. All 5,283 continuation updates
passed integrity.

The six hourly held-out touch rates were `66.022572`, `75.795357`, `79.884180`, `75.911922`,
`83.108105`, and `85.483708` per simulated minute. Final goals/minute was `2.496403` and
no-touch truncation was `0.003418`. This is bounded stochastic self-play evidence, not external
Rocket League competence.

Complete evidence is in `docs/RIVAL2_OVERNIGHT_RESULTS.md` and
`results/rival2/overnight/`. The exact final resumable checkpoint is
`checkpoints/rival2/overnight/rival2_overnight_final_6h_resume.pt` with SHA-256
`4DC158DC2A9D16B79FB5FE7D868E3B50928AB113B55DFCC753F3734F8D87372E`.

No preflight/regression/parity/lint/test ceremony or viewer work was run, and v0.6 has not begun.

### Campaign 04 source checkpoint

Rival 2.0 Campaign 04 resumed the exact Campaign 03 checkpoint at update 12 / 100,663,296
cumulative samples, including optimizer, RNG, counter, opponent-assignment, and historical-policy
state. The unchanged Reward V2 line completed 108 more green updates and stopped exactly at
update 120 / **1,006,632,960 cumulative samples**. Update 121 did not run.

The authorized stochastic self-play curve increased touches/minute from `1.308672` at 100M to
`3.202896` at 250M, `6.453265` at 500M, `8.712013` at 750M, and `16.661451` at 1B. No-touch
truncation fell from `0.936279` at 100M to `0.550293` at 1B. The frozen primary-axis trend is
`CONTINUING`, although goal rate was non-monotonic and declined from `0.426426` at 750M to
`0.311649` at 1B. This remains bounded self-play evidence, not external Rocket League competence.

Complete evidence is in `docs/RIVAL2_CAMPAIGN04_RESULTS.md` and
`results/rival2/campaign04/`. The exact final resume checkpoint is under
`checkpoints/rival2/campaign04/` with SHA-256
`DB5AA09B2CAD40D4C1F5DB1014FDE245C58994A6948458212751724F782BE6B0`.

Rival 2.0 Campaign 03 is closed at update 12 / **100,663,296 agent decision samples**. It
preserved Reward V1 and introduced `RIVAL2_REWARD_V2` with one per-agent 30-Hz true-distance
approach delta, `(distance_before - distance_after) / 4096.0`, measured from decision start to
the final pre-reset transition state. The targeted GPU reward sign/reset-leakage smoke passed,
all 12 training updates stayed green, and the final checkpoint passed exact reload.

The single authorized final 4,096-world stochastic self-play evaluation increased touches per
simulated minute from Campaign 02's `0.291182` to `1.308672`, increased goals/minute from
`0.040362` to `0.243800`, and reduced no-touch truncation from `0.989746` to `0.936279`. This is
bounded training evidence, not external Rocket League competence. Exact results are in
`docs/RIVAL2_CAMPAIGN03_RESULTS.md` and `results/rival2/campaign03/`; the final full resume
checkpoint is under `checkpoints/rival2/campaign03/`.

Rival 2.0 Campaign 02 is closed with execution **`COMPLETE`** and the independently classified
behavioral result **`IMPROVED`**. It reproduced Campaign 01's initialization exactly, changed only
the campaign-layer entropy coefficient from `0.01` to `0.0`, and stopped at update 12 /
**100,663,296 agent decision samples**.

Final ordinary self-play touches/minute improved to `0.291182`, versus `0.272091` at initialization
and `0.175624` at Campaign 01 final. Stochastic touch differential against initialization improved
to `+35`, versus `+15` initially and `-46` in Campaign 01; goal differential was `-3`, slightly
worse than initialization's `-2` but better than Campaign 01's `-16`. Maximum KL was `0.008194`,
maximum clip fraction was `0.087534`, and final representative analog standard deviation remained
near `1.015` instead of rising toward `exp(1)`. Complete direct comparison evidence is in
`docs/RIVAL2_CAMPAIGN02_RESULTS.md` and `results/rival2/campaign02/`. This does not establish
external Rocket League competence.

Rival 2.0 Campaign 01 is also closed. Using the unchanged frozen v0.5 stack, a fresh policy trained
for exactly 12 PPO updates and stopped at **100,663,296 agent decision samples**, the first update
crossing the bounded 100M target. The 131,072-world horizon-32 capacity passed preflight; all 12
updates and five fixed evaluations passed numerical/state integrity, zero-transfer, checkpoint,
and exact-continuation gates.

Campaign execution is **`COMPLETE`**, while the independent behavioral result is honestly
**`DEGRADED`**. Final ordinary self-play touch rate fell from 0.272091 to 0.175624 per simulated
minute. Against frozen initialization, the final stochastic checkpoint lost 7–23 with a -46 touch
differential, and deterministic play lost 0–819. No setting was retuned in response. Complete
evidence is in `docs/RIVAL2_CAMPAIGN01_RESULTS.md` and `results/rival2/campaign01/`. The final full
resume checkpoint is published under `checkpoints/rival2/campaign01/`.

RivalSim v0.5 adds a clean-slate GPU-native reinforcement-learning system directly to the
accepted standard-Soccar transition engine. It freezes a 182-value symmetric float32 observation,
five tanh-Gaussian analog controls, three Bernoulli buttons, compact zero-sum reward and episode
contracts, four-tick/30-Hz cadence, CUDA rollout storage, GAE/PPO, exact checkpoint continuation,
current-policy self-play, and a bounded GPU historical-opponent pool.

The ordinary path remains device resident through 48 proven Warp/PyTorch CUDA aliases. The
selected 131,072-world complete rollout+GAE+PPO point reaches **2,233,901.63 agent samples/s** and
**89,505.78 simulated game-seconds/s** with **0.588% CV**, 14,414,032,896 peak observed VRAM bytes,
and zero timed H2D/D2H traffic. The bounded fixed-seed smoke improves its prospectively declared
held-out clipped PPO objective by 4.226 standard errors. This proves integrated learning is not a
no-op; it is not a skill or external-transfer claim.

All v0.4/v0.3/v0.2.2/v0.1 regression gates remain green and all published prior evidence is
unchanged. Exact contracts and evidence are in `docs/RIVAL2_TRAINING_CONTRACT.md`,
`docs/V0_5_RESULTS.md`, and `results/v0.5/manifest.json`.

### Frozen v0.4 simulator baseline

RivalSim v0.4 completes the bounded standard-Soccar 1v1 game-transition milestone. The public
`CompleteWorldSim` composes the accepted v0.3 two-Octane/one-ball physics with GPU-resident state
for all 34 boost pads, goals and score attribution, five deterministic standard kickoff layouts,
demolition disable/timing, four source-valid respawn locations per team, world/episode clocks,
raw lifecycle events, and deterministic full-world reset.

Lifecycle choices are explicit state. Kickoff selectors advance modulo five and respawn selectors
advance modulo four; no host-global RNG, pointer value, allocator layout, case ID, or expected
output participates in the runtime. Car membership does not change during kickoff, demolition,
or respawn, so the source-proven per-world v0.3 car visitation order is preserved.

The content-addressed native lifecycle authority passes:

- every one of 34 pads for both cars: **68 / 68 pickup cases**;
- source float32 recharge boundaries: **1,201 ticks** for large pads and **480 ticks** for small;
- both visitation-order branches for pad contention;
- **6 / 6 goal-boundary cases** and both scoring directions;
- **5 / 5** standard two-car kickoff layouts;
- both teams at all four respawn locations: **8 / 8 poses**;
- exact demolition timer and respawn at **tick 360**;
- deterministic 64-world, 400-tick mixed lifecycle/reset stress with zero timed transfers.

The inherited v0.3 Phase A/B/C/D gates remain 31,216/31,216, 8,192/8,192,
8,192/8,192 against both branches, and 512/512 across both branches. The v0.2.2 static corpus
remains 39,236/39,236, v0.1 remains 27/27, both 4,608-ray backends pass, and the configured
repository suite is 70/70 passing. Published v0.1 through v0.3 evidence is byte-for-byte
unchanged.

The complete v0.4 path reaches **191,748.10 aggregate simulated game-seconds/s**
(23.01 million world ticks/s) at 131,072 worlds with **0.856% CV**, retaining 97.52% of v0.3.
The reset-heavy path reaches **225,005.06 sim-s/s** and **3.375 million reset transitions/s**
with **0.723% CV**. Both timed paths record zero host/device transfers.

RocketSim does not define a training episode terminal/truncation policy. The frozen v0.4 layer
therefore exports policy-neutral raw lifecycle state and keeps `terminated=truncated=0`; v0.5
adds its separately hashed trainer-owned episode policy without changing v0.4 authority.

### Explicitly excluded from the completed v0.5 boundary

RivalSim v0.5 does **not** implement RLBot/CPU RocketSim deployment, Rocket League transfer
validation, legacy Rival/Wisp compatibility, mechanics curricula, multi-GPU training, arbitrary
body counts, other game modes, rendering, or a generic Bullet API. v0.6 has not begun.

## Architecture

RivalSim stays **GPU-resident and batched**. Do not build a Python object graph per environment and do not round-trip world state through CPU every tick.

The v0.2 arena is a single shared GPU asset. Extracted Rocket League collision meshes are not committed to this public repository; only loader code, provenance, hashes, statistics and reproduction instructions belong in Git.

NVIDIA Warp remains the primary implementation layer until profiling proves a reason to replace a measured hotspot with native CUDA/C++.

## Performance references

Current full Rival CPU RocketSim/RLGym reference:

- 56 environments;
- 12,039 agent-steps/s;
- 200.65 aggregate simulated game-seconds/s.

RivalSim remains a partial simulator, so this is a system reference rather than an apples-to-apples comparison.

The v0.4 package classifies the complete game-transition path as:

- **PASS_GREEN:** every lifecycle/fidelity/regression gate passes and throughput is >=100,000
  sim-s/s;
- **PAUSE_PERF:** local parity passes but throughput is <100,000 sim-s/s;
- **PAUSE_FIDELITY:** any required local parity failure remains.

## Published v0.5 result

The current result package is:

- `docs/V0_5_RESULTS.md`;
- `docs/REPRODUCING_V0_5.md`;
- `docs/RIVAL2_TRAINING_CONTRACT.md`;
- `results/v0.5/tensor_bridge.json`;
- `results/v0.5/observation.json`;
- `results/v0.5/action_distribution.json`;
- `results/v0.5/reward_episode.json`;
- `results/v0.5/rollout_gae.json`;
- `results/v0.5/ppo.json`;
- `results/v0.5/checkpoint_resume.json`;
- `results/v0.5/self_play.json`;
- `results/v0.5/learning_smoke.json`;
- `results/v0.5/benchmark.json`;
- `results/v0.5/regression.json`;
- `results/v0.5/manifest.json`.

## Published Rival 2.0 Campaign 01 result

The completed bounded campaign package is:

- `docs/RIVAL2_CAMPAIGN01_RESULTS.md`;
- `results/rival2/campaign01/config.json`;
- `results/rival2/campaign01/preflight.json`;
- `results/rival2/campaign01/checkpoints.json`;
- `results/rival2/campaign01/evaluation_000m.json`;
- `results/rival2/campaign01/evaluation_010m.json`;
- `results/rival2/campaign01/evaluation_025m.json`;
- `results/rival2/campaign01/evaluation_050m.json`;
- `results/rival2/campaign01/evaluation_100m.json`;
- `results/rival2/campaign01/training_curve.json`;
- `results/rival2/campaign01/summary.json`;
- `checkpoints/rival2/campaign01/rival2_campaign01_100m_resume.pt`.

## Published Rival 2.0 Campaign 02 result

The controlled entropy-off campaign package is:

- `docs/RIVAL2_CAMPAIGN02_RESULTS.md`;
- `results/rival2/campaign02/config.json`;
- `results/rival2/campaign02/initialization_control.json`;
- `results/rival2/campaign02/checkpoints.json`;
- `results/rival2/campaign02/evaluation_000m.json`;
- `results/rival2/campaign02/evaluation_010m.json`;
- `results/rival2/campaign02/evaluation_025m.json`;
- `results/rival2/campaign02/evaluation_050m.json`;
- `results/rival2/campaign02/evaluation_100m.json`;
- `results/rival2/campaign02/training_curve.json`;
- `results/rival2/campaign02/comparison_campaign01.json`;
- `results/rival2/campaign02/optimizer_diagnosis.json`;
- `results/rival2/campaign02/summary.json`;
- `checkpoints/rival2/campaign02/rival2_campaign02_100m_resume.pt`.

## Published v0.4 authority and result

The current result package is:

- `docs/V0_4_RESULTS.md`;
- `docs/REPRODUCING_V0_4.md`;
- `docs/V0_4_AUTHORITY.md`;
- `results/v0.4/boost_pads.json`;
- `results/v0.4/goals_kickoff.json`;
- `results/v0.4/demolition_respawn.json`;
- `results/v0.4/match_lifecycle.json`;
- `results/v0.4/oracle_data.json`;
- `results/v0.4/rules_source.json`;
- `results/v0.4/regression.json`;
- `results/v0.4/benchmark.json`;
- `results/v0.4/manifest.json`.

## Published v0.3 authority and result

The current result package is:

- `docs/V0_3_RESULTS.md`;
- `docs/REPRODUCING_V0_3.md`;
- `docs/V0_3_ORACLE_CACHE.md`;
- `results/v0.3/ball_world.json`;
- `results/v0.3/car_ball.json`;
- `results/v0.3/car_car.json`;
- `results/v0.3/integrated.json`;
- `results/v0.3/oracle_data.json`;
- `results/v0.3/source_port.json`;
- `results/v0.3/regression.json`;
- `results/v0.3/benchmark.json`;
- `results/v0.3/manifest.json`.

## Published v0.2.2 authority and result

The frozen v0.2.2 result package is:

- `docs/V0_2_2_RESULTS.md`;
- `docs/REPRODUCING_V0_2_2.md`;
- `docs/V0_2_2_ORACLE_CACHE.md`;
- `results/v0.2.2/oracle_data.json`;
- `results/v0.2.2/source_port.json`;
- `results/v0.2.2/parity.json`;
- `results/v0.2.2/regression.json`;
- `results/v0.2.2/benchmark.json`;
- `results/v0.2.2/manifest.json`.

## Published v0.2.1 authority and result

The active handoff is preserved at `handoff/v0.2.1/CODEX_START_PROMPT.md`. The immediate
2026-08-23 user steering adjustment governs the final 1/4/8/12-tick validation boundary and is
recorded in the v0.2.1 evidence and reproduction guide.

The result package is:

- `docs/V0_2_1_RESULTS.md`;
- `docs/REPRODUCING_V0_2_1.md`;
- `results/v0.2.1/divergence_index.json`;
- `results/v0.2.1/parity.json`;
- `results/v0.2.1/coverage.json`;
- `results/v0.2.1/benchmark.json`;
- `results/v0.2.1/manifest.json`.

## Published v0.2 authority

The root prompt and completed bounded package are preserved at:

`handoff/v0.2/CODEX_START_PROMPT.md`

Package contents:

- `handoff/v0.2/README.md`
- `handoff/v0.2/V0_2_SPEC.md`
- `handoff/v0.2/BENCHMARK_AND_PARITY.md`
- `handoff/v0.2/CODEX_START_PROMPT.md`
- `handoff/v0.2/PACKAGE_MANIFEST.md`

The result package is:

- `docs/V0_2_RESULTS.md`;
- `docs/REPRODUCING_V0_2.md`;
- `results/v0.2/benchmark.json`;
- `results/v0.2/parity.json`;
- `results/v0.2/manifest.json`.

## Relationship to RocketSim

RocketSim remains the primary CPU physics oracle until RivalSim earns replacement through performance **and transfer fidelity**.

RivalSim is a training transition engine, not a replacement Rocket League client. Work is intended for offline bot training and research, not cheating in online Rocket League.
