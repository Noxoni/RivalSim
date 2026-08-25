# RivalSim v0.5 — Acceptance and Evidence Gates

This document defines the blocking acceptance protocol for Rival 2.0 GPU-native training.

## 1. Frozen baseline integrity

Before implementation:

- verify `origin/main` is the authorized v0.4 handoff parent;
- record v0.4 release `8a422a86c69f16f0d62073992e515575f88733b5`;
- record v0.4 implementation `da34c6d8a9ad4eb6aaced955ef0fe96575e1ec56`;
- hash the v0.4 published evidence set.

At v0.5 completion, all prior published evidence must remain byte-for-byte unchanged.

## 2. Tensor interoperability gate

Prove the normal training path does not round-trip world state through CPU memory.

Required evidence:

- CUDA device identity for RivalSim and PyTorch tensors;
- zero-copy storage/view proof for the selected Warp/PyTorch bridge or equivalent;
- pointer/storage alias evidence where meaningful;
- contiguous/dtype/shape assertions;
- no NumPy conversion in rollout generation;
- no `.cpu()`, host list packing, per-world Python object construction, or device synchronization in the timed decision loop except where explicitly required by the backend and proven not to transfer state.

Instrument timed H2D and D2H traffic. Ordinary rollout must report zero routine state/action/reward transfer.

## 3. Observation contract gate

Freeze `RIVAL2_OBS_V1` before PPO training.

Required checks:

- exact field-order schema;
- exact dimension;
- finite float32 output;
- deterministic output for identical world state;
- blue/orange perspective symmetry using a proper 180-degree arena rotation, not a reflection;
- canonical boost-pad ordering under team perspective;
- fixed normalization constants;
- self/opponent identity is correct after perspective transform;
- demo/jump/flip/contact/timer state comes from the accepted RivalSim state, not reconstructed heuristically;
- previous action/history state updates only at policy decision boundaries;
- no hidden dependency on host-global state.

Build a deterministic corpus spanning kickoff, ordinary driving, aerials, wall contact, ball contact, pad cooldowns, demos, respawns, goals, and reset boundaries. Hash the schema and corpus outputs.

## 4. Hybrid action distribution gate

For actor output `[mu5, log_std5, button_logits3]`:

- analog sample uses a Gaussian in pre-tanh space;
- emitted analog controls are finite and within `[-1,+1]`;
- digital controls are exactly 0/1 in the simulator input;
- stochastic log probability includes the tanh Jacobian correction;
- Bernoulli log probability is summed for the three buttons;
- deterministic inference uses `tanh(mu)` plus button probability >= 0.5;
- entropy/KL diagnostics are finite;
- log-standard-deviation clamp is explicit and configurable;
- same RNG state + same inputs + same weights => same sampled actions;
- no fixed action table or lookup vocabulary participates.

Compare the GPU implementation against an independent small CPU/reference calculation on deterministic random tensors. This reference is a math oracle only and is not part of the runtime.

## 5. Mechanics4 cadence gate

The action selected at one policy boundary must remain unchanged for exactly four 120-Hz physics ticks.

Test:

- ordinary ground controls;
- jump press/release sequences;
- boost toggles;
- powerslide;
- simultaneous pitch/yaw/roll;
- goal/reset boundary;
- demolition/respawn boundary.

Policy decisions must occur at 30 Hz in simulated game time with no accidental extra inference inside the four-tick interval.

## 6. Reward and episode gate

Freeze `RIVAL2_REWARD_V1` and the v0.5 episode policy before the learning smoke.

Required semantics:

- rewards are computed once per 30-Hz decision transition from the complete four-tick interval;
- goal event is counted once and yields symmetric zero-sum goal reward;
- ball-progress shaping is zero-sum and based on canonical attacking direction;
- touch shaping is event-based and cannot repeat from a persistent contact bit;
- demolition shaping uses the accepted v0.4 demo event and cannot double-count during the disabled interval;
- goal => `terminated=1` for both agents;
- no-touch timeout => `truncated=1` for both agents;
- hard episode-duration limit => `truncated=1` for both agents;
- terminated/truncated worlds bootstrap values correctly under PPO semantics;
- reset happens once at the documented boundary and returns a clean accepted kickoff state.

Reference-test reward and done outputs on deterministic hand-constructed transitions.

## 7. Rollout-buffer gate

Required buffer fields:

- observations;
- eight emitted controls;
- old action log probability;
- value prediction;
- reward;
- terminated;
- truncated;
- bootstrap/nonterminal mask as required by the implementation;
- policy version;
- opponent version where applicable.

Required properties:

- device resident;
- no stale rows after asynchronous per-world resets;
- correct time/world/agent indexing;
- no cross-world contamination;
- reproducible hashes for a small deterministic rollout;
- memory use is bounded and reported.

## 8. GAE gate

Test GPU GAE against an independent small reference across:

- no terminal state;
- true termination;
- truncation with bootstrap;
- multiple worlds with different done positions;
- two agents/world;
- rollout ending mid-episode;
- mixed reward signs;
- constant-value analytic cases.

Report maximum absolute advantage and return error. Any semantic mismatch in terminal/truncation bootstrapping blocks PPO.

## 9. PPO objective gate

Test a frozen small batch against an independent reference implementation for:

- probability ratio;
- clipped surrogate objective;
- value loss;
- entropy term;
- total loss;
- approximate KL;
- clip fraction.

Then require:

- finite forward values;
- finite gradients;
- finite optimizer state;
- nonzero actor parameter update;
- nonzero critic parameter update;
- gradient clipping acts at the configured threshold;
- no autograd graph retained across iterations unintentionally.

## 10. Checkpoint/resume gate

Save a checkpoint after a known iteration, reload into a fresh trainer, and verify:

- weights match;
- optimizer state matches;
- counters match;
- observation/action/reward contract hashes match;
- PPO configuration matches;
- RNG state required for reproducible sampling is restored;
- historical-opponent metadata is restored;
- the next deterministic evaluation action/value outputs match.

Refuse incompatible contract hashes by default.

## 11. Self-play/opponent gate

Current-vs-current:

- both agent slots use the same trainable policy weights;
- each side receives its own canonical perspective;
- gradients are accumulated from both sides correctly.

Historical opponents:

- selection occurs only at reset;
- selected version remains fixed for that episode;
- frozen opponent gets no gradient;
- current policy remains trainable;
- no CPU inference path is introduced;
- deterministic seed reproduces opponent assignment.

## 12. Learning sanity gate

Run a bounded fixed-seed training campaign from a newly initialized Rival 2.0 policy.

Before the run, declare one simple metric that should improve under `RIVAL2_REWARD_V1`, such as mean episodic return over a fixed deterministic evaluation set.

The gate is not a skill benchmark. It only requires evidence that:

- optimization is not a no-op;
- the policy changes;
- the critic learns nonconstant values;
- the declared evaluation metric improves materially beyond seed/noise variation over the bounded smoke.

Do not tune the reward after looking at the gate result unless the milestone is explicitly reopened and the contract identity changes.

## 13. Performance sweep

Measure practical world counts from the authorized list that fit in VRAM.

At each point:

- warm up;
- run at least five timed repeats;
- report CV;
- separate rollout-generation and PPO-update timings;
- report complete iteration wall time;
- report agent decision samples/s generated;
- report PPO samples/s consumed;
- report simulated game-seconds/s;
- report VRAM peak;
- report hot-loop H2D/D2H;
- report policy-version/sample-age lag.

Selected points used for release claims must have CV < 5%.

No absolute throughput number may justify moving observation/reward/rollout/GAE/PPO work to CPU. If the GPU implementation is unexpectedly slow, profile and report the boundary.

## 14. Inherited simulator regressions

Before v0.5 release, rerun all mandatory v0.4 release regressions, including:

- v0.4 lifecycle authority gates;
- v0.3 Phase A/B/C/D;
- v0.2.2 39,236-case static gate;
- v0.1 27 live RocketSim scenarios;
- both 4,608-ray backends;
- deterministic mixed lifecycle stress;
- repository tests/lint/compile/diff checks.

Training integration may not weaken simulator fidelity.

## 15. Final evidence package

Commit compact evidence under `results/v0.5/`, including at minimum:

- `tensor_bridge.json`;
- `observation.json`;
- `action_distribution.json`;
- `reward_episode.json`;
- `rollout_gae.json`;
- `ppo.json`;
- `checkpoint_resume.json`;
- `self_play.json`;
- `learning_smoke.json`;
- `benchmark.json`;
- `regression.json`;
- `manifest.json`.

Human-readable reports should include:

- `docs/V0_5_RESULTS.md`;
- `docs/REPRODUCING_V0_5.md`;
- `docs/RIVAL2_TRAINING_CONTRACT.md`.

The final manifest must identify:

- v0.5 implementation commit;
- v0.4 frozen parent/release;
- contract hashes;
- policy architecture/config hash;
- selected performance point;
- zero-copy/residency verdict;
- learning-smoke verdict;
- inherited regression verdicts;
- `v0_6_begun: false`.

Do not publish `PASS_GREEN` unless every blocking gate above passes.
