# RivalSim v0.5 — Rival 2.0 GPU-Native Training

Status: **AUTHORIZED / NOT YET IMPLEMENTED**

This milestone turns the completed v0.4 standard-Soccar 1v1 transition engine into a complete GPU-native reinforcement-learning system and defines the first training contract for **Rival 2.0**.

## Frozen starting boundary

Release commit:

`8a422a86c69f16f0d62073992e515575f88733b5`

Implementation commit:

`da34c6d8a9ad4eb6aaced955ef0fe96575e1ec56`

v0.4 is `PASS_GREEN` and remains a mandatory regression baseline:

- complete GPU-resident standard 1v1 lifecycle;
- all 34 boost pads;
- scoring and five kickoff layouts;
- demolition disable plus tick-360 respawn;
- deterministic reset/lifecycle state;
- inherited v0.3 A/B/C/D, v0.2.2 static, and v0.1 live parity gates;
- 191,748.10 aggregate simulated game-seconds/s at 131,072 worlds;
- zero timed hot-loop transfers.

Do not rewrite or regenerate published `results/v0.1/` through `results/v0.4/` evidence.

## Rival 2.0 means clean slate

v0.5 does **not** preserve training compatibility with the existing `Noxoni/Rival` bot.

Do not use its:

- Wisp-derived policy weights;
- 432-value observation contract;
- 90/158-action lookup tables;
- mechanics action appendices;
- old PPO implementation;
- old reward/curriculum code;
- deployment logits/action assumptions.

Those artifacts may remain historically useful but are not inputs to this milestone. Rival 2.0 is allowed to require a new v0.6 deployment adapter.

## Phase A — zero-copy tensor bridge + `RIVAL2_OBS_V1`

Expose the accepted RivalSim world state to PyTorch CUDA without routine copies.

Preferred architecture:

- create persistent torch views over Warp/CUDA storage where supported;
- use DLPack / Warp torch interoperability or an equivalently proven zero-copy path;
- prove device identity and storage aliasing;
- do not materialize NumPy/CPU world state in the rollout path.

Build a new fixed-size float32 observation tensor directly from RivalSim state:

`[world, agent, obs_dim]`

`RIVAL2_OBS_V1` must be clean-slate, symmetric, compact, and Markov-oriented. Required semantic blocks are defined in `RIVAL2_CONTRACT.md`.

Before PPO training begins, freeze:

- field order;
- normalization/scales;
- team-perspective transform;
- boost-pad canonical ordering;
- event/history semantics;
- exact `obs_dim`;
- schema/content hash.

No running host-side normalization is allowed.

## Phase B — fixed hybrid controller + actor/critic

The Rival 2.0 control standard is fixed:

- throttle: continuous `[-1, +1]`;
- steer: continuous `[-1, +1]`;
- pitch: continuous `[-1, +1]`;
- yaw: continuous `[-1, +1]`;
- roll: continuous `[-1, +1]`;
- jump: Bernoulli;
- boost: Bernoulli;
- handbrake/powerslide: Bernoulli.

Decision cadence remains `mechanics4`:

- physics: 120 Hz;
- policy: 30 Hz;
- sampled controller held for exactly four physics ticks.

Actor output contract:

- five Gaussian means;
- five Gaussian log standard deviations;
- three Bernoulli logits;
- 13 actor outputs total.

Sample analog controls in pre-tanh space and apply the correct tanh log-Jacobian correction in PPO log probabilities. Clamp log standard deviations to a safe configurable range. Deterministic inference uses `tanh(mean)` for analog channels and probability >= 0.5 for buttons.

Do not use a fixed action lookup, discrete action mask, 256-code vocabulary, or old Rival action table.

Implement a compact fixed-size actor/critic suitable for very large CUDA batches. Default architecture may be a shared MLP trunk with separate actor and critic heads, but the implementation must keep architecture/config explicit and checkpointed.

## Phase C — GPU-native rewards, episodes, resets, and rollout storage

Implement the initial `RIVAL2_REWARD_V1` contract from `RIVAL2_CONTRACT.md` directly on device.

The initial reward intentionally avoids mechanics-specific shaping. It is centered on:

- goals;
- ball progress toward the opponent goal;
- ball touches;
- demolitions.

The initial episode policy is also frozen in the contract:

- goal => terminated;
- no-touch timeout => truncated;
- hard episode-time limit => truncated;
- terminal/truncated worlds reset through the accepted deterministic kickoff lifecycle.

No curriculum/random-state system is required for v0.5. Build the trainer first; broader state curricula can be separately authorized later.

Rollout storage must remain GPU resident and include at minimum:

- observation;
- sampled 8-channel action;
- old log probability;
- value prediction;
- reward;
- terminated/truncated mask;
- bootstrap value state;
- policy/opponent version metadata as required.

Use a tensor layout that supports vectorized GAE and minibatch PPO without host packing.

## Phase D — GPU-native GAE/PPO and checkpointing

Implement standard clipped PPO for the hybrid distribution:

- GAE;
- bootstrapped returns;
- advantage normalization;
- clipped policy objective;
- value loss;
- entropy/exploration term;
- gradient clipping;
- configurable epochs/minibatches;
- optimizer state;
- save/resume.

The complete action log probability is the sum of:

- five tanh-squashed Gaussian log probabilities;
- three Bernoulli log probabilities.

GAE, return construction, shuffling/index generation, minibatch gather, losses, and optimizer execution must remain on CUDA during ordinary training.

Checkpoint files may leave the device. Saving a checkpoint must not mutate the active policy or stall every rollout iteration by design.

A checkpoint must contain enough information for exact training resume:

- actor/critic weights;
- optimizer state;
- training counters;
- PPO config;
- observation/reward/action contract hashes;
- RNG state required for reproducible policy sampling;
- opponent-pool metadata.

## Phase E — self-play and historical opponents

Baseline self-play uses the same trainable Rival 2.0 policy on both sides of the 1v1 world.

Add bounded historical-policy support so a configurable subset of worlds can play against frozen older Rival 2.0 checkpoints. Default initial contract:

- current-vs-current remains the majority;
- historical-opponent chance defaults to 20% once eligible snapshots exist;
- bounded resident pool defaults to 16 versions;
- opponent selection changes only at world reset;
- historical policies never receive gradients;
- no CPU policy inference in rollout generation.

If grouping many opponent versions causes pathological inference fragmentation, preserve the semantics while batching by policy version or use another GPU-native solution. Do not silently fall back to per-world CPU inference.

## Phase F — prove learning and find the real throughput point

First prove correctness with a bounded learning smoke:

- finite losses/gradients;
- actor parameters change;
- critic parameters change;
- checkpoint save/resume reproduces the expected counters/state;
- sampled controls stay within the contract;
- terminal/reset handling is correct across mixed worlds;
- reward/return/GAE tests match an independent small reference calculation.

Then run a short fixed-seed learning sanity campaign. The purpose is not to produce a strong bot; it is to prove the integrated system can improve a policy rather than merely execute optimizer steps.

Finally sweep practical world counts, including where memory permits:

- 8,192;
- 16,384;
- 32,768;
- 65,536;
- 131,072.

For every practical point report:

- rollout agent decisions/s;
- simulated game-seconds/s;
- actor/critic inference time;
- observation/reward/reset time;
- GAE time;
- PPO update samples/s and wall time;
- update frequency;
- sample age/policy version lag;
- VRAM peak;
- GPU utilization if measurable without disturbing the timed path;
- H2D/D2H bytes in the normal hot path;
- coefficient of variation.

Select the training configuration by **end-to-end learning throughput and freshness**, not by raw simulator throughput alone.

## Residency rule

The ordinary loop must be:

`RivalSim -> obs -> actor/critic -> sample controls -> RivalSim x4 -> rewards/dones -> rollout -> GAE/PPO -> updated weights`

with world state, observations, actions, rewards, rollout storage, advantages, returns, and model execution all remaining on GPU.

Allowed CPU responsibilities:

- launch/configuration;
- human-readable logging;
- infrequent metrics snapshots;
- checkpoint serialization;
- explicit offline diagnostics.

A Python orchestration layer is acceptable. A Python/CPU **data path** is not.

## Explicitly out of scope

Do not implement in v0.5:

- compatibility with old Rival/Wisp training weights or action tables;
- RLBot deployment;
- CPU RocketSim deployment/evaluation;
- Rocket League transfer validation;
- other game modes;
- arbitrary car/ball counts;
- rendering;
- generic Bullet support;
- mechanics-specific curricula or hand-authored action vocabularies;
- distributed multi-GPU training.

Those are later decisions. v0.6 remains the transfer gate.

## Completion boundary

v0.5 may be published `PASS_GREEN` only if:

1. `RIVAL2_OBS_V1`, action, reward, and episode contracts are frozen and hashed;
2. the hybrid 13-output actor produces the exact eight native controls at 30 Hz;
3. normal rollout state/data remain GPU resident with zero routine per-step H2D/D2H copies;
4. reward/terminal/reset/rollout semantics pass deterministic unit and integration gates;
5. GAE/PPO math passes independent reference tests and finite-gradient stress;
6. checkpoint save/resume works and preserves contract identities;
7. current-policy self-play works, and historical-opponent support works when enabled;
8. a bounded fixed-seed learning sanity run demonstrates measurable policy improvement on its declared metric;
9. practical world-count performance is swept and a stable selected point is reported;
10. all inherited v0.4/v0.3/v0.2.2/v0.1 regression gates remain green;
11. compact evidence/manifest is committed and remotely verified;
12. no v0.6 deployment/transfer work has begun.

If a genuine tensor-interoperability, correctness, memory, or performance boundary blocks completion, stop there with explicit evidence rather than moving work back to CPU.
