# Rival 2.0 Training Contract — v0.5

This document freezes the intended v0.5 policy/training semantics. Exact observation field indices and hashes are finalized by implementation before PPO training begins, but the semantic contract below is controlling.

## 1. Identity

Policy generation: **Rival 2.0**

Rival 2.0 is not required to load, imitate, preserve, or export the legacy Rival/Wisp training format. The old Rival observation/action schemas are explicitly non-authoritative for this milestone.

## 2. Decision cadence

Physics rate: 120 Hz.

Policy rate: 30 Hz.

One sampled controller action is held unchanged for exactly four physics ticks.

The policy sees one observation and receives one reward transition per four-tick decision interval.

## 3. Native controller contract

External action order is exactly:

1. throttle — continuous `[-1,+1]`;
2. steer — continuous `[-1,+1]`;
3. pitch — continuous `[-1,+1]`;
4. yaw — continuous `[-1,+1]`;
5. roll — continuous `[-1,+1]`;
6. jump — binary `{0,1}`;
7. boost — binary `{0,1}`;
8. handbrake/powerslide — binary `{0,1}`.

There is no discrete action table.

### Actor distribution

The actor emits 13 scalar outputs per agent:

- `mu[5]`;
- `log_std[5]`;
- `button_logits[3]`.

Analog sampling:

`u = mu + exp(log_std) * epsilon`

`analog_action = tanh(u)`

with `epsilon ~ Normal(0,1)`.

The PPO log probability must use the Gaussian pre-tanh density minus the tanh change-of-variables correction.

Button sampling:

`button ~ Bernoulli(sigmoid(logit))`

The combined action log probability is the sum of all five analog and three button log probabilities.

Default safe `log_std` clamp: `[-5, +1]`, configurable and checkpointed.

Deterministic inference:

- analog = `tanh(mu)`;
- button = `1` when sigmoid(logit) >= 0.5, else `0`.

No state-dependent action mask is part of v0.5. Impossible/no-effect inputs are allowed to be learned naturally from outcomes.

## 4. `RIVAL2_OBS_V1`

The observation is a fixed-size float32 tensor built separately for each agent from that agent's perspective.

### Team canonicalization

Every agent observes itself as attacking toward positive arena Y.

For the orange perspective, apply a proper 180-degree rotation around world Z to dynamic vectors/positions:

- x -> -x;
- y -> -y;
- z -> z.

Apply the same proper rotation to basis vectors, linear velocity, angular velocity, relative vectors, and fixed boost-pad indexing.

Do not use a mirror/reflection that changes handedness.

### Fixed physical normalization

Use explicit domain scales, not a host running mean/variance. At minimum normalize:

- position by fixed arena extents/declared position scale;
- linear velocity by the authoritative relevant speed scale;
- angular velocity by a fixed declared angular scale;
- boost to `[0,1]`;
- timers by their authoritative maximum/relevant duration;
- binary flags as 0/1.

The exact constants become part of the observation schema hash.

### Required observation blocks

The exact flattened order is frozen during implementation, but `RIVAL2_OBS_V1` must contain all of the following.

#### Ball

- canonical position xyz;
- canonical linear velocity xyz;
- canonical angular velocity xyz.

#### Self car

- canonical position xyz;
- canonical linear velocity xyz;
- forward basis xyz;
- up basis xyz;
- canonical angular velocity xyz;
- boost;
- on-ground state;
- jump availability/state;
- flip/dodge availability/state;
- demolition state and normalized remaining timer;
- four wheel-contact flags;
- source-backed jump/flip/air timers or equivalent internal mechanic state required to make the action-response process Markov-oriented.

#### Opponent car

The same physical/mechanic state block as self, transformed into the observing agent's canonical frame.

#### Relative state

- ball position relative to self xyz;
- ball velocity relative to self xyz;
- opponent position relative to self xyz;
- opponent velocity relative to self xyz.

These are intentionally redundant with absolute state to make local control easier without inserting tactical heuristics.

#### Boost pads

All 34 standard Soccar pads in a frozen canonical order.

Per pad include:

- active flag;
- normalized remaining cooldown/time-to-ready.

Pad positions are static and therefore do not need to be repeated in every observation once canonical index order is frozen.

#### Previous action

The previously emitted eight native controller values at the prior 30-Hz decision boundary.

#### Lifecycle/event state

Include the minimal current/interval state needed for learning and temporal correctness, including:

- kickoff/reset indicator;
- self ball-touch event since the previous decision;
- opponent ball-touch event since the previous decision;
- self demolition event since the previous decision;
- opponent demolition event since the previous decision;
- normalized episode age;
- normalized no-touch age.

Do not include reward values, future state, expected action, case IDs, or handcrafted tactical labels.

### Observation design rule

Prefer direct state over derived strategy. Relative geometry is allowed; hardcoded concepts such as "challenge now", "shooting lane", "good aerial", or old Rival tactical metrics are not part of `RIVAL2_OBS_V1`.

Before training, publish the exact field index table, exact dimension, normalization constants, pad remap, and schema hash.

## 5. `RIVAL2_REWARD_V1`

The initial reward is deliberately compact and zero-sum. It should provide enough dense signal to bootstrap learning without prescribing mechanics.

For blue define canonical attacking ball Y directly. Orange receives the exact negative team reward for every transition.

### Goal

On a scoring event:

- scoring side: `+10.0`;
- conceding side: `-10.0`.

Count the event once.

### Ball progress

For each 30-Hz transition, before any terminal reset:

`progress = 0.5 * (canonical_ball_y_after - canonical_ball_y_before) / 5120`

Blue receives `progress`; orange receives `-progress`.

If implementation uses a slightly different authoritative Soccar Y normalization constant, freeze it before the first PPO run and include it in the reward hash. Do not tune it after observing learning-smoke results without changing the contract version.

### Touch event

For each unique ball-touch event in the interval:

- touching side: `+0.05`;
- other side: `-0.05`.

A persistent contact may not repeatedly generate touch reward without a new accepted touch event.

### Demolition event

For each unique accepted demolition event:

- demolishing side: `+0.10`;
- demolished side: `-0.10`.

Do not reward the disabled timer or respawn itself.

### No other v0.5 shaping

`RIVAL2_REWARD_V1` does not reward or penalize:

- boost pickup/use;
- aerial height;
- flips;
- speed;
- powerslide;
- recoveries;
- wall play;
- proximity to ball;
- possession heuristics;
- saves/shots inferred by heuristics.

Those behaviors are free to emerge from the game objective and can be revisited under a later reward contract if evidence requires it.

## 6. Episode contract

Episode state belongs to the trainer, not RocketSim authority.

### Termination

A goal terminates the current episode for both agents after the goal reward is recorded.

### Truncation

Truncate for both agents when either condition is reached without a goal:

- no accepted ball touch for 15.0 simulated seconds;
- episode age reaches 45.0 simulated seconds.

For truncation, bootstrap the critic from the final pre-reset observation according to standard PPO/GAE semantics.

For true goal termination, do not bootstrap beyond the terminal state.

### Reset

After terminal/truncated accounting, reset that world through the accepted deterministic standard kickoff lifecycle from v0.4.

v0.5 does not add a random-state curriculum. Every new episode begins from a source-valid standard kickoff selected by the existing per-world deterministic kickoff selector.

## 7. Initial actor/critic architecture

Use a fixed-size CUDA-friendly MLP rather than an entity-attention system for the first Rival 2.0 trainer.

Default contract:

- shared trunk: 3 hidden layers of 512 units;
- activation: SiLU or another explicitly frozen smooth activation selected once before training;
- actor head: 13 outputs;
- critic head: 1 output.

The exact chosen activation, initialization, dtype/autocast policy, and parameter count must be recorded in evidence and checkpoints.

Do not change network width/depth during the v0.5 learning-smoke run after the contract is frozen.

## 8. PPO defaults

Defaults are configuration, not physics authority, but must be recorded and checkpointed.

Initial values:

- gamma: `0.995` per 30-Hz decision;
- GAE lambda: `0.95`;
- PPO clip range: `0.20`;
- value-loss coefficient: `0.50`;
- entropy coefficient: `0.01`;
- max gradient norm: `0.50`;
- actor/critic optimizer: Adam;
- initial learning rate: `3e-4` unless a single shared optimizer requires another explicitly recorded value;
- PPO epochs: `2`;
- rollout horizon: configurable, initial target `32` decisions;
- minibatch size: chosen to use the RTX 5090 efficiently and recorded by the benchmark.

The implementation may reduce world count or adjust minibatch size for memory/performance. Do not silently change the action, observation, reward, or episode semantics to improve throughput.

## 9. Self-play contract

Current-policy self-play is the baseline: both sides use the same trainable Rival 2.0 weights and contribute experience.

Historical policy support:

- enabled only after valid snapshots exist;
- default probability: 20% of worlds at reset;
- default resident pool bound: 16 snapshots;
- opponent selection remains fixed until that world's next reset;
- historical policy receives no gradient;
- current agent remains the learning side for those worlds;
- opponent inference remains on GPU.

Selection RNG/state must be reproducible and checkpointable.

## 10. Deployment consequence

Rival 2.0 will need a new deployment adapter in v0.6 that reproduces `RIVAL2_OBS_V1`, 30-Hz cadence, and the hybrid eight-channel deterministic controller in RocketSim/RLBot.

That adapter is explicitly not part of v0.5.
