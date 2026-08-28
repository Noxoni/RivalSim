# Rival 2.0 Training Contract — RivalSim v0.5

Status: **FROZEN / PASS_GREEN**

This is the immutable historical 30 Hz V1 contract. It is not silently reinterpreted by the
active 120 Hz V2 line. See `docs/RIVAL2_120HZ_TRANSITION_V1.md` for the new action, observation,
reward, PPO timing, checkpoint, and validation identities.

This document records the implemented Rival 2.0 v0.5 training interface. The machine-readable
authority is `rivalsim/rival2_contracts.py` plus the hashes in `results/v0.5/manifest.json`.
Changing any observation, action, reward, or episode semantic requires a new contract identity.

## Contract identities

| Contract | SHA-256 |
|---|---|
| `RIVAL2_OBS_V1` | `10669E7D240D553BEA601F8AD7AEF9F9029310E55EA6DC4072E866F34BB218AF` |
| `RIVAL2_ACTION_V1` | `145AF5B49E1A0C85209022A6DE862F05EF996EB89B90B124072A59BC9936727B` |
| `RIVAL2_REWARD_V1` | `E3C97C7B3EA97D15F6AFB3AF21C40BAFBD206F0ED1124BAD6EA2C5A2ED14786F` |
| `RIVAL2_EPISODE_V1` | `E22B6014C6D975D700D1129B9F554D6F35E4CA5003F1C1BD09C7D394D4F9347E` |

## Observation: `RIVAL2_OBS_V1`

The observation is contiguous float32 with shape `[world, agent, 182]`. Each agent is placed in
its own canonical perspective and always attacks toward positive Y. Orange uses a proper
180-degree rotation around +Z: `(x,y,z) -> (-x,-y,z)`. The same rotation applies to positions,
linear and angular vectors, basis vectors, and relative vectors. This is not a reflection.

The exact flattened index map is:

| Indices | Fields, in exact order |
|---|---|
| 0–8 | ball position xyz; linear velocity xyz; angular velocity xyz |
| 9–47 | self car block listed below |
| 48–86 | opponent car block listed below, with identical internal order |
| 87–98 | relative ball position xyz; ball velocity xyz; opponent position xyz; opponent velocity xyz |
| `99 + 2i` | canonical boost pad `i` active flag, for every `i` in 0–33 |
| `100 + 2i` | canonical boost pad `i` normalized cooldown, for every `i` in 0–33 |
| 167–174 | previous throttle, steer, pitch, yaw, roll, jump, boost, handbrake |
| 175–181 | kickoff/reset; self touch; opponent touch; self demoed; opponent demoed; episode age; no-touch age |

The 39-field car block, with offset 9 for self and 48 for opponent, is exactly:

| Relative indices | Fields, in exact order |
|---|---|
| 0–2 | position xyz |
| 3–5 | linear velocity xyz |
| 6–8 | forward basis xyz |
| 9–11 | up basis xyz |
| 12–14 | angular velocity xyz |
| 15–25 | boost; on-ground; has-jumped; is-jumping; has-double-jumped; has-flipped; is-flipping; jump-available; dodge-available; is-demoed; demo timer |
| 26–29 | wheel contact front-left, front-right, back-left, back-right |
| 30–38 | jump time; air time; air time since jump; flip time; boosting time; time since boosted; is-supersonic; supersonic time; sticky ticks |

The orange canonical pad remap, indexed by canonical pad, is exactly:

`[1,0,5,4,3,2,33,32,31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6]`

The complete 182-name list is also embedded verbatim in `results/v0.5/observation.json`.

### Fixed normalization

| Quantity | Divisor / scale |
|---|---:|
| position x, y, z | 4096, 5120, 2044 uu |
| car linear velocity | 2300 uu/s |
| ball linear velocity | 6000 uu/s |
| angular velocity | 6 rad/s |
| boost | 100 |
| demolition timer | 3 s |
| jump time | 0.2 s |
| air time | 1.25 s |
| flip time | 0.95 s |
| boosting time | 0.1 s |
| time since boosted | 1 s |
| supersonic time | 1 s |
| sticky ticks | 3 ticks |
| episode age | 5,400 ticks / 45 s |
| no-touch age | 1,800 ticks / 15 s |

Timer outputs are clamped to `[0,1]`. Binary state is encoded as 0/1. Previous action and interval
event state update only at 30-Hz decision boundaries. There is no running host normalization,
reward leakage, case ID, expected output, tactical label, or host-global state dependency.

## Hybrid native action: `RIVAL2_ACTION_V1`

External controller order is exactly throttle, steer, pitch, yaw, roll, jump, boost, handbrake.
The first five values are continuous in `[-1,+1]`; the last three are exactly 0/1. One action is
selected at 30 Hz and held unchanged for exactly four 120-Hz physics ticks.

The actor emits 13 float32 values in this order:

1. five means for throttle, steer, pitch, yaw, roll;
2. five log standard deviations in the same order;
3. three logits for jump, boost, handbrake.

Log standard deviation is clamped to `[-5,+1]`. Analog controls sample a Gaussian in pre-tanh
space and use the tanh change-of-variables Jacobian in the PPO log probability. Buttons use
Bernoulli probabilities and their three log probabilities are summed with the five analog terms.
Deterministic inference is `tanh(mean)` plus `sigmoid(logit) >= 0.5`. No lookup table, discrete
vocabulary, action mask, old Rival parser, or Wisp policy participates.

## Reward and episode contracts

`RIVAL2_REWARD_V1` is computed once per complete four-tick transition and is exactly zero-sum:

- goal: +10 scoring side / -10 conceding side, once;
- ball progress: `0.5 * delta(canonical ball Y) / 5120`;
- unique accepted ball-touch entry: +0.05 / -0.05;
- unique accepted demolition event: +0.10 / -0.10;
- no other v0.5 shaping.

A persistent ball contact is latched and cannot generate another touch reward until separation
and a new accepted contact entry. Disabled time and respawn do not repeat demolition reward.

`RIVAL2_EPISODE_V1` terminates both agents on a goal. It truncates both agents after 15 seconds
without an accepted touch or at 45 seconds total. A goal does not bootstrap. A truncation uses the
critic value of the final pre-reset observation. Accounting happens before exactly one selective
reset through the accepted deterministic standard-kickoff lifecycle.

## Actor/critic and PPO defaults

The float32 model is a shared three-layer 512-unit SiLU trunk with a 13-output actor and one-output
critic. It has 626,190 parameters, no autocast, orthogonal sqrt(2) trunk initialization, and 0.01
actor/critic head initialization. Policy configuration hash:

`58C7409F34EA24CB7FAE7505A7F5FE2CC1B65021EE48B5200ED12BB8990C6136`

Default PPO configuration is gamma 0.995, GAE lambda 0.95, clip 0.20, value coefficient 0.50,
entropy coefficient 0.01, maximum gradient norm 0.50, Adam learning rate 3e-4, two epochs,
32-decision rollout horizon, and configurable CUDA minibatches (initial target 65,536).

The bounded learning and throughput gates explicitly used entropy coefficient 0.0. That is a
recorded PPO configuration choice; it did not modify the frozen observation, action, reward, or
episode contracts. GAE, normalization, shuffling, minibatch gathers, losses, gradients, and
optimizer state remain on CUDA.

## Self-play, checkpoints, and residency

Current-vs-current is the majority path. Historical opponents become eligible after snapshots
exist, are selected only at reset with default 20% probability, remain fixed for the episode,
receive no gradients, and occupy a bounded 16-policy GPU-resident pool. Assignment and policy
sampling RNG state are checkpointed.

Checkpoints contain model and optimizer state, counters, PPO/policy/self-play configuration,
all four contract hashes, CPU/CUDA and dedicated generator states, current assignments, and
historical-opponent metadata and weights. Loading refuses incompatible contract or configuration
hashes by default.

The ordinary path is:

`Warp CUDA state -> zero-copy PyTorch views -> observation -> actor/critic -> action -> four physics ticks -> reward/done -> rollout -> GAE/PPO`

World state, observations, emitted controls, rewards, masks, rollout storage, advantages, returns,
policy inference, and optimization remain on GPU. CPU work is limited to startup/configuration,
explicit diagnostics, human-readable metric snapshots, and checkpoint serialization.

## Boundary

This contract is v0.5 training authority only. It does not implement an RLBot/CPU RocketSim
deployment adapter, Rocket League transfer validation, legacy Rival compatibility, curricula,
other modes, arbitrary body counts, rendering, generic Bullet, or multi-GPU training. v0.6 has
not begun.
