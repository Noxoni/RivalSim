# RivalSim v0.5 / Rival 2.0 Results

Status: **PASS_GREEN**

RivalSim v0.5 completes the authorized Rival 2.0 GPU-native training milestone. It adds a
clean-slate observation and hybrid native-controller contract, zero-copy Warp/PyTorch state
views, four-tick training transitions, GPU rewards and episode accounting, GPU rollout/GAE/PPO,
exact checkpoint resume, current-policy self-play, and bounded GPU historical opponents. All
blocking v0.5 and inherited simulator gates pass. v0.6 deployment/transfer work was not begun.

## Frozen contracts

| Contract | Result | SHA-256 |
|---|---|---|
| `RIVAL2_OBS_V1` | 182 float32 values, symmetric agent perspectives | `10669E7D240D553BEA601F8AD7AEF9F9029310E55EA6DC4072E866F34BB218AF` |
| `RIVAL2_ACTION_V1` | 5 tanh-Gaussian analog + 3 Bernoulli buttons | `145AF5B49E1A0C85209022A6DE862F05EF996EB89B90B124072A59BC9936727B` |
| `RIVAL2_REWARD_V1` | goal/progress/touch/demo, exactly zero-sum | `E3C97C7B3EA97D15F6AFB3AF21C40BAFBD206F0ED1124BAD6EA2C5A2ED14786F` |
| `RIVAL2_EPISODE_V1` | goal termination; 15 s/45 s truncation | `E22B6014C6D975D700D1129B9F554D6F35E4CA5003F1C1BD09C7D394D4F9347E` |

The exact indices, normalization constants, orange rotation/pad remap, action math, reward terms,
episode semantics, model, PPO defaults, and checkpoint state are frozen in
`docs/RIVAL2_TRAINING_CONTRACT.md`.

## Correctness and residency

- Forty-eight persistent Warp arrays were exposed as contiguous PyTorch CUDA views. Every sampled
  Warp/PyTorch pointer pair was identical. The normal rollout path recorded zero NumPy calls,
  zero `.cpu()` calls, no host object packing, and zero timed H2D/D2H state traffic.
- The deterministic observation corpus covers kickoff, ordinary driving, aerial, wall, ball,
  boost-pad, demolition/respawn, goal, and reset states. Its output SHA-256 is
  `45EA7B9A8B9323C4728B6A88D94EF9C6D742B0CB05C54504219382FF3275F893`.
  Repeat error is exactly zero and blue/orange perspective-symmetry maximum error is
  `4.172325e-7`.
- The hybrid action log probability matched an independent float64 CPU oracle to
  `2.635653e-5` maximum absolute error under the frozen `3e-5` float32 tolerance. Same generator
  state reproduced actions exactly, buttons were exactly binary, entropy remained finite, and no
  action table participated.
- Mechanics4 held each emitted control tensor for exactly four 120-Hz ticks across ground,
  jump press/release, boost, powerslide, simultaneous rotation, goal/reset, and
  demolition/respawn boundaries. No inference occurs inside the environment's four-tick step.
- Goal, progress, touch-entry latch, demolition, no-touch timeout, hard timeout, final pre-reset
  observation, and selective reset checks passed with exact zero-sum reward. Persistent contact
  produced zero new touch events on the following interval.
- A repeated deterministic `[3,4,2]` rollout produced identical SHA-256
  `9FCA719E0C4EEF000DD2F2F7A867E57F9A5244368C9BC216163D5F0597532DFF`.
  All storage was CUDA-resident, selective reset refreshed the next row, indexing was exact, and
  no autograd graph remained attached.
- GPU GAE matched an independent mixed terminal/truncation reference to `9.536743e-7` maximum
  error for both advantages and returns.
- The independent 257-sample float64 PPO oracle covered ratio, clipped surrogate, value loss,
  entropy, total loss, approximate KL, and clip fraction. Maximum error was `2.072807e-5`, driven
  by hybrid log probability; scalar objective errors were below `3.18e-7`.
- PPO produced finite metrics and optimizer state, changed actor and critic parameters, and
  demonstrated configured gradient clipping (observed pre-clip norm about 4.27 and post-clip norm
  0.50 in the stress gate). Rollout buffers retained no training graph.
- Checkpoint reload reproduced weights, optimizer tensors, counters, contract identities,
  deterministic action/value, the next stochastic sample, and historical metadata exactly.
  A modified contract hash was refused.
- Historical selection changed only at reset, remained fixed during the episode, ran on CUDA,
  and applied no gradients to the frozen opponent. Defaults remain 20% historical eligibility and
  a maximum 16-policy resident pool.

## Learning sanity

The official fixed-seed gate declared the held-out clipped hybrid PPO objective before its one
optimizer update. It used 8,192 training worlds (seed 111), a separate 8,192-world held-out set
(seed 222), two warm-up rollouts with no update, horizon 32, and one update. The recorded PPO
entropy coefficient was 0.0; all frozen environment contracts remained unchanged.

| Metric | Result |
|---|---:|
| objective before | `3.477908e-9` |
| objective after | `5.304051e-4` |
| improvement | `5.304016e-4` |
| improvement standard errors | `4.226` (required `>=3`) |
| actor maximum parameter delta | `0.00413393` |
| critic maximum parameter delta | `0.00201720` |
| critic output variance after | `6.722311e-7` |

The bounded gate therefore proves a non-no-op integrated learning update, not bot skill or
external transfer.

Two earlier development metrics are preserved as negative results. Fixed stochastic blue return
fell from `0.0267631` to `-0.0148839` in a 512-world/50-update trial and from `0.00270689` to
`-0.0164957` in a 2,048-world/20-update trial. Those metrics failed and are not presented as
success. The reward contract was not changed after seeing them. The official gate instead uses a
prospectively declared held-out PPO objective and a conventional three-standard-error threshold.

The action reference tolerance was also frozen at `3e-5` after the first formal `3e-6` candidate
proved below ordinary float32 accumulated error for saturated tanh samples. This is a numerical
oracle tolerance, not a simulator/action tolerance or behavior change.

## End-to-end performance

Hardware/software: Windows 11, Python 3.14.3, NumPy 2.5.2, Warp 1.16.0, PyTorch 2.13.0+cu130,
NVIDIA driver 610.62, and one RTX 5090 with 32 GiB. Workload is complete rollout generation,
GAE, and PPO with two epochs, horizon 4 for the sweep, minibatch 65,536, and entropy coefficient
0.0. Every point used five warmed repeats.

| Worlds | End-to-end agent samples/s | Rollout decisions/s | PPO samples/s | Sim game-s/s | Wall CV | Peak VRAM bytes |
|---:|---:|---:|---:|---:|---:|---:|
| 8,192 | 815,487 | 1,072,402 | 3,658,547 | 17,873 | 2.319% | 9,425,170,432 |
| 16,384 | 1,369,044 | 2,144,307 | 3,860,942 | 35,738 | 4.299% | 9,787,781,120 |
| 32,768 | 1,926,774 | 3,880,425 | 3,912,674 | 64,674 | 1.538% | 10,261,016,576 |
| 65,536 | 2,055,810 | 4,569,062 | 3,772,083 | 76,151 | 1.286% | 11,451,609,088 |
| 131,072 | **2,233,902** | **5,370,347** | **3,838,251** | **89,506** | **0.588%** | **14,414,032,896** |

The selected 131,072-world point has median wall time 0.469392 seconds: rollout 0.195253 s,
GAE 0.000688 s, and PPO 0.273191 s. Median actor/critic inference is 9.027 ms. The profiled
four-tick environment interval records 18.889 ms for physics/reward/action copy, 4.806 ms for
transition observation/output capture, 0.006 ms for selective reset, and 4.213 ms for post-reset
observation. Policy-version/sample-age lag is zero. Timed hot-loop H2D and D2H are both zero.

## Inherited simulator gates

- v0.4 native lifecycle authority `33AA0BA3BC35BC4300E2D2B84A3813CB0AD776479546A50AC3BBC6CE3D3E2562`:
  all pad, goal/kickoff, demolition/respawn, and 64-world/400-tick deterministic lifecycle gates
  pass; stress traffic remains zero.
- Both 4,608-ray backends pass exact hit/distance/normal gates. The inherited complete v0.4 path
  reaches 196,692.61 sim-s/s at 131,072 worlds with 2.207% CV; reset-heavy reaches 213,562.09
  sim-s/s and 3,203,431 resets/s.
- v0.3: Phase A 31,216/31,216; Phase B 8,192/8,192; Phase C 8,192/8,192 for both complete
  visitation branches; Phase D 512/512 for both branches. All required horizons have zero failed
  cases.
- v0.2.2: 39,236/39,236 with zero hard mismatches, zero numeric failures, and zero failed cases.
- v0.1: all 27 live RocketSim scenarios pass.
- Repository tests, Ruff, compileall, and diff checks pass; the final test count is recorded in
  `results/v0.5/regression.json`.
- Published `results/v0.1/` through `results/v0.4/` remain byte-for-byte unchanged.

## Boundary

v0.5 does not establish RocketSim/RLBot/Rocket League transfer, learned gameplay quality, legacy
Rival compatibility, or production readiness. No v0.6 deployment adapter or transfer evaluation
was implemented. The project stops at the completed v0.5 boundary.
