# RivalSim v0.1 measured result

## Outcome

**RivalSim v0.1 passes the bounded performance/parity continuation gate.** On the measured
RTX 5090 workstation, the best stable GPU point advanced 131,072 independent two-car,
one-ball contact-free worlds at a median **4,910,323,436 world ticks/s**, or
**40,919,361.97 aggregate simulated game-seconds/s**. The best same-equation vectorized
NumPy CPU point reached 11,125.38 simulated game-seconds/s, making the measured
same-equation GPU/CPU ratio **3,678.02x**.

The 27-scenario parity corpus passed GPU/CPU same-equation checks, live RocketSim checks,
and hard axis/sign/state-timing checks at 1, 4, 8, 30, 60 and 120 ticks.

This result supports recommending a separately authorized v0.2 arena/ground-contact proof.
No v0.2 mechanic was implemented in this run.

Canonical machine-readable evidence:

- [`results/v0.1/benchmark.json`](../results/v0.1/benchmark.json)
- [`results/v0.1/parity.json`](../results/v0.1/parity.json)
- [`results/v0.1/manifest.json`](../results/v0.1/manifest.json)

## What v0.1 implements

Each world has exactly two cars and one free ball. State is flattened into device arrays;
there is no per-world Python object graph. The timed hot loop keeps state and controls on
the GPU and uses a fused Warp tick kernel with one thread per car. The even-indexed car
thread also advances the ball for its world.

The fixed 120 Hz contact-free mechanics are:

- gravity and Bullet-order semi-implicit translation;
- normalized exponential-map quaternion integration from world angular velocity;
- car and ball linear/angular speed caps;
- source-backed air throttle, boost acceleration, minimum boost time, consumption and
  depletion;
- first-jump edge/roof impulse, three sticky ticks, RocketSim's pre-minimum jump scale,
  held-jump acceleration through 0.2 s and legal double-jump timeout;
- source-backed dodge direction, impulse, relative torque, vertical damping, flip timer and
  pitch-lock behavior;
- source-backed airborne pitch/yaw/roll torque and damping;
- free-ball gravity, drag, translation, rotation and speed caps;
- previous controls, boost/jump/double-jump/flip state and timers, and supersonic state.

A vectorized FP32 NumPy simulator implements the same v0.1 equations for an apples-to-apples
CPU baseline. `rocketsim==2.2.1` in `GameMode.THE_VOID` supplies the live contact-free oracle.

The deliberately excluded v0.2+ mechanics remain arena/Bullet collision, suspension,
wheels and ground driving, ball-world/car-ball/car-car contact, boost pads, goals, demos,
RLGym integration, observations, rewards, PPO and rendering.

## Source custody and environment

The measurement used:

- Windows 11 `10.0.26200`, x86-64;
- AMD Ryzen 7 9800X3D, 8 physical/16 logical cores;
- Python `3.14.3` (`MSC v.1944`, 64-bit);
- NumPy `2.5.2`;
- NVIDIA Warp `1.16.0`, whose wheel reports bundled CUDA Toolkit `12.9`;
- NVIDIA GeForce RTX 5090, `sm_120`, 170 SMs, 34,190,917,632 bytes VRAM;
- NVIDIA driver `610.62`, CUDA driver API `13.3`;
- installed CUDA compiler toolkit `13.3`, `nvcc V13.3.73`;
- primary `ZealanL/RocketSim` source commit
  `c2baacb8f4b441dd8505e63c2aeb5a1679b60b02`;
- `mtheall/RocketSim` binding commit
  `2da51b1dac7b8127127613a5ff30e490bdd70dd8`, package `rocketsim==2.2.1`;
- installed `RocketSim.pyd` SHA-256
  `E3EE24CA82445B4BFCC754583F6778D7B0D8B7A7F7D64F872BE8C65E621A63D0`.

Python 3.12 was preferred by the handoff. Only Python 3.14.3 was registered on this
workstation, and the selected Warp/RocketSim wheels installed and passed the complete suite
under 3.14, so no global Python or CUDA installation was changed.

See [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) for license notices and
[`docs/REPRODUCING_V0_1.md`](REPRODUCING_V0_1.md) for exact setup and execution commands.

## Parity protocol

The corpus was first run in measurement-only mode with both tolerance dictionaries empty.
Two oracle-adapter defects were fixed before tolerance selection: the Python binding returns
the car basis in rows, requiring a transpose for the expected orientation convention, and
`ArenaConfig.no_ball_rot` defaults to true, requiring an explicit false setting. The corrected
raw measurement was then used to freeze horizon-specific tolerances. The committed gate run
was performed only after that freeze.

The 27 deterministic cases cover free gravity/translation/rotation and caps; boost from rest,
nonzero speed, depletion and throttle combinations; tap/minimum/full held jump trajectories,
legal double jumps at 0.05/0.40/1.00 s, timeout and tilted-roof impulse; isolated and combined
air torque, forward/diagonal dodge state and angular cap; and free-ball drag/rotation/caps.

### Aggregate live RocketSim error and tolerance by horizon

Each cell is `maximum measured error / frozen tolerance`. Units are unreal units (uu),
uu/s or radians as labeled.

| Ticks | Car pos (uu) | Car vel (uu/s) | Orientation (rad) | Angular vel (rad/s) | Ball pos (uu) | Ball vel (uu/s) | Ball orientation (rad) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 6.104e-5 / 2e-4 | 2.460e-4 / 5e-4 | 1.119e-4 / 5e-4 | 7.451e-9 / 1e-6 | 2.441e-4 / 5e-4 | 0 / 5e-4 | 0 / 5e-4 |
| 4 | 1.221e-4 / 3e-4 | 2.461e-4 / 5e-4 | 3.287e-4 / 7e-4 | 4.768e-7 / 2e-6 | 2.441e-4 / 5e-4 | 1.258e-4 / 5e-4 | 0 / 5e-4 |
| 8 | 1.223e-4 / 3e-4 | 2.461e-4 / 5e-4 | 3.295e-4 / 7e-4 | 5.340e-7 / 2e-6 | 1.221e-4 / 5e-4 | 1.526e-4 / 5e-4 | 2.289e-4 / 5e-4 |
| 30 | 2.446e-4 / 7e-4 | 8.548e-4 / 2e-3 | 2.169e-4 / 7e-4 | 7.907e-7 / 5e-6 | 6.104e-4 / 1e-3 | 3.766e-4 / 1e-3 | 0 / 5e-4 |
| 60 | 8.028e-4 / 2e-3 | 1.590e-3 / 3e-3 | 2.494e-4 / 7e-4 | 4.061e-6 / 1e-5 | 6.574e-4 / 2e-3 | 4.529e-4 / 1e-3 | 0 / 5e-4 |
| 120 | 1.357e-3 / 3e-3 | 6.372e-4 / 2e-3 | 1.291e-4 / 7e-4 | 3.068e-6 / 1e-5 | 1.386e-3 / 3e-3 | 6.488e-4 / 2e-3 | 1.972e-4 / 5e-4 |

Boost and exposed timers/flip torque had zero aggregate error. Every discrete jump/flip state
matched exactly. Ball angular-velocity error was zero. GPU/CPU same-equation maxima at 120
ticks were 1.221e-4 uu position, 3.052e-5 uu/s velocity, 0 rad orientation and 3.053e-6
rad/s angular velocity, with exact discrete-state agreement.

### Worst live RocketSim tolerance cell by scenario

This is a compact scenario-by-scenario view. It shows the largest `error / tolerance` ratio
among all continuous metrics and all six horizons for each scenario. All other cells were no
worse; all discrete state/sign checks were exact. The complete metric-by-scenario-by-horizon
matrix is preserved in `parity.json`.

| Scenario | Worst tick | Metric | Error / tolerance | Fraction used |
| --- | ---: | --- | ---: | ---: |
| `stationary_gravity_drop` | 1 | position | 6.104e-5 / 2e-4 | 30.5% |
| `arbitrary_linear_velocity` | 120 | position | 1.162e-3 / 3e-3 | 38.7% |
| `arbitrary_angular_velocity` | 8 | orientation | 3.295e-4 / 7e-4 | 47.1% |
| `car_speed_limit_crossing` | 4 | linear velocity | 2.461e-4 / 5e-4 | 49.2% |
| `combined_rotation_integration` | 8 | orientation | 2.456e-4 / 7e-4 | 35.1% |
| `boost_from_rest` | 1 | position | 6.104e-5 / 2e-4 | 30.5% |
| `boost_nonzero_velocity` | 60 | linear velocity | 1.590e-3 / 3e-3 | 53.0% |
| `boost_depletion` | 1 | position | 6.104e-5 / 2e-4 | 30.5% |
| `forward_air_throttle` | 1 | position | 6.104e-5 / 2e-4 | 30.5% |
| `reverse_air_throttle` | 1 | position | 6.104e-5 / 2e-4 | 30.5% |
| `boost_throttle_combined` | 1 | position | 6.104e-5 / 2e-4 | 30.5% |
| `tap_jump_airborne_phase` | 4 | position | 1.221e-4 / 3e-4 | 40.7% |
| `minimum_three_tick_jump` | 4 | position | 6.104e-5 / 3e-4 | 20.3% |
| `full_point_two_second_jump` | 8 | position | 1.221e-4 / 3e-4 | 40.7% |
| `double_jump_delay_0.05` | 8 | position | 6.104e-5 / 3e-4 | 20.3% |
| `double_jump_delay_0.40` | 8 | position | 6.104e-5 / 3e-4 | 20.3% |
| `double_jump_delay_1.00` | 8 | position | 6.104e-5 / 3e-4 | 20.3% |
| `double_jump_after_timeout` | 1 | position | 6.104e-5 / 2e-4 | 30.5% |
| `tilted_roof_double_jump` | 8 | position | 6.110e-5 / 3e-4 | 20.4% |
| `isolated_pitch` | 8 | orientation | 2.690e-4 / 7e-4 | 38.4% |
| `isolated_yaw` | 4 | orientation | 3.122e-4 / 7e-4 | 44.6% |
| `isolated_roll` | 1 | position | 6.104e-5 / 2e-4 | 30.5% |
| `combined_air_torque` | 1 | position | 6.104e-5 / 2e-4 | 30.5% |
| `forward_dodge` | 4 | orientation | 3.287e-4 / 7e-4 | 47.0% |
| `diagonal_dodge_flip_timer` | 120 | position | 1.357e-3 / 3e-3 | 45.2% |
| `angular_speed_limit` | 1 | position | 6.104e-5 / 2e-4 | 30.5% |
| `free_ball_drag_rotation_limits` | 30 | ball position | 6.104e-4 / 1e-3 | 61.0% |

### Oracle limitations

- Live first-jump ground contact and wheel sticky force cannot be created in contact-free
  `THE_VOID`; those equations have source-backed unit checks, while live jump trajectories
  start immediately after the first-jump impulse.
- The Python binding does not expose `time_since_boosted` or `is_boosting`; those fields use
  same-equation CPU/GPU checks only.
- `THE_VOID` sleeps an exactly motionless ball. Non-ball cases seed it at 0.001 uu/s so the
  unrelated ball state does not generate a false mismatch.

## Benchmark protocol

The published GPU sweep ran 6,000 ticks (50 simulated seconds/world) per repeat, five repeats,
120 untimed warmup ticks, proper synchronization and deterministic seed `20260822`. Controls
and state remained resident. Each timed launch replayed a captured eight-tick CUDA graph;
this choice is part of the reported protocol because direct per-tick Warp launch timing on
Windows/WDDM showed host-scheduling jitter at underfilled batches. Verification readback was
outside timing. NVML/CPU telemetry was collected in a separate untimed pass to avoid
perturbing the short measured loop.

The CPU sweep ran 360 ticks (three simulated seconds/world) per repeat and three repeats.
Both implementations used the same FP32 state, fixed controls and equations. A point is
stable when its repeated aggregate-throughput coefficient of variation is at most 5% and it
has no NaN/error. All published points passed.

The required GPU sweep continued by powers of two while the median world-tick gain was at
least 5%. Throughput peaked at 131,072 worlds and dropped at 262,144, so the adaptive sweep
stopped there.

### GPU sweep

| Worlds | World ticks/s median | Aggregate sim-s/s median | CV | GPU util mean/max | Device VRAM peak (bytes) | Logical state (bytes) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 53,910,815 | 449,256.79 | 1.466% | 13/13% | 4,168,695,808 | 97,280 |
| 512 | 105,880,285 | 882,335.71 | 1.377% | 13/13% | 4,168,826,880 | 194,560 |
| 1,024 | 209,531,215 | 1,746,093.46 | 1.289% | 90/90% | 4,168,630,272 | 389,120 |
| 2,048 | 416,984,800 | 3,474,873.33 | 0.970% | 90/90% | 4,167,188,480 | 778,240 |
| 4,096 | 805,213,442 | 6,710,112.01 | 1.703% | 88/88% | 4,166,991,872 | 1,556,480 |
| 8,192 | 1,614,876,625 | 13,457,305.21 | 0.735% | 88/88% | 4,166,991,872 | 3,112,960 |
| 16,384 | 2,774,746,597 | 23,122,888.31 | 2.366% | 67/67% | 4,166,991,872 | 6,225,920 |
| 32,768 | 3,570,412,907 | 29,753,440.89 | 1.195% | 82/82% | 4,158,865,408 | 12,451,840 |
| 65,536 | 4,416,187,387 | 36,801,561.56 | 0.672% | 84/84% | 4,187,701,248 | 24,903,680 |
| **131,072** | **4,910,323,436** | **40,919,361.97** | **0.440%** | **89.33/98%** | **4,220,076,032** | **49,807,360** |
| 262,144 | 1,661,021,907 | 13,841,849.23 | 1.920% | 95.16/99% | 4,296,163,328 | 99,614,720 |

Device VRAM is whole-device NVML use and includes roughly 4 GiB already used by the Windows
desktop and other system activity. At the best point, Warp's current mempool use was
58,195,968 bytes (108,003,328-byte high-water mark), and controls used another 8,388,608
logical bytes. Timed hot-loop H2D and D2H traffic counters were both zero at every point.
Small-batch utilization samples are informational because their separate telemetry pass is
shorter than NVML's sampling cadence.

### Same-equation CPU sweep

| Worlds | World ticks/s median | Aggregate sim-s/s median | CV |
| ---: | ---: | ---: | ---: |
| 256 | 475,890 | 3,965.75 | 1.453% |
| 512 | 736,993 | 6,141.60 | 1.433% |
| 1,024 | 998,966 | 8,324.71 | 0.682% |
| 2,048 | 1,185,928 | 9,882.74 | 0.453% |
| **4,096** | **1,335,045** | **11,125.38** | **0.226%** |
| 8,192 | 1,194,305 | 9,952.54 | 3.307% |
| 16,384 | 1,316,911 | 10,974.26 | 0.874% |

## Gate evaluation

| Condition | Measured evidence | Result |
| --- | --- | --- |
| GPU at least 4x same-equation CPU | 3,678.02x | Pass |
| GPU at least 2,000 aggregate sim-s/s | 40,919,361.97 | Pass |
| Stable scaling into thousands | All required points stable; scaling continued through 131,072 | Pass |
| GPU-resident hot loop | 0 timed H2D and 0 timed D2H bytes | Pass |
| No NaNs/errors | None at any published point; 2,400-tick random stress also passed | Pass |
| Basic RocketSim parity | 27/27 scenarios, six horizons, axis/sign/state checks | Pass |

The ratio to Rival's existing full RocketSim/RLGym reference of 200.65 aggregate sim-s/s is
203,934.02x. **That is explicitly not an apples-to-apples speedup**: the v0.1 kernel omits
arena collision, suspension, dynamic contacts, game logic and the environment/training stack.
It is useful only as evidence that the bounded kernel has substantial headroom before those
costs are added.

## Validation and boundary

The final release validation recorded 20 passing tests, a clean Ruff run, successful
`compileall`, successful evidence JSON parsing and a clean `git diff --check`. `pip check`
reports `rocketsim 2.2.1 is not supported on this platform`: the wheel's downloadable name is
correctly tagged `cp36-abi3-win_amd64`, but its internal `WHEEL` metadata incorrectly says
`cp311-cp311-win_amd64`. The extension imports under Python 3.14 and every live RocketSim test
passed; the installed metadata was not patched. This known upstream packaging warning is
preserved in the release manifest rather than represented as a successful `pip check`.

The smallest next step is review of this v0.1 evidence. If accepted under separate authority,
v0.2 should begin with an extracted stadium collision mesh and GPU-resident acceleration
structure, then add wheel/suspension and car-world contact while preserving these v0.1
artifacts unchanged. That work was not begun here.
