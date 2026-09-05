# Update 250: no established acquisition improvement; trajectory investigation blocked on CPU

Evaluation completed 2026-09-05 16:17:11 UTC, at 1,474,560,000 accepted trainable
samples. Each deterministic case contains 64 original episodes from the same
frozen seeds, with a maximum of 30 seconds. Results are scenario outcomes,
not full-match win rates. Training was not interrupted.

| Metric | Initial | Update 200 | Update 250 |
| --- | ---: | ---: | ---: |
| Acquisition touch coverage | 12/64 (18.75%) | 9/64 (14.0625%) | 10/64 (15.625%) |
| Acquisition contacts/min | 0.97275 | 0.55856 | 0.62240 |
| Median first touch, touched cases only | 3.500 s | 0.683 s | 0.750 s |
| Acquisition goals scored / conceded | 0 / 0 | 0 / 0 | 0 / 1 |
| Acquisition no-touch truncations | 63/64 | 64/64 | 63/64 |
| Finishing touch coverage | 24/64 (37.5%) | 47/64 (73.4375%) | 43/64 (67.1875%) |
| Finishing native contacts | 67 | 54 | 46 |
| Finishing goals scored / conceded | 3 / 0 | 15 / 0 | 13 / 0 |
| Finishing no-touch truncations | 58/64 | 49/64 | 51/64 |
| Nexto kickoff goals scored / conceded | 0 / 64 | 0 / 64 | 0 / 64 |
| Nexto kickoff Rival contacts | 0 | 0 | 0 |

One more acquisition episode obtains contact than at 200, but coverage and
rate remain below initialization. Finishing coverage and goals fell versus 200,
although remain above initialization. There is no demonstrated broad pursuit,
possession, kickoff competitiveness, or named mechanic competence. The ten-case
conditional first-touch median does not describe the 54 cases without contact.
No-touch truncation means 15 seconds since the last contact, not necessarily
zero contacts during the entire episode. The tiny changes are not evidence
that routine acquisition is developing.

## Focused investigation, not a new training/evaluation authority

Prior native-initial-observation CPU inference showed update 200 requesting
boost in 59/64 acquisition starts and all 64 kickoff starts, without initial
jump or handbrake commands. Live exploratory training telemetry also shows
substantial movement. These observations do not determine subsequent paths.

To inspect the missing temporal evidence without competing for the GPU, added
`benchmarks/inspect_fresh_ground_30hz_trajectories.py`: a read-only, bounded
64-world acquisition diagnostic with the production `FreshGroundEnv` on CPU,
only replacing CUDA stream activation with a CPU-device assertion. Its intended
default is 16 seconds, full per-decision trajectories and descriptive motion
summaries. It neither trains nor modifies the running source package. Its
outputs would be diagnostic CPU recomputation, not GPU parity or promotion
authority. Descriptive speed bins are not rewards or mechanic detectors.

**The diagnostic is BLOCKED and produced no completed trajectory.** The CPU
vehicle kernel cannot compile: generated native C++ in
`rivalsim.kernels.vehicle` uses `fminf` and `fmaxf` without available declarations.
The generated compiler errors were at lines 613, 615 and 3644 in module
`wp_rivalsim.kernels.vehicle_2f213aa`; Warp's retry without precompiled headers
failed too. Python raises `CPU kernel build failed with error code -1` during
the first `world.step(1)`, in `wheel_pre_tick`. The first attempt exposed this
failure; a second captured its structured traceback after adding failure
reporting to the diagnostic. Neither attempt completed a decision.

`cpu_trajectory_diagnostic_failure.json` records the exact invocation, traceback,
script hash, zero optimizer steps and zero Torch CUDA allocation. Python syntax
compilation passed, but **the trajectory script has not passed runtime validation**.
It is retained as the reproducible blocked investigation, not successful evidence.
No simulator kernel, production source, reward, or optimizer was patched to
work around the CPU limitation. No GPU diagnostic was launched. It would be
unsupported to claim the result demonstrates overshooting, circling, or any
particular downstream controller failure.

## Integrity and continuation

CPU checkpoint audit at accepted update 269 passed: 1,586,626,560 accepted
trainable samples, finite model/Adam, real optimizer steps and changed weights,
fresh lineage, exact cadence/LRs, KL telemetry only, numerical rollback intact.
Frozen source/package/authority/preflight and initial checkpoint identities
were verified again after the diagnostic attempts. Worker PID 35748 continued
through update 279 during investigation; campaign stderr remained empty.
The CPU diagnostic failure is not a fault of the active GPU learner.

The routine-acquisition transition criterion remains unmet. Training remains
pure current self-play; Nexto is evaluation-only. No learning settings or
capability semantics were changed. The user authorized training until stopped;
the persistent lack of general gameplay progress and incomplete trajectory
investigation are explicitly reported, not silently treated as success.

Permanent checkpoint: `checkpoints/rival2/fresh_ground_30hz_v1/u000250.pt`.
SHA-256: `FA585B04E3ADD2F062A4BF46A6DFAD9B1C2746668A9D3A07C3CCD3C91FF0DF75`.
Full evaluation: `../evaluations/u000250.json`.
Stable curve/audit: `curve_through_u000269.json`, `u000269.json`.
