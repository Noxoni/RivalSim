# V5 restart: longer retrospective credit assignment

User-authorized amendment, not a new random initialization or a new reward contract.
The preceding V5 restart stopped cleanly at local update **10**, with 58,831,798
trainable agent samples. Its immutable `final.pt` is the sole transition parent:
`EFDB7725D8DBFAFBFA75A66F4AAF618AC561DE6BDC660B0976B6AFDAE775927B`.
Original Unified V5 remains the root (`955C93BF...B7C216`).

## Exactly two PPO changes

| Setting | Before | Amended |
|---|---:|---:|
| Rollout/recurrent training length | 128 ticks / 1.0667 seconds | 360 ticks / 3 seconds |
| GAE lambda | 0.9872585449014338 | 0.9993279525461148 |
| Actual PPO and potential-shaping gamma | 0.9987476493904754 | unchanged |

Lambda is derived prospectively, not tuned against evaluation:
`lambda = 2^(-1/360) / gamma`. The direct TD-error trace `(gamma*lambda)^k`
therefore has a **three-second half-life**. At one/two/three seconds its weight
is 0.793700526 / 0.629960525 / 0.5, versus about 0.184672 / 0.034104 / 0.006298
previously. These are TD-error weights, not guaranteed causal blame or the entire
influence of a goal. The critic also propagates consequences. A 360-tick fragment
contains at most a 359-step separation; the exact 360-step impulse is unit-tested
on 361 samples. Rollout boundaries still truncate traces. True terminals stop
bootstrapping; administrative truncations bootstrap final-state value and stop
cross-episode traces. The network does not predict a three-second future.

Longer traces reduce reliance on bootstrap estimates but can increase variance.
There is no automatic claim of better gameplay. No direct action penalties are added.

## Unchanged

All model tensors (actor, recurrence, independent critic), Adam moments/steps,
RNG state and cumulative training counters are preserved by the transition.
The runtime resumes with fresh simulator episodes and zero recurrent state under
the existing checkpoint-resume semantics, not exact live-physics replay.

Keep 32,768 worlds, 120 Hz, policy LR 1e-4, critic LR 3e-4, two epochs, clipping
0.2, max gradient norm 0.5, entropy zero, family-local advantage normalization,
critic gradient isolation, KL telemetry-only and nonfinite rollback. Keep mature
exploration (sigma 0.04, button temperature 0.25), 40/40/20 current/Nexto/frozen-V5
episode assignments, physical reset corpus, action/observation contracts, and all
six state potentials with terminal +/-10. Gamma, potential weights and all reward
code remain unchanged. No named-mechanic rewards or action classifications.

The historical base PPO contract is not edited: the amendment explicitly binds
its effective config/hash rather than claiming the old horizon/lambda hash applies.

## Cost, boundaries, and evidence

Keep the existing total local **100-update bound**, not 100 additional updates.
Reuse the just-finished deterministic update-10 evaluation as the exact transition
baseline; evaluate/snapshot at total updates 50 and 100 using the same 3600-tick,
1024-world protocol. Preserve original update-0 evaluation separately.
Report actual cumulative trainable samples and elapsed time: one amended update
collects 2.8125 times as many physics ticks as one old update. Training is not being
compared fairly by update count alone. This is a sequential amendment, not a
randomized causal A/B comparison. Missing backward-driving/possession telemetry
must not be inferred from goalward-touch counts or training losses.

The 65,536-tick minibatch cap yields 182 complete 360-tick sequences = 65,520 ticks,
with the normal smaller final minibatch. No sequence/frame shuffling change.
Raw rollout storage is about 17.99 GiB. The shared runner now releases the completed
rollout before collecting another, preventing accidental double residency without
changing samples or gradients. Exact-scale preflight performs a real 360-tick
rollout and full recurrent-minibatch backward, but **no optimizer step**; first real
accepted updates supply whole-update peak memory, timing and safety evidence.

## Commands

```powershell
.\.venv\Scripts\python.exe benchmarks/run_rival2_ssl_foundation_v5_long_trace_v1.py --prepare-amendment --implementation-commit <commit>
.\.venv\Scripts\python.exe benchmarks/run_rival2_ssl_foundation_v5_long_trace_v1.py --memory-preflight-only
.\.venv\Scripts\python.exe benchmarks/run_rival2_ssl_foundation_v5_long_trace_v1.py
```

Operational resume only:
`--resume G:/dev/RivalSim-runs/ssl-foundation-v5-long-trace-v1/rolling.pt`.
Never resume the superseded 128-tick run or remove its intentional STOP_REQUESTED.
Authorities/transition and memory preflight must be committed and pushed before
accepted learning. The original wall-clock deadline and total-update bound remain.
