# Reset-aware recurrent execution repair

## Scope and checkpoint preservation

User authorized the repair and asked for comparison against alternatives.
The slow worker was stopped on 2026-09-04 after verifying that no amended
update had been accepted and no rolling checkpoint existed. Only unfinished
update 11 was discarded. The exact update-10 model, Adam state, counters and
RNG remain the continuation source. Original startup file SHA-256:
`FDDDE55887CCD9AD44B62AC672E9C05C078D5146FFC309E2098FAED65E89A37E`.
The original authority and checkpoint were also copied into the run directory's
`performance_repair_original` archive before stopping the worker.

No reward, physics, action/observation contract, model architecture, world count,
opponent distribution, effective PPO minibatch, learning rate, epoch count,
exploration, clipping, GAE lambda, gamma, evaluation protocol, total-update
bound, or wall-clock deadline is changed. Existing KL-as-telemetry-only and
nonfinite/corruption protection remain in force.

## Cause

The original unified GRU executes a whole sequence one tick at a time whenever
any reset occurs. CUDA-to-Python boolean tests synchronize every tick. An
initial reset alone thus caused 360 separate recurrent calls per forward.
Microbatching 182 sequences into groups of 32 amplified this overhead, including
repeated post-step KL and completed-update telemetry forwards.

The previous live nonblocking profile found 483 of 504 captured main-thread
samples in this recurrent loop; 264 were on its reset-boolean synchronization.
That short profile had 83 sample errors and is not a complete GPU-time analysis.
The slow process had not completed its first amended update after 17 minutes.
The prior 128-tick run's first two updates took 53.16 and 51.31 seconds.

## Selected implementation

`RIVAL2_RESET_BOUNDARY_GRU_V1` batches contiguous reset-free spans. Only the
hidden-state rows that actually reset are zeroed at each boundary. Gradients
flow throughout each episode without hidden-state detaches. The one-tick
inference path no longer needs a host boolean synchronization. No packed or
padded copies of episodes or new model parameters are introduced.

The internal chunk cap becomes 182 instead of 32. The existing effective
minibatch remains 182 complete 360-tick sequences (65,520 frames), with the same
smaller last group, masking, normalization, one gradient clip and one Adam step.
This removes unnecessary accumulation for the main groups without changing the
mathematical PPO batch. Floating-point reduction order may differ within the
validated tolerance; bitwise gradient/update identity is not claimed.

## Alternatives and measured results

`exact_scale_comparison.json` used one authoritative 32,768-world, 360-tick
rollout from the preserved checkpoint, then real forward/backward/post-step
telemetry computations with **no optimizer step**. All candidates used identical
observations, actions, advantages, masks, weights, and optimizer state. Three
timings per candidate include full effective-minibatch computation. Full
rollout storage remained resident; no reduced-scale substitute was used.

| Execution | Internal chunk | Median seconds | Peak tensor GiB | Gradient parity |
|---|---:|---:|---:|---|
| Original tick loop | 32 | 2.62935 | 18.804 | reference |
| Reset-boundary spans | 32 | 0.27419 | 18.876 | PASS |
| Packed episodes | 32 | 0.33727 | 18.823 | PASS on real batch |
| Packed episodes, TF32 disabled | 32 | 1.16874 | 18.890 | FAIL |
| Reset-boundary spans | 64 | 0.15696 | 19.285 | PASS |
| Reset-boundary spans | 91 | 0.12681 | 19.694 | PASS |
| **Reset-boundary spans** | **182** | **0.08210** | **20.737** | **PASS** |

The selected approach is approximately 32.0 times faster for this operation,
not a promised whole-update speedup. Real rollout collection took 22.88 seconds
and does not disappear with this repair. The chosen method was fastest among
the tested equivalents and simpler than episode packing. This does not claim
global optimality over every possible implementation or future workload.

Synthetic GRU tests include initial-only, staggered and dense resets. Packing
can be faster for very dense random resets but adds gather/pad/scatter work and
different GPU numerical behavior; small-shape GPU comparisons showed errors
around 1e-4 under the default cuDNN precision policy. Disabling TF32 did not
match the established execution numerics and was slower. It is not selected.
These rejected-candidate observations are not policy-retention failures.

## Correctness and memory

- CPU/GPU reset tests check outputs, final hidden states, input/hidden/parameter
  gradients, Adam updates, no-reset cases, mixed/staggered/dense/every-tick
  resets, single-tick inference, reset at the final tick, and invalid masks.
- Explicit gradient tests prove resets cut only the relevant row's history;
  other rows retain preceding gradients.
- Call-count regression: a reset at tick zero uses one GRU call, not 360.
- Full production-batch selected-path actor/value/hidden outputs matched the
  reference exactly for the inspected 32-sequence output probe. The maximum
  clipped-gradient absolute discrepancy was 1.293e-6 across the full effective
  batch, passing unchanged `atol=3e-6, rtol=1e-4` checks.
- The separate deliberately every-tick-reset memory stress case, which is not
  used for learning, peaked at 20.783 GiB allocated / 20.943 GiB reserved at
  chunk 182. It preserved model and optimizer bytes.
- Both no-step scale reports verify unchanged model/optimizer hashes.
- Focused tests, including adjacent PPO/critic tests: 40 passed before authority
  rebinding; the machine-readable JUnit file is committed here.

## Reproduction

From the repository, use `.venv/Scripts/python.exe`:

```text
-m pytest -q tests/test_rival2_reset_execution.py tests/test_rival2_ssl_foundation_long_trace.py tests/test_rival2_ssl_foundation_amendment.py tests/test_rival2_recurrent_ppo.py tests/test_rival2_unified_policy.py
benchmarks/benchmark_rival2_reset_execution.py --output <report.json>
benchmarks/validate_rival2_reset_execution_scale.py --output <report.json>
benchmarks/validate_rival2_reset_execution_scale.py --stress --repeats 1 --output <stress.json>
```

The no-step scale comparison is deliberately bound to the archived pre-repair
authority. It is not a production-launch bypass. Production independently
requires new committed authority hashes and its exact-scale memory preflight.
See the subsequent production verification artifact for full-update timing;
the component benchmarks above alone do not establish production throughput.

## Production verification

The new authority and launch bindings were committed and pushed before
resuming. The repaired process resumed the same update-10 learning state at
19:47:09 Eastern on 2026-09-04. It accepted and saved:

- Update 11: **82.3701 seconds**, 722 Adam steps, 21.023 GiB peak PyTorch tensor
  allocation / 21.463 GiB reserved.
- Update 12: **91.72 seconds**, 722 Adam steps, approximately 20.84 GiB peak
  PyTorch tensor allocation / 21.47 GiB reserved.

These are real complete rollout-plus-PPO updates, not disposable benchmark
updates. The immutable verification snapshot is
`checkpoints/rival2/ssl_foundation_v5_long_trace_v1/execution_repair_u0012.pt`.
Its exact hash and read-only audit are in `production_verification.json`.
The audit checks model/optimizer finiteness, 1,444 additional Adam steps for
every optimizer state, contiguous accepted boundaries, sample/physics counters,
unchanged incoming learning state, unchanged source checkpoint, and unchanged
deadline. It performs no additional learning. Training continues from the
live rolling checkpoint rather than restarting from the verification copy.

The first two full updates average about 87 seconds. Approximately 38 further
updates to evaluation 50 would therefore take 55 minutes plus evaluation time
at that rate; this is an estimate from two updates, not a promised completion
time. No gameplay-improvement claim follows from this performance repair.
