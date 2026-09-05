# Further throughput investigation (diagnosis only)

The live training worker was sampled without suspension or restart for 120
seconds at 50 Hz. No production code, weights, optimizer state, authority,
rewards, training settings, or scheduled evaluations were changed. Completed
updates 33-36 took 83.89, 85.90, 86.62, and 87.08 seconds. The original large
reset-loop bottleneck remains fixed.

## What the profile supports

There were 5,728 reported samples, 271 sampling errors, and 1,026 empty stacks.
Of the 4,702 identifiable stacks, 40.54% were in rollout collection, 29.71% in
the learning minibatch path excluding its post-step KL pass, 15.82% in that
post-step KL pass, and 13.50% in completed-update diagnostics. Saving checkpoints
was only 0.11%. This is not GPU kernel-time attribution: a Python synchronization
site can wait for earlier asynchronous GPU work. The window spans partial
update boundaries, so these percentages are not exact per-update timing shares
or achievable speedup percentages.

Raw profile SHA-256:
`D9ACF55D0F17AFE7DFFD6E19A28C74D6B0F10C67F3BB7A6DFAD9CFD8DA124DC6`.

## Recommended order

1. **Avoid computing outputs that are discarded.**
   `rival2_ssl_foundation_training.py:225` invokes the full recurrent actor and
   independent critic for next-state values, then discards actor and hidden
   outputs. `IndependentCriticActorCritic.isolated_value` already calculates
   the required value without actor work. Conversely, the post-step KL path
   in `rival2_recurrent_ppo.py:393` invokes the full model and discards the
   independent multilayer critic result. Add a separate actor-only inference
   entry point for that use, preserving all KL samples and calculations.
   This is the smallest, clearest scope for a first optimization patch.

2. **Reuse reset/mask metadata and consolidate host checks.**
   `gru_reset_spans` transfers reset-boundary metadata for both gradient and
   post-step forwards, although the masks did not change. The profile includes
   664 samples at that metadata-transfer line. `_finite_parameters` checks
   24 parameter tensors individually at each of 722 steps: up to 17,328 host
   checks per update. Reducing them to one combined host decision per boundary
   must still check every tensor and retain the same nonfinite stop/rollback.
   Do not assume the full observed wait time is removable: it includes GPU work.

3. **Restrict frozen-V5 inference to its actual opponent rows.**
   The frozen model runs on all 65,536 agent rows every tick. With 20% frozen
   opponent worlds and one frozen agent per such world, only about 10% of
   these rows are used under the configured mix. This can remove substantial
   frozen-model work, not 90% of the complete trainer's work. Correct handling
   of episode reassignment, hidden resets, current-policy random draws, and
   batched numerical parity is required before deployment.

4. **Treat fused Adam as secondary, not the primary fix.**
   The live profile already shows `_multi_tensor_adam`, with 53 inclusive
   samples. Checkpoint flags are `foreach=None`, `fused=None`; this is not the
   slow single-parameter loop. A fused candidate may help, but Adam moment and
   update parity must be checked before changing this established path.

The independent critic remains independently trained. No critic gradients are
removed by the proposed inference-only separation. Do not reduce worlds,
sequence length, trainable frames, epochs, or telemetry to claim a throughput
gain. Do not remove nonfinite checks, change precision, or shorten the longer
credit-assignment trace.

## Bounded probe and its limits

A CPU-only probe loaded the immutable update-20 checkpoint, evaluated synthetic
`[8,32,182]` inputs with staggered resets, and compared full-forward values with
the existing independent-critic-only call. Values were bit-identical (maximum
absolute error zero); hooks confirmed the critic-only call executes no actor
trunk/GRU. The checkpoint hash was unchanged. No optimizer was created or
stepped. RivalSim import enumerates Warp devices, but this probe performed no
CUDA tensor computation. See `cpu_value_probe.json`.

This proves a narrow structural equivalence, not full-scale CUDA equivalence or
performance. A subsequent implementation should compare old/new paths on the
same real rollout and preserved checkpoint, with no optimizer step during
timing; check values, actor outputs, hidden trajectories, masks, targets,
gradients, RNG, memory and required diagnostics. Then verify complete accepted
update time. No speedup factor is promised by this investigation.

## References and reproduction

PyTorch documents `.item()`, `.nonzero()` and similar operations as potential
[CPU/GPU synchronization points](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html).
Its [Adam documentation](https://docs.pytorch.org/docs/main/generated/torch.optim.Adam.html)
distinguishes foreach and fused execution. The
[py-spy documentation](https://github.com/benfred/py-spy) explains nonblocking
sampling and its consistency limitations. These support the mechanism, not
specific performance promises for this machine.

The live profile command was:

```powershell
& 'G:/dev/RivalSim-runs/diagnostics-long-trace-20260904/tools/bin/py-spy.exe' record --pid 44708 --nonblocking --rate 50 --duration 120 --format raw --output 'G:/dev/RivalSim-runs/diagnostics-long-trace-speed-20260904/live.raw'
```

The recorded PID is historical: resolve the current worker before repeating.
`summary.json` contains exact attribution counts and all stated limitations.
