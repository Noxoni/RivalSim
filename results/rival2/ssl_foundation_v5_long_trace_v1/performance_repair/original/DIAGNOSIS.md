# Long-trace PPO throughput investigation

Inspected 2026-09-04 19:33 Eastern. Repository HEAD:
`9644bc7c50bdcaa9effb0d1a1b0684a895bb0c31`.

## Scope

Read-only inspection of training worker PID 28504; no production code, policy,
reward, optimizer, rollout/trace configuration, or running process was changed
by this investigation. An isolated py-spy 0.4.2 installation was used outside
the repository and active Python environment. No additional training was launched.

## Measured evidence

- Worker launch: 19:16:23 Eastern. At 19:33:46, no first amended update had
  completed; last accepted checkpoint remained local update 10.
- First two previous 128-tick updates took 53.1631 and 51.3145 seconds.
- GPU snapshot: 31% utilization, 29,637/32,607 MiB physical VRAM usage.
- `profile_1934.json`: nonblocking 20-second Python sampling profile at 50 Hz,
  captured approximately 19:32:53-19:33:13 Eastern. The filename is merely a
  label, not its actual acquisition timestamp.
- 916 total thread samples, including 504 nonempty main-thread samples and
  412 empty background-thread samples. 83 sample errors occurred, consistent
  with the limitations of nonblocking sampling; proportions are approximate,
  not a GPU kernel-time measurement or a whole-update timing decomposition.
- 483/504 main-thread samples (95.8%) included `_context_with_resets`.
- 264/504 (52.4%) were at `bool(torch.any(reset))`, line 184.
- 267/504 were inside the training forward and 221/504 inside post-step
  telemetry forward. All captured main-thread samples were in PPO optimization,
  not simulator rollout/Nexto inference.

## Cause

`rivalsim/rival2_unified_policy.py:170-188` takes the slow path for an entire
batch sequence whenever any reset is present. That path calls the GRU once per
physics tick and converts a CUDA reset reduction to a Python boolean each tick,
introducing host-device synchronization. Even a reset only at tick zero causes
all 360 ticks to execute separately. Resume initializes reset_before to true
for every world-agent sequence (`rival2_recurrent_training.py:490`), so this
condition applies to the first amended rollout.

The memory correction splits each effective 182-sequence minibatch into six
microbatches (five of 32, one of 22). Post-step KL telemetry repeats those
sequence forwards. Completed-update diagnostics use 32-sequence chunks too.

Code-derived first-update execution count, assuming nonempty train masks:

- 65,536 world-agent sequences; two epochs.
- 360 full effective groups of 182 plus a last group of 16 per epoch.
- 4,322 training forwards, 4,322 post-step forwards, 2,048 completed-update
  diagnostic forwards: 10,692 sequence forwards.
- With 360 individual GRU calls per sequence forward: 3,849,120 one-tick GRU
  forward calls, excluding rollout and backward work.
- Old first-update path: 256 training + 256 post-step + 128 diagnostic forwards,
  each with 128 single-tick GRU calls = 81,920 calls.
- Approximately 47 times the recurrent-call count for 2.8125 times the rollout
  data. This is a code-derived call-count comparison, not a measured runtime
  speed ratio. Groups without trainable frames would be skipped.

## Recommended correction (not implemented)

Process contiguous reset-free time spans in batched GRU calls. At each actual
reset boundary, zero only the appropriate hidden-state rows. Keep the complete
360-tick gradient graph and trainable-frame-weighted accumulation; no hidden
detach, shorter trace, smaller effective PPO batch, reward change, or
architecture change is required by this correction.

Verify forward, final-hidden, gradient, and optimizer parity within numerical
tolerance (including mixed reset masks), then measure full-scale memory and
end-to-end update throughput before resuming under a properly recorded
operational revision. Fused GPU execution can change floating-point rounding;
do not promise bitwise numerical identity or an unmeasured speedup.

The preceding memory and gradient-equivalence preflight did not establish
acceptable complete-update throughput. That validation gap allowed this launch.
