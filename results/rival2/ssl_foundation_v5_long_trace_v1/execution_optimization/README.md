# Execution reuse at the update-50 boundary

User-authorized implementation of the preceding `speed_investigation`. This is
not another learning amendment or a new campaign. The healthy old worker stays
running through accepted update 50 and its already scheduled deterministic
evaluation, then an owned `STOP_REQUESTED` pauses it for isolated GPU validation.
The existing monitor is temporarily paused during maintenance.

## Scope

- Evaluate only the independent critic for rollout bootstrap values. Use exactly
  the same transition observation, including pre-reset truncation observations.
- Evaluate only the actor for post-step KL. Preserve every trainable sample and
  all the existing KL telemetry; no thresholds or telemetry are removed.
- Cache reset-time metadata and gathered minibatch observations/hidden state for
  the post-step pass. Cache is local to one effective minibatch and bound to the
  same unmodified Boolean tensor. No recurrent gradients are detached.
- Check every model parameter for nonfinite values at the same post-step boundary
  with one host reduction, instead of one per parameter. Existing corruption
  exceptions and whole-update model/Adam/RNG rollback remain unchanged.
- Run frozen V5 inference only for the assigned frozen opponent, not all world
  perspectives. Current policy sampling still draws the original complete batch
  from the same generator. Unused frozen hidden rows are **not equivalent** to the
  old unused computation: they are deliberately not advanced. They cannot become
  active without the existing world reset, which zeroes both sides. Active hidden
  state, actor outputs, button decisions and reassignment resets are tested.

The architecture, precision, 32768 worlds, 360-tick recurrent sequences,
182-sequence effective minibatch, two epochs, 722 Adam steps, learning rates,
three-second GAE half-life, reward, scenarios, 40/40/20 opponent mix, deadline,
update-100 bound and evaluation protocol stay unchanged. The default shared PPO
path remains unoptimized unless explicitly enabled by this campaign.

## Validation and deployment

Focused tests compare outputs, values, hidden state, input/parameter gradients,
Adam tensors, telemetry, skipped microbatches, inactive/active opponent changes,
and every-parameter nonfinite injection. CPU/CUDA tolerance remains the preceding
FP32 execution gate: `atol=3e-6, rtol=1e-4`; button decisions are exact.

`validate_rival2_execution_optimization.py` is a separate **no-step** benchmark:
one real 32768-world x 360-tick rollout, matched full-size input inference timings,
five different 182-sequence minibatches with repeated full backward/post-KL
comparisons, memory/finite checks, source/model/Adam identity checks. Frozen
noncandidate environment files must match the pre-optimization authority.
Component timings are not a promised end-to-end speedup. The active run must be
paused before GPU tests or this benchmark run.

After it passes, `prepare_rival2_execution_optimization.py` archives old authority
and state records, freezes the new implementation hashes and rebinds only
source/phase-transition authority metadata of the preserved accepted update-50
checkpoint. Every other checkpoint field must hash identically. It cannot reset
model weights, Adam, counters or RNG, overwrite the preserved checkpoint, or
extend the original wall-clock deadline. Commit/push the authority before resume.

Production timing and update-50 evaluation results will be recorded separately
after validation; a clean implementation test is not a gameplay capability claim.
