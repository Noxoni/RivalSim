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
- Gather/scatter the frozen V5 **base trunk and actor heads** only for assigned
  frozen opponents. Preserve the original full-batch recurrent encoder and GRU.
  Current policy sampling still draws the complete batch from the same generator.
  If there are no frozen opponents, their inference can be skipped entirely;
  inactive hidden state cannot become active without the existing world reset,
  which zeroes both sides. Active hidden state, actor outputs, button decisions
  and reassignment resets are tested.

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

Production timing is in `production_verification.json`; update-50 gameplay is in
`../evaluation_u0050_comparison.json`. A clean implementation test is not a
gameplay capability claim.

## Rejected compact recurrent path

The first compact-batch candidate failed GPU hidden-state parity, both in focused
tests and at production scale: `compact_batch_initial_tests.xml` (52 passed,
one failed) and `compact_batch_initial_scale.json`. Full-scale maximum hidden
discrepancy was 0.0001582503318786621. No optimizer step or production update ran
with that candidate. Model/Adam/source stayed unchanged, and the tolerance was
not relaxed. Keeping just the GRU full-sized still failed the small GPU test
(maximum hidden discrepancy approximately 0.0000431).

The corrected candidate keeps **both** recurrent encoder and GRU full-sized,
reducing only base-trunk/head work. The same focused GPU parity test then passed.
This is narrower than the initial performance proposal, selected to preserve
numerical behavior rather than silently accept the changed recurrent calculation.

## Verified deployment

- 53 focused CPU/CUDA tests passed (`full_tests.xml`). Two cuDNN warnings come
  from deep-copied synthetic test models, not production logs.
- Exact-scale final candidate: PASS, 32768 worlds x 360 ticks, five different
  182-sequence backward comparisons, no optimizer step. Measured actor/active
  hidden/value outputs and all compared gradients had zero absolute discrepancy.
  Parameter, optimizer and source identities remained unchanged.
- Peak allocated tensor memory: 20.813 GiB in the isolated benchmark.
- Matched-input bootstrap inference median: 6.060 -> 2.145 ms.
- Matched-input frozen-opponent inference median: 3.865 -> 2.053 ms.
- Mean of five paired minibatch medians: 52.505 -> 49.351 ms (about 6.4% more
  minibatches/second). Includes full backward and post-step KL, but no Adam step.
- Both the original accepted update50 and execution-metadata-rebound update50
  are committed as immutable checkpoints. All 33 non-authority checkpoint fields
  hash identically across the rebind, including weights, Adam, counters and RNG.
- New authority SHA256: `AEA37884836DD4055BACCD605E3F660A0EC2D70162DF5C3032DF6C43134DFD72`.
  New launch SHA256: `E8B08BD9A0FA0CBAF81B352F98AC94DECBFBAA79E32EA932103B35B2AE10BB1A`.
  Authority and rebound checkpoint were pushed/read back before accepting update51.
- Worker45640 (launcher15400) resumed at 2026-09-04 21:04:28 America/New_York.
  Inspect `campaign.optimized.stdout.log` and `.stderr.log` in the same run directory.
- Accepted updates51/52: 58.864 and 65.132 seconds, mean61.998, versus73.956 seconds
  for the seven immediately preceding stable updates44-50. This is an early
  sequential observation (16.2% less time, 19.3% greater throughput), **not** a
  paired causal estimate; scenario/reset mix and runtime resource conditions vary.
- Published update52: `execution_verified_u0052.pt`, SHA256
  `2F2942A12C1C7396D1850DF204CC1242AE61E1B85DE623D01B8A636B408CD84E`.
  All24 Adam step counters advanced exactly1444 (722/update). Model and optimizer
  finite, contracts unchanged, original parent artifacts unchanged, no hard failure.
- Training remains active under the original deadline and total-update100 bound.
  Next scheduled evaluation is100. No extra optimization, evaluation or campaign
  was authorized by the benchmark itself.

## Update50 evaluation (before execution optimization)

Same1024-world/3600-tick deterministic scenario protocol as20, not a full-match
win-rate benchmark. Against Nexto:174/946 ->178/930 goals for/against,
13.5664 ->14.1406 touches/min, no-touch resets0 ->0, speed1194.2 ->1177.0uu/s,
goalward-touch fraction0.8883 ->0.8646. Against frozenV5:400/386 ->359/355,
15.5039 ->15.9453 touches/min, no-touch resets2 ->3, speed1186.0 ->1163.9uu/s,
goalward-touch fraction0.8594 ->0.8246. This is mixed/incremental gameplay movement,
not a large capability gain; none of it is attributed to execution optimization.
