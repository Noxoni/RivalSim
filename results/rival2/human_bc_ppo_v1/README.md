# Rival 2.0 Human BC V1 to 120 Hz PPO campaign

This directory is the authoritative evidence package for the bounded 10-hour
PPO campaign starting from the accepted Human BC V1 step-160 policy. The frozen
configuration is `frozen_config.json`. The user's later wall-clock instruction
prospectively supersedes the original +120 stop; +120 remains an ordinary
checkpointed point, not the campaign boundary.

The transition deliberately combines only these compatible pieces:

- model weights from the accepted Human BC V1 checkpoint;
- observation/action/reward/PPO contracts from the clean 120 Hz bootstrap;
- the bootstrap's bounded historical Rival pool with each legacy snapshot's
  original 30 Hz cadence metadata;
- a fresh split Adam optimizer with no historical or BC Adam moments;
- a retention corpus collected by rollout-only inference from the BC parent.

The active opponent distribution is 80% current Rival and 20% historical Rival.
Nexto and Wisp have zero probability and must have zero assignments, samples,
and policy calls in every recorded rollout.

## Execution boundary

`benchmarks/run_rival2_human_bc_ppo_v1.py --preflight-only` performs the static
and runtime mechanics-removal audit without an optimizer step. Its evidence must
be committed and pushed to `origin/main` before the training mode will run.

Training writes crash-recovery state after every accepted update and a milestone
checkpoint every 30 accepted updates to the explicitly selected external work
directory. It also records an immutable wall-clock deadline so operational
restarts do not silently extend the campaign. Only the final resumable checkpoint
is copied into the repository; hashes and metadata for every milestone are stored
here. This avoids treating hundreds of megabytes of redundant model weights as
Git evidence.

## Interpretation

The 120 Hz production reward includes trusted gameplay components and the
physical unnecessary-flip contact guard only. Historical mechanic source and
evidence remain in Git, but Gameplay V3 is not constructed, its kernels cannot
launch, named-mechanic rewards are zero, and named/controlled-flick exemptions
are disabled. The older Gameplay V2 strict-dash detector is also not launched in
the active reward mode and its component must remain exactly zero in preflight
and campaign telemetry.

## Outcome

The original `1e-4` fresh-Adam proposal reproduced its KL rejection exactly and
was rolled back. The frozen same-minibatch transition sweep selected `1.25e-5`,
the highest candidate satisfying both `0.02` soft limits. The transition
authority was committed and pushed before the first accepted PPO step.

The wall-clock campaign then completed 36,008.227 seconds and accepted 4,067
PPO updates. No hard safety guard fired. There were 84 allowed soft retention
early-stops, 2,447 LR backoffs, and 2,531 transactional retries. The maximum
accepted minibatch KL was 0.0199983064, the maximum retention KL was
0.0199981406, and the maximum completed-update mean KL was 0.0217673108, below
the unchanged 0.05 hard completed-update guard.

The transition warmup remained active through the final update because the
policy did not complete two consecutive clean `1e-4` updates. It therefore
never entered the normal production reset rule; this is recorded as a bounded
adaptive outcome, not silently relabeled as normal `1e-4` operation.

Across first-100 versus last-100 update windows, touches rose from 16.5118 to
19.1980 per minute, goals rose from 1,307.98 to 1,424.92 per update, movement
speed rose by 65.73 uu/s, and no-touch truncations fell from 6.38 to 4.14 per
update. Unnecessary flip contacts rose from 7.5577 to 8.7754 per minute while
their fraction of flip-active contacts fell from 0.6172 to 0.5421. This mixed
ball-contact result should be judged visually before more training.

The final resumable checkpoint and the +30 milestone are stored in Git. See
`FINAL_AUDIT.md`, `final_evidence.json`, `checkpoint_milestones.json`, and the
full `training_curve.jsonl` for the reviewer package.
