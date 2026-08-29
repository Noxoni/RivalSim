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

The mechanics-removal gate passed and was pushed before training. The first
fresh-Adam minibatch proposal then produced KL 0.414318710565567, exceeding the
frozen 0.10 hard guard. Retention KL was 0.12192033976316452. Transactional
rollback restored model parameters, optimizer state, and Adam step counters
exactly; no PPO update was accepted and no campaign checkpoint was selected.

This is classified as a policy-displacement/capability safety failure, not a
reward-path or operational failure. Training remains stopped. See
`failure_analysis.json` for the full evidence and the prospective next-step
recommendation.
