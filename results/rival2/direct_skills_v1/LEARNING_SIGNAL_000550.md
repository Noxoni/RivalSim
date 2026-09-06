# +550 bounded learning-signal diagnosis

Training was intentionally paused at accepted +554 after the +550 review.
The exact evaluated +550 checkpoint was used privately for this diagnosis.
No optimizer was constructed or stepped. No policy, reward, environment,
opponent, action table or native observation contract was changed.

## Scope and integrity

The prospectively published `ADVANTAGE_DIAGNOSTIC_PLAN_000550.md` specified
1,024 worlds, seed2026090655, four continuous90-decision rollouts:12seconds/world.
The completed diagnostic contains552,960 learner decisions,184,320 against
Nexto. Four lossless archives preserve the native buffers and read-only taps.
`advantage_diagnostic_000550/summary.json` binds their hashes and all20 unchanged
production sources. `reduction.json` is a deterministic CPU-only reduction.

Two failed **diagnostic helper** attempts are retained separately. Attempt0
collected3seconds/world and stopped before gradients on an invented0.001
replay-comparison assertion. Attempt1 collected another3seconds/world and
saved its numeric buffer, then failed because the helper omitted `model.train()`
before cuDNN GRU backward. Production PPO already sets the mode correctly.
Neither constructed/stepped an optimizer or altered a checkpoint. These extra
6seconds/world are not hidden inside the planned successful12seconds/world.

The completed four-rollout diagnostic verifies:

- Independent CPU GAE equals stored GPU GAE exactly; goal reward signs,
  four-tick discounting and reward-component accounting match.
- True-goal returns equal terminal rewards; final-state truncation bootstrap
  and episode cuts use the existing recurrence, not reset-state bootstrap.
- Full-batch tick-by-tick replay reproduces all stored log probabilities and
  critic values exactly, including across recurrent resets.
- Whole-sequence batching changes sampled log probabilities by at most
  0.0052514076. Ratios stay inside the production0.8..1.2 clipping interval;
  maximum observed ratio1.00526524. The original helper assertion still would
  fail and is explicitly reported, not relabeled PASS. This is a batch-execution
  numerical difference, not a demonstrated lost-hidden-state problem; no
  low-level kernel-specific cause is claimed.
- Actor gradients are finite and nonzero. Critic loss has exactly zero actor,
  shared-policy-trunk, GRU and entity-encoder gradient. Model state and both
  preserved checkpoint files remain unchanged.

## Findings

The policy is close to deterministic, especially against Nexto:

| Learner domain | Mean entropy, nats | Mean highest action probability | States with highest probability>99% |
|---|---:|---:|---:|
| All |0.267739|0.913947|52.56%|
| Nexto |0.170543|0.944178|64.26%|
| Current self-play |0.316337|0.898831|46.71%|

This agrees with the live mean-entropy decline0.341858→0.288297 between the
pre-500 and501–550 windows. It is evidence of narrow exploration, **not proof
that increasing exploration will improve deterministic match performance**.

Local task gradients do conflict: challenge-vs-Nexto and natural-vs-Nexto
actor gradients have negative cosine in all four inspected minibatches;
finishing-vs-Nexto and natural-vs-Nexto in three of four. These are local,
normalized same-buffer gradients, not a causal decomposition of the plateau.
Raw FP32 cosine calculations have small rounding overshoot at self-cosine
(up to about1.00012); do not treat that as an exact geometric measurement or
acceptance gate. Gradient signs of the cited non-near-zero pairs are unambiguous.

The mixed reward lanes also expose critic calibration differences:

| Terminal learner samples | Count | Mean actual reward | Mean critic value | Mean raw advantage |
|---|---:|---:|---:|---:|
| Natural scoring |122|+8.470058|+8.902651|-0.432593|
| Natural conceding |102|-8.616086|-9.314032|+0.697945|
| Skill scoring |270|+9.982157|+9.195373|+0.786784|
| Skill conceding |348|-9.980152|-9.652234|-0.327919|

Natural terminal rewards include the correct `-Phi(current)` correction;
direct-skill terminals do not. **A positive advantage when conceding is a
baseline-relative residual, not a positive concede reward, nor proof the
preceding action caused the goal.** The native goal term itself remains±10
with the correct intra-decision discount. No reward-sign/cadence bug was found.

A descriptive affine calibration fitted on rollouts0–1 and measured on2–3
did not improve critic error: original held-out MSE1.817602, pooled affine
1.824347, per-role affine1.828190. These were CPU least-squares statistics,
never applied to a policy or optimizer. This does not rule out richer critic
conditioning, but does not justify changing architecture from this test alone.

The distinction between potential-shaped values and action advantages follows
[Ng, Harada and Russell's shaping result](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf).
GAE trades estimator variance against value-approximation bias; a terminal
residual alone is not action causality. See the
[original GAE paper](https://arxiv.org/abs/1506.02438).

## Concrete continuation decision

Keep +554 as the continuation parent, with its model, Adam, counters and RNG.
Do not restart from550 or a selected older snapshot. Do not alter rewards,
critic architecture, role balance, resets, opponents, PPO cadence or LR here.

The next bounded experiment changes **one training distribution setting**:
categorical logits are divided by temperature2.0 during both rollout sampling
and PPO likelihood/entropy computation. Keep entropy coefficient0.001.
This is on-policy softening of the same90-action distribution, not additive
controller noise, scripted actions, a different action vocabulary or a new bot.
Positive-temperature scaling preserves deterministic argmax before learning.

Freeze a separate prospective exploration amendment and its tested runtime
hashes before the first accepted step. Preserve old authority/evidence intact.
Run554→600, saving each accepted boundary, then complete the same64-per-family
and ten-match Nexto evaluations and pause for the review. Compare actual scoring,
conceding, ongoing-play and finishing outcomes, not just entropy increases.
No claim of learned shooting, possession or SSL promotion follows from this
diagnosis. This is a bounded experiment, not a promise that temperature fixes it.
