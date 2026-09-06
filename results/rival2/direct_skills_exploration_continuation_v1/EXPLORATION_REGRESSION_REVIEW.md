# Exploration and regression: evidence limits before the remaining evaluations

Written on 2026-09-06, while the frozen continuation is still running. This is
a read-only interpretation of existing artifacts and source, not a new training
authority. No rollout, optimizer step, coefficient change, deployment, or
checkpoint selection is performed by this review.

## What the evidence establishes

- Original cumulative650: 0/10 wins, 7 scored /196 conceded, 9.12 contacts/min.
- Exploration+30 / cumulative680: 0/10, 2/228, 8.88 contacts/min.
- Exploration+50 / cumulative700: 0/10, 1/222, 10.30 contacts/min.
- Every one of these standing-kickoff evaluations has zero Rival first contacts.
- The 650-to700 comparison is regression in scoring and goal difference, despite
  a higher contact rate. The brief recovery from exploration+15 to+30 did not
  establish recovery to650, let alone competitive improvement.
- TASK_PROGRESS_700.md shows that the tasks themselves are not mastered:
  Nexto finishing goals/observed endings declined from10.7901% in651-660 to
  8.0642% in691-700. These are on-policy windows, not matched task trials.

Sources: the two parent match artifacts and progress_000020.json, plus the
hash-bound task_progress_700.json. Physical re-entry at681, evolving self-play,
changing sampled states, and correlated match trajectories limit causal claims.

## What the exploration term actually does

The active source is rivalsim/direct_skills_log_barrier_v1.py. Its additional
minimization loss is:

    0.01 * KL(Uniform90 || pi_temperature2)

The ordinary entropy term remains0.001. The barrier is computed on the same
temperature-two logits used for on-policy sampling and likelihood. For those
logits its derivative is0.01*(pi_i-1/90); relative to raw pre-temperature logits
there is an additional factor1/2. This gives suppressed actions a nonvanishing
exploration gradient, unlike an almost-saturated ordinary entropy gradient.

It also means the extra pressure does not turn off merely because exploration
has recovered. The barrier has its minimum at a uniform action distribution.
It opposes concentration even when concentration is useful in a particular
state. This is an actor-objective change, although environment rewards are
unchanged. It is not old-policy retention and does not reject updates for KL.

The fixed evaluation uses raw-logit argmax, not temperature sampling. Positive
temperature scaling alone cannot change a fixed model's argmax. Thus the worse
deterministic matches cannot be explained as simply sampling extra noise during
evaluation. Training under the changed distribution/objective can still alter
the model's deterministic action rankings and state visitation.

## Why the existing gradient calibration does not settle causality

The already-completed, no-optimizer post+30 probe is stored in
results/rival2/direct_skills_post_exploration_probe_v1/barrier_calibration.json.
On its eight128-sequence training microbatches, coefficient0.01 had median added
barrier/PPO gradient-norm ratio0.0966524610 and maximum0.1641616583. Several
barrier/PPO gradient cosines were negative. These are raw aggregate parameter
gradients on that one sampled corpus, not measured Adam-preconditioned updates,
per-task causal effects, or a bound on cumulative behavioral change.

Consequently, neither 'the barrier dominates learning' nor 'the barrier is too
small to harm useful learning' follows from the calibration. The candidate
selector also labeled0.03 eligible on this probe; that does not justify raising
the live coefficient, and0.03 was not adopted. No completed gradient probe is
being rerun.

## Research cross-check, not a transferable convergence guarantee

Garg et al. document slow escape from suboptimal softmax saturation and study
an alternative policy-gradient estimator. Rival is not using their estimator;
the paper does not validate Rival's coefficient or expected improvement rate.
[An Alternate Policy Gradient Estimator for Softmax Policies](https://proceedings.mlr.press/v151/garg22b.html).

Yuan et al., section4.2.1/equation28, explicitly analyze a return objective with
an added uniform-to-policy KL regularizer. Their softmax-tabular results concern
specific assumptions and gradient estimators, not this recurrent neural PPO
implementation. This supports distinguishing exploration regularization from a
neutral numerical repair; it is not evidence that constant0.01 is optimal here.
[A general sample complexity analysis of vanilla policy gradient](https://proceedings.mlr.press/v151/yuan22a/yuan22a.pdf).

## Next decision, not an automatic extension

Finish the already-frozen725/750 evaluations with unchanged settings. Do not
promote700 or equate increased entropy/contact count with better gameplay.
At the terminal review, compare all checkpoints against650 as well as680.
No settings in this arm should be tuned from this review.

If the completed arm still fails to recover competitive performance, a useful
next *proposal* is a bounded matched-parent comparison of continued barrier
pressure against removing or prospectively tapering the extra pressure, with
rewards, task corpus, temperature, architecture and evaluation held fixed.
Use the same entry model/optimizer/RNG and equivalent fresh physical starts for
both arms; do not compare different parents and call that an ablation. Choose
one alternative prospectively, not a retrospective coefficient sweep against
the match set. Such a comparison can distinguish a harmful ongoing pressure
from ineffective task learning more directly than another raw-gradient probe.
No such comparison is authorized by this document or launched here.

The regression's cause remains unproven. This review narrows a testable
hypothesis and corrects overinterpretation of earlier diagnostics; it does not
claim an implementation bug, a guaranteed fix, or SSL-level capability.
