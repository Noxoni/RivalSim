# The measured exploration gradient is not dominating PPO

One bounded probe completed after the actual +30 worker exited. It used the
preserved final checkpoint, 1024 fresh training-distribution worlds and the
existing eight-microbatch gradient measurement. No optimizer was constructed
or stepped. Checkpoint and in-memory model stayed unchanged.

For the actual beta 0.01 term:

- Median exploration/PPO actor-gradient norm ratio: **0.09665**.
- Maximum across eight microbatches: **0.16416**.
- Median gradient cosine: **-0.22586**.
- Mean exploration component along the PPO gradient, relative to PPO norm:
  **-0.01659** (about -1.66%).

This is much smaller than the initial parent's median ratio of 0.37308. It
does not support the proposed explanation that the extra exploration gradient
is now overwhelming the reward-learning gradient. The small negative cosine
indicates some opposition in these batches, not dominance or a demonstrated
cause of failed gameplay.

The reused calibration routine also prints a candidate table and selects 0.03
under its original *initialization* rule. **That output is not adopted.** It is
not a training authority and does not justify increasing the coefficient.

The +30 model recovered versus +15 in all ten paired goal differences but still
failed to outperform the controlled parent. Neither "more exploration fixes
it" nor "the critic never updated" is established. Do not promote the model or
change rewards/architecture merely to fit those explanations.

This probe is complete and must not be rerun. It measures newly collected
training states, not the original update-30 minibatches, optimizer-preconditioned
steps or a counterfactual match under another coefficient. Its conclusions are
limited to those measured gradients. Any next learning experiment requires a
new prospective plan and must retain parent and +30 match comparisons.
