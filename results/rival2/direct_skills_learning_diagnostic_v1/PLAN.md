# No-update learning-signal diagnostic

Question: after the completed kickoff-outcome arm failed to improve actual
Nexto matches, do the saved logs and parent-generated trajectories show useful
reward exposure, active exploration and action-likelihood movement?

Compare preserved parent650 and kickoff child+15, exact hashes bound in the
diagnostic script. No old/new optimizer is created or called. Rewards, scenarios,
model architecture and running campaigns are not modified. No new campaign.

Read the complete committed 15-row curve; aggregate updates3..5 versus13..15.
These are descriptive exposure windows, not independently paired episodes.

Generate one bounded 2048-world /90-decision rollout with the existing kickoff
arm's environment, seed, native-v5 Nexto and T2 parent policy. It is a newly
generated diagnostic subset, not the original32768-world training minibatch.
Replay its identical observation sequences and initial/reset hidden states
through parent and child. Verify parent collection/replay likelihood parity,
model/checkpoint immutability and finite outputs. No child trajectory is generated.

Report per reward role and opponent: normalized/raw advantage, entropy,
maximum action probability, jump probability, sampled-action log-probability
change and argmax changes. Separate paid first touches, native scores/concedes
and advantage signs. Episode-start outputs are separate from the full rollout.
These are diagnostic associations, not causal proof that a reward caused a
particular weight change. No new reward detector or action target is added.

Preserve per-sample arrays and summary in Git. Do not claim an assisted versus
unassisted success breakdown from aggregate logs; that partition was not logged.
Use the result to choose the next intervention rather than automatically
launching more of the completed experiment.
