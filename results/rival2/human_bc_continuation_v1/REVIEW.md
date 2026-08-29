# Rival human BC continuation V1 review

Verdict: **BLOCKED**.

The immutable V1 parent reproduced exactly. Cumulative step 192 was accepted by
every frozen retention and distribution guard and improved validation complete-
action RMSE from 0.560036 to 0.549960 for gameplay and from 0.535502 to 0.524622
for mechanics. Its combined selection score also improved from 0.917482 to
0.917237.

The continuation authority incorrectly applied the 0.0025 material-score delta
both to plateau-patience reset and to replacement of the best checkpoint. The
step-192 score improved by only 0.000245, so the runner kept the V1 parent as the
selected checkpoint even though step 192 was the lower-score eligible candidate.
The next cumulative step-224 boundary failed only the frozen critic
maximum-absolute-drift guard at every prospectively frozen LR retry. All attempts
rolled back exactly and no safety threshold was weakened.

The task-local test access evaluated only the unchanged V1 parent after the
incorrect selection. The accepted step-192 candidate never touched the human or
simulator test split. This run is retained as negative evidence rather than
retroactively changing its selection rule.

The required correction is a prospectively frozen continuation V2 rule:

- update the best checkpoint for any strict improvement in the combined held-out
  score among eligible guard-safe candidates;
- use the 0.0025 material delta only to reset validation-plateau patience;
- require both human validation families to improve materially over the parent;
- keep every data, objective, sampling, test, and retention contract unchanged.

No PPO, reward, mechanic-definition, closed-loop mechanic, raw-recording, or
observation/action-contract work occurred.
