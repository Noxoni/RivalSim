# Prospective exploration-only correction

The newly measured parent kickoff-start distribution has mean top-action mass
0.996659 and jump-action mass8.20e-12 even at training temperature2. This is a
local saturation problem, not evidence that higher aggregate entropy elsewhere
means kickoff exploration is healthy. Parent/child argmax changes at these
154 sampled starts: zero. Collection/replay log-probability max discrepancy was
1.48e-5; no optimizer or model mutation occurred in that diagnostic.

Ordinary entropy has a vanishing softmax-logit gradient for very low-probability
actions. A uniform log barrier has derivative pi_i -1/K, so suppressed actions
still receive an exploration gradient. Proposed loss addition:

`beta * KL(Uniform_90 || pi_T2) = beta * (-mean(log_softmax(logits_T2)) - log(90))`.

This is an exploration regularizer, NOT an old-policy retention objective, KL
rejection, action reward, jump reward or controller override. Every joint action
has equal reference weight. PPO's clipped policy loss, critic isolation, finite
safeguards, rewards, task starts, controller cadence and native Nexto remain
unchanged. Deterministic evaluation still uses the original raw actor argmax.

Before any learning, measure policy, ordinary entropy and unit-barrier parameter
gradient norms on eight deterministically selected128-sequence microbatches from
a fresh1024-world/90-decision parent rollout. Fixed candidates beta=.01,.03,.1.
Select the highest candidate with median barrier/policy gradient norm<=0.5 and
maximum<=1.0 across those microbatches. If none pass, stop at calibration; do not
weaken that selection rule after seeing outcomes. Use simulator training states
only, no test or post-training result for coefficient choice. No optimizer step.

This regularizer is an experiment, not a guarantee of SSL or improved kickoffs.
Do not launch until the chosen coefficient and complete bounded campaign
authority, tests and preflight have been committed and remotely verified.

Research context: Garg et al. document poor adaptability of suboptimally saturated
softmax policy-gradient policies. We are not implementing their alternate
estimator or claiming their experimental/theoretical results apply to this PPO
intervention: https://proceedings.mlr.press/v151/garg22b.html . The derivative
above is checked directly against autograd, not inferred from that paper.
