# Development objective and competing explanations

SSL-like gameplay is the user's objective, not a checkpoint label we have earned.
The current bottleneck is ordinary ball acquisition and recovery after a miss.
Scenario scoring, stable PPO updates, replay integrity and deployable competitive
strength are separate outcomes. Ultimately we need natural matches against
strong stable opponents and an actual Rocket League deployment check; repeated
short Nexto kickoff scenarios cannot establish SSL rank or strength.

## Current experiments and boundaries

1. **Excessive independent analog exploration:** the two-arm committed pilot
   isolates a 0.5 multiplier of effective learned sigma, retaining the parent
   weights/Adam, Bernoulli buttons, state rewards and curriculum. This is a
   falsifiable hypothesis, not an asserted diagnosis. Early no-step rollout
   contacts were 23,871 versus 24,280 over three simulated seconds; this small
   immediate change does not establish better learning.
2. **Ground approach control:** completed native trajectories show initial
   boosting, pass-by/misses and failure to return. An isolated scripted ground
   pursuit reference is prepared to check whether the *existing observation
   and native controls* can solve these situations. Its results are never
   attributed to Rival, added to training, used as reward or deployed. It has
   no PPO, imitation objective, mechanic detector or scripted training prefix.
3. **Difficulty/coverage:** if the exploration comparison is inconclusive,
   investigate closer, smaller-angle acquisition before another large campaign.
   This is a possible subsequent prospectively defined curriculum experiment,
   not a change to the running pilot. Do not mix it into the noise comparison.
4. **Control parameterization:** the primary RLGym implementation offers a
   [90-choice action lookup table](https://github.com/RLGym/rlgym/blob/main/rlgym/rocket_league/action_parsers/lookup_table_action.py).
   The simulator author's [practical PPO guide](https://github.com/ZealanL/RLGym-PPO-Guide/blob/main/intro.md)
   recommends such a fully discrete parser over continuous controls as an easier
   starting point. This is relevant evidence for a *later* controlled comparison,
   not proof that continuous control cannot learn or permission to silently
   reinterpret existing checkpoints. No action-parser or architecture change
   is made in this pilot.
5. **Learning signals:** current rewards remain terminal goals plus potential
   differences. Potential shaping is not permanent payment for being closer,
   touching, moving or retaining possession. A no-touch truncation is not a
   direct penalty. Thus the learner does not receive the simplistic causal
   labels 'all preceding actions were bad' after every timeout. The 90-step
   GAE window bootstraps at its right edge; it is not three seconds of observed
   consequences for every action. Do not conceal a direct behavior reward inside
   a claimed potential-only contract or add blanket jump/flip penalties.

The first audit found exact stored-versus-recomputed log likelihood agreement
for the checked full-length native sequences, stable observation storage,
consistent critic values, finite updates and unchanged parent tensors. These
tests eliminate specific timing/distribution failure hypotheses; they do not
prove the entire learner is optimal or the environment free of every possible
defect. Zero net terminal reward across the two self-play players is expected
from +10/-10 symmetry, not evidence that goals were unpaid.

## Checkpoint interpretation

`exploration` retains the inherited base learned-distribution description.
The experiment's `analog_sigma_scale`, `arm`, effective `policy_config` bounds,
and `comparison_authority_sha256` jointly specify the effective distribution.
For half_sigma, apply the scaled model forward path; never load it into an
unmodified stochastic sampler and silently discard the multiplier. Deterministic
mean/button inference remains identical at initialization. Independent evaluation
explicitly reconstructs that same scaled model. Every comparison checkpoint has
a new format/lineage identifier so the old runner rejects accidental resume.
