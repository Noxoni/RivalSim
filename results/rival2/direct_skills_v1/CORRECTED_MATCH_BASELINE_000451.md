# Corrected standard-reset baseline: +0 versus +451

Method: `RIVAL2_STANDARD_KICKOFF_CONTACT_CACHE_RESET_V2`, prospectively published
at `74b18d7b54fa8bcab1ada83cc6e01aa5d9adc29e` with all eleven changed files
remotely read back. No optimizer, model selection, action injection or opponent
change. The original +0 and preserved latest +451 checkpoints remain hash-exact.

| Same corrected ten-match method | Parent +0 | Latest +451 |
|---|---:|---:|
| Wins / losses | 0 / 10 | 0 / 10 |
| Rival goals | 1 | 59 |
| Nexto goals | 272 | 156 |
| Rival contacts | 203 | 538 |
| Contacts/min | 4.06 | 10.76 |
| Same-player next contacts / resolved | 0 / 200 | 142 / 482 |
| Matches without Rival contact | 0 | 0 |

Every latest match improves goal differential versus its matched parent, but
every one remains a loss: 7/17, 8/16, 6/15, 6/13, 7/15, 5/16, 2/19, 6/15,
6/16, 6/14. This is learned improvement over the entering parent, not SSL,
Nexto parity, or proof of shooting/possession mastery.

Latest contact accounting: 142 same-player + 340 opponent + 52 goal-ended +
4 unclosed = 538. Same-player follow-up fraction 0.2946058091286307 is not
possession duration. The 488 positive velocity-change contacts are not 488
shots. Net displacement bins are 42 backward / 223 neutral / 269 forward.
There is one native demolition; intent is unknown. No concessions have at most
one contact since reset under this method. No-touch resets are disabled, so
their zero count is protocol, not behavioral proof.

Rival scores the first goal in all ten latest matches, but only three second
goals. It scores 49 later goals while conceding 156 later goals. The remaining
opening-versus-later gap is not solved by cache invalidation; opponent cadence,
history and different encountered trajectories remain possible contributors.
Do not conclude all failures were the wheel-cache bug.

All eleven native-match integrity checks pass for both checkpoints; zero new
optimizer steps. Raw outcomes, event arrays, source and checkpoint identities
are in `corrected_reset/`. The evaluator's already-completed-result path was
checked without rerunning matches; its integrity-field lookup was corrected
from `checks` to the reducer's actual `integrity` key. No native run failed or
was replaced by that bookkeeping fix.

## Continuation decision

Resume the intact +451 model/Adam/RNG after this comparison, retaining fresh
physical episodes and zero hidden according to the existing resume protocol.
No reward or PPO hyperparameter adjustment. Finishing remains 20% of starts.
The next scheduled +500 skill test remains comparable to old skill tests because
training/task reset arrays were proved bit-identical. For full matches, compare
+500 with this corrected +451 and corrected +0, not old +400/+450 results.
Keep periodic evaluation, and investigate genuine stagnation rather than
equating safe optimizer progress or frequent on-target proxies with competence.
No stronger-opponent/SSL objective has been completed.
