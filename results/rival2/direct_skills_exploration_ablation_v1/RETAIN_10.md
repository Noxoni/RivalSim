# Retain arm +10: modest scoreline improvement, still noncompetitive

The first frozen evaluation completed all ten300-second native-v5 Nexto
matches at cumulative760. These are the same development layouts/sides as
parent750, raw-argmax Rival, no sampling noise in Rival evaluation, and
unassisted standing kickoffs. No matches were rerun or omitted.

| Metric | Common parent750 | Retain +10 /760 |
| --- | ---: | ---: |
| Wins | 0/10 | 0/10 |
| Rival goals | 2 | 3 |
| Nexto goals | 194 | 184 |
| Goal difference | -192 | -181 |
| Rival contacts/min | 14.34 | 17.72 |
| Standing-kickoff first contacts | 0 | 0 |

Paired goal difference: five worlds improved, two unchanged, three worsened.
The aggregate scoreline improves by eleven goals, but this remains extremely
poor scoring and no wins. No statistical significance or SSL claim follows
from ten fixed development matches. The withdraw arm has not yet run, so this
is not evidence that retaining the added exploration loss is preferable.

Rival made886 contacts versus Nexto's1,288. Rival was the next contacting
player in497/881 resolved followups (56.4132%), versus49.0113% at750.
Repeated nearby collisions can produce this: it is not proof of continuous
possession, deliberate control or learned mechanics. Forward ball-velocity
change occurred at576/886 contacts; forward displacement before the next
contact/goal at484/884 resolved records.

The first ten updates consumed44,236,800 trainable decisions and117,964,800
physical world ticks, with1,414 Adam steps. Maximum completed-update mean KL
was0.0044209996255176204 and maximum sample KL1.5045428276062012. KL remained
telemetry only, with zero KL rejections. All28 checkpoint/lineage checks,
seven objective checks and eleven match-integrity checks passed. These
checks establish execution integrity, not useful gameplay.

Checkpoint: `checkpoints/rival2/direct_skills_exploration_ablation_v1/retain/child_000010.pt`.
SHA-256: `FBAB00F0728C4EF79C58D27037B3DDAE09839741996794FD4C9EE4B7D060D493`.

The worker continued toward its frozen+30 endpoint after evaluation. Do not
change coefficients, rewards, scenarios, stopping or selection from this
intermediate result. Withdrawal will start from the same original750 parent,
not from this checkpoint or the retain arm's final weights. Judge the final
equal-offset paired experiment on scoring/conceding with tradeoffs explicit.
