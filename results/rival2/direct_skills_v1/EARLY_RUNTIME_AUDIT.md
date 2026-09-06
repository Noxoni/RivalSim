# First four accepted updates: runtime audit, not a capability verdict

Read-only audit during the healthy, uninterrupted Direct Skills V1 process15812.
No optimizer, reward, curriculum or policy changes were made by this review.
Source: training_curve_through_000004.jsonl, an immutable prefix of the external
training curve. Repeated PPO epochs are not additional environment samples.

| Update | Learner decisions | Nexto learner decisions | Native goals | World resets | Timed resets | Failed player-drill attempts | Completed mean KL |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4,423,680 | 1,474,560 | 2,515 | 2,515 | 0 | 0 | 0.0049207042 |
| 2 | 4,423,680 | 1,474,560 | 4,672 | 4,672 | 0 | 0 | 0.0034994738 |
| 3 | 4,423,680 | 1,474,560 | 6,347 | 6,347 | 0 | 0 | 0.0036365589 |
| 4 | 4,423,680 | 1,474,560 | 7,202 | 17,027 | 9,825 | 12,466 | 0.0034078190 |

All four rows have exactly one-third Nexto learner exposure and zero KL
rejections. Goals produce actual world resets; the twelve-second skill limit
becomes reachable in the fourth three-second rollout. Failed attempts are
player-level counts and may exceed world-level timed resets because both
current agents learn in self-play. No-touch resets are zero in these early rows;
that longer boundary is not yet exercised by this prefix.

For each row, reconstruct the return sum as:

`sum(potential_reward_components except total) + sum(by_reward_role_and_opponent[*].direct_reward)`

The absolute difference from the logged total reward is at most
`0.000013192231563152745`, consistent with floating-point reduction order.
Natural-lane direct rewards remain zero; skill-lane natural rewards remain zero.
The totals are not silently dropping the direct rewards or goal outcome.

**These are not win rates or evidence of improvement.** Scenario ages, resets,
opponent exposure and task proportions change between early rollouts. The rising
goal counts cannot be compared as though each row were the same fixed test.
The next capability evidence is the scheduled offset10 fixed-case evaluation.

The CPU-only `benchmarks/report_rival2_direct_skills.py --update 10` compares
completed artifacts without running another policy or selecting a candidate.
It validates all64cases per family, raw-versus-summary conservation, matching
scenario hashes and authority, and zero evaluation optimizer steps. It reports
paired goal outcomes separately from declared task-proxy events. Nine focused
tests cover identity, changed outcome and corrupt/incomparable evidence.
No new acceptance threshold, detector, reward or automatic tuning is introduced.
