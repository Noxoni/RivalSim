# More contact is not evidence that the task skills have been learned

This CPU-only reduction uses the two already committed, immutable training
prefixes through cumulative700. No new rollout, evaluator, model loading,
optimizer, reward change or detector was involved. Five consecutive ten-update
windows are reported, not cherry-picked individual minibatches. Both opponent
families and all five reward roles are retained in task_progress_700.json.

## Shooting remains a learning gap, not only a match-transfer gap

Finishing success in the active code is an actual native goal. Projected
on-target touches remain raw telemetry and pay no finishing bonus. Across every
source row and both opponent families, success_endings equals goals,
direct_reward equals negative failed_attempt, and approach reward is zero.

| Update window | Nexto shooting goals / observed endings | Nexto goal fraction | Self-play shooting goals / observed endings |
| --- | ---: | ---: | ---: |
| 651-660 | 1027 / 9518 | 10.7901% | 4494 / 14020 |
| 661-670 | 872 / 11245 | 7.7546% | 3153 / 14118 |
| 671-680 | 896 / 11392 | 7.8652% | 2986 / 14257 |
| 681-690 | 978 / 10043 | 9.7381% | 3056 / 13642 |
| 691-700 | 879 / 10900 | 8.0642% | 2269 / 13686 |

These figures do not support a claim that finishing has already been mastered
in training and merely needs deployment adaptation. They also do not identify
the cause of weak learning or prove any particular optimizer/reward correction.
Self-play's defender evolves too, so its changing goal fraction is not an
evaluation against fixed opposition.

## Kickoff contact recovery has not established control

Rates below are per learner minute facing Nexto, pooled over each full window.
The exposure denominator is learner samples/1800 at30Hz; no averaging of ratios.

| Window | Kickoff race-first-contact rate | Kickoff control-gain rate | Natural-play contact rate | Natural-play goal rate |
| --- | ---: | ---: | ---: | ---: |
| 651-660 | 0.6745 | 0.2736 | 7.6445 | 0.1219 |
| 661-670 | 0.0009 | 0.1141 | 5.6734 | 0.0770 |
| 671-680 | 0.0832 | 0.0683 | 4.8789 | 0.0760 |
| 681-690 | 0.7247 | 0.0571 | 5.4679 | 0.0816 |
| 691-700 | 1.2015 | 0.0867 | 6.1729 | 0.0849 |

Kickoff control is the existing own-touch/proximity/relative-speed/opponent-gap
predicate held for eight decisions, not an independently adjudicated possession
or kickoff win. Half the dedicated training kickoff starts are momentum-assisted.
Their stochastic first-contact rate is not comparable to deterministic standing
kickoff first-contact wins; the latter remained zero at the700 evaluation.

For challenges against Nexto, the control-gain rate was0.7095/min at671-680 and
0.7608/min at691-700, below0.9225/min at651-660. For defense, the recorded clear
rate was3.1461/min at671-680 and3.2232/min at691-700, below3.4243/min at651-660.
These are existing task proxies, not proof of general possession/save mastery.

## Limits and next decision

These are on-policy training states under T2 sampling, not identical replayed
states. Episode age, early entry transients, visitation and the evolving self-play
opponent can affect trends. A fresh physical initialization occurred at681.
Event windows can straddle episode boundaries: dividing first-touch events by
episode endings would not be an attempt success fraction. Native goal counts
are joined to their actual terminal endings; the table uses observed endings,
not a frozen cohort of attempts starting inside each window.

Read these numbers alongside REVIEW_000020.md: the held-out development matches
had more contacts but only1 goal for/222 against,0/10 wins. There is no evidence
here to promote the checkpoint, declare SSL, or claim that the only remaining
issue is using already-mastered skills in normal play.

Keep the current frozen continuation unchanged through its remaining725/750
evaluations. At review, if finishing remains weak, treat it as an unlearned
basic outcome rather than adding another specialist or assuming more touch
reward would solve it. This analysis does not authorize a reward/parameter change.

Rebuild (standard-library only):

```powershell
.venv\Scripts\python.exe -B benchmarks/summarize_direct_skills_task_progress_700.py
```

Seven focused tests passed; deterministic rebuild twice produced
0D0E892792FAA28B2C5B98B35472F250CF7D030844D9A58E0649E62F36612214.
Input identities are embedded in the script and each JSON window; changed
prefixes, gaps, missing roles, invalid counts, wrong30Hz sample allocation or
changed finishing-payment semantics are rejected.
