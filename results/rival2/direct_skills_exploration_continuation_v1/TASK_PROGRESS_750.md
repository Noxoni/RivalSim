# Task progress through750

CPU-only reduction of the committed30-update prefix and the closed,
audited70-update continuation. No additional training or matches.

## Shooting outcomes

Fractions use observed episode endings, not an independently tracked
attempt cohort. On-policy states, episode age and self-play change.

| Updates | Nexto goals / endings | Nexto fraction | Self-play goals / endings | Self-play fraction |
| --- | ---: | ---: | ---: | ---: |
| 651-660 | 1027 / 9518 | 10.7901% | 4494 / 14020 | 32.0542% |
| 661-670 | 872 / 11245 | 7.7546% | 3153 / 14118 | 22.3332% |
| 671-680 | 896 / 11392 | 7.8652% | 2986 / 14257 | 20.9441% |
| 681-690 | 978 / 10043 | 9.7381% | 3056 / 13642 | 22.4014% |
| 691-700 | 879 / 10900 | 8.0642% | 2269 / 13686 | 16.5790% |
| 701-710 | 958 / 10927 | 8.7673% | 2332 / 13674 | 17.0543% |
| 711-720 | 947 / 10561 | 8.9670% | 2203 / 13716 | 16.0615% |
| 721-730 | 854 / 10700 | 7.9813% | 2001 / 13431 | 14.8984% |
| 731-740 | 997 / 10593 | 9.4119% | 2114 / 13651 | 15.4860% |
| 741-750 | 1019 / 10638 | 9.5789% | 2044 / 13592 | 15.0383% |

## Task telemetry against Nexto

Every rate below is per learner-minute (1,800 decisions at30Hz).
These are existing event definitions, not newly defined skill detectors.

| Updates | Natural touches | Natural goals | Challenge control gain | Defense clear | Kickoff race first contact | Kickoff control gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 651-660 | 7.6445 | 0.1219 | 0.9225 | 3.4243 | 0.6745 | 0.2736 |
| 661-670 | 5.6734 | 0.0770 | 0.7944 | 3.1160 | 0.0009 | 0.1141 |
| 671-680 | 4.8789 | 0.0760 | 0.7095 | 3.1461 | 0.0832 | 0.0683 |
| 681-690 | 5.4679 | 0.0816 | 0.7157 | 3.6302 | 0.7247 | 0.0571 |
| 691-700 | 6.1729 | 0.0849 | 0.7608 | 3.2232 | 1.2015 | 0.0867 |
| 701-710 | 6.1237 | 0.0858 | 0.7857 | 3.1693 | 1.1848 | 0.1041 |
| 711-720 | 6.6775 | 0.1009 | 0.8754 | 3.2612 | 1.2891 | 0.1147 |
| 721-730 | 6.6369 | 0.0989 | 0.8346 | 3.1752 | 1.2558 | 0.1060 |
| 731-740 | 6.6885 | 0.0927 | 0.8440 | 3.1977 | 1.1655 | 0.1424 |
| 741-750 | 6.5779 | 0.1147 | 0.8479 | 3.1309 | 1.1549 | 0.1335 |

## Interpretation limits

- 30 Hz learner decisions; rates use pooled sums, not averages of rates.
- Goals/concedes native. Finishing success is exactly scored goals, not on-target projection.
- Challenge/control and defensive clear are existing role-specific predicates, not independent possession/save proof.
- Training combines standing and momentum-assisted starts; race-first-contact is not possession.
- Descriptive on-policy windows, not the same fixed states. Fresh physical entry at681 and episode-age/scenario visitation can confound trends.
- Events and endings may straddle window edges. First touches divided by endings is NOT an attempt success rate.
- Do not substitute stochastic task rates for the completed deterministic full-match evaluation.

The JSON retains all five roles and both opponent families, raw counts,
exposure denominators, source hashes and final-closeout hash.
Use the final matched Nexto comparison for competitive conclusions.
