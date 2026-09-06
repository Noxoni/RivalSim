# Reviewed continuation: test whether late recovery becomes net improvement

The completed exploration experiment ended at +30, with worse aggregate
performance than its control: 2/228 scored/conceded versus 7/196, and 0/10 wins.
It is not promoted. However, +15 to +30 improved goal difference in all ten
paired matches, with contacts recovering from 6.64 to 8.88 per minute. A
subsequent no-update probe measured the exploration gradient at 9.67% of the
PPO gradient median, not a dominating term. The saved critic analysis also did
not establish a stuck critic as the sole cause.

Hypothesis: the broadened policy needs further accepted reward-learning updates
for that late recovery to become useful gameplay. This is uncertain. A short
continued run is a more direct test than changing multiple settings without
evidence. This plan is an explicit new decision after review, not an extension
silently inserted into the completed 30-update authority.

## Parent and unchanged training

- Exact parent: `checkpoints/rival2/direct_skills_log_barrier_v1/child_000030.pt`.
- SHA-256: `16BF2B904785D49E0363B869AF953CFCCC62FF754286A307D386729AC691DDAD`.
- Cumulative update 680, Adam step 148930. Restore exact model, Adam groups and
  moments/counters, four learner RNG states and native-opponent RNG/counters.
- Fresh physical episodes and zero recurrent hidden. Native episode caches
  reset explicitly; this is not an exact physical-world continuation.
- 32768 worlds; 120 Hz physics, 30 Hz decisions, four-tick hold; horizon 90;
  two epochs; actor LR 1e-4, critic LR 3e-4; gamma .995, lambda .9973145188572297.
- Same entity/recurrent joint90 model and independent critic. T2 sampling and
  likelihood; original raw-argmax deterministic evaluation.
- Same beta .01 uniform log barrier and entropy .001. No loss or reward change.
- Same scenario bank: 30% natural, 20% challenge, 20% shooting, 15% defense and
  15% kickoff starts. Actual goal-based shooting reward; no retired shot bonus.
- Same 50/50 self-play/native-v5 Nexto worlds, one-third of learner decisions
  against Nexto. No opponent targets in PPO, task ID, scripted Rival or router.
- Existing finite/corruption protection. KL telemetry only; no new rejection.

## Bound and evaluations

At most **70 additional updates / 309,657,600 learner decisions**. Final
cumulative update 750, total exploration updates 100. Preserve an entry, first
accepted checkpoint, each evaluation checkpoint, and alternating rolling
checkpoint after every accepted update.

| This continuation | Total exploration | Cumulative ancestry |
| ---: | ---: | ---: |
| +20 | 50 | 700 |
| +45 | 75 | 725 |
| +70 | 100 | 750 |

Use the same ten complete native-v5 Nexto matches at those boundaries. Compare
both the immediate +30 parent and the original control650. No easier start,
new benchmark, checkpoint reselection, or claim that exploration/finite tests
prove skill. Report scoring/conceding, contacts and kickoff acquisition. Same
player next-contact identity is not continuous possession.

Stop at +70 for review; no automatic extension. Respect user STOP and existing
numerical/corruption protection throughout. No reward/parameter retuning inside
the block. If recovery fails to produce net competitive improvement, this arm
does not establish a useful solution and must not be promoted.

## Execution and validation

21 focused tests passed; the 1024-world / 90-decision backward-only preflight
passed all 14 checks with zero optimizer steps and unchanged parent/model/Adam.
The initial reduced-world controller-cache setup issue is preserved separately.
Freeze, commit, push and remotely verify package/source identities before run.

```powershell
.venv\Scripts\python.exe -B benchmarks/run_direct_skills_exploration_continuation_v1.py verify
.venv\Scripts\python.exe -u -B benchmarks/run_direct_skills_exploration_continuation_v1.py run
.venv\Scripts\python.exe -B benchmarks/report_direct_skills_exploration_continuation_v1.py 1
```

Reporter boundaries: 1, 20, 45, 70. It checks new parent-relative counters,
exact native RNG lineage, unchanged reward and the separately bound objective.
Do not reuse the old auditor's hardcoded parent650 or old finalizer's +30 cap.
External state: `G:/dev/RivalSim-runs/direct-skills-exploration-continuation-v1`.
Use the existing exclusive GPU lease; never overlap jobs or clear unknown STOP.
The overall SSL goal remains active, not achieved by completing this block.
