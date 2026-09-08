# Completed acquisition probe 127: regression in both groups

This is a newly completed read-only acquisition development check, not a full
match evaluation. It uses the unchanged 1,024 fixed starts, active native-v5
Nexto, deterministic Rival, and native original-episode focal contact criteria.

| Metric | Previously reported 117 | Newly completed 127 |
| --- | ---: | ---: |
| Easy contact within 5 seconds | 315/512 (61.5234375%) | 223/512 (43.5546875%) |
| Varied contact within 8 seconds | 230/512 (44.921875%) | 120/512 (23.4375%) |
| Easy median first-contact time, successes only | 0.650000 s | 0.783333 s |
| Varied median first-contact time, successes only | 1.200000 s | 2.125000 s |

There were 92 fewer successful easy attempts and 110 fewer successful varied
attempts. Both success rates regressed substantially, and conditional contact
times increased. Different successful subsets contribute to each median; these
are not paired same-state speed comparisons. All failure tick values remain in
the saved arrays. Six easy cases contacted after the 5-second deadline but within
the full 8-second diagnostic, so deadline failures and no-contact fractions differ.

The temporary acquisition curriculum remains active, with no passing streak.
There is no evidence of reliable contact acquisition and no new full-match result
in this publication. In particular, this does not show recovery from the already
reported update-100 full-match regression. It also cannot identify the causal
training or execution mechanism. A focused diagnosis is warranted; the monitor
did not alter reward, PPO, exploration, curriculum or policy semantics in response.

## Integrity and exposure

- Checkpoint SHA: `185D3656B39E406045E26AACDBABDFF4054E3C3135C0AAA30110A379219AC4B4`.
- Raw probe-file SHA: `7AB1F9BB8FFB920E4B15CDD2054C715B616DD8651A30C5707A59D49D64156A4F`.
- Scenario SHA: `E331A5A5EEB6BE101945647E246FF9FDEAEEE7B5F081CB722B2248FB99B2925F`.
- Specification SHA: `3D7565AA4BE4A12352FF8D909513FCF03ADD096FC4B5BCFF6EA037A36526D1B2`.
- Completed UTC: 2026-09-08T15:37:19.615588Z.
- Original and stationary-kickoff authorities match current files and origin/main.
- The checkpoint carries the correct stationary-kickoff identity from update 76.
- Model, Nexto and checkpoint were unchanged; evaluation optimizer steps were zero.
- Checkpoint/result/started-receipt hashes agree. All 1,024 first-contact ticks,
  counts, fractions and conditional medians were independently checked on CPU.
- All 16 CPU audit checks passed: lineage, contracts, update continuity, finite
  model/Adam, counters, Nexto mode and KL-as-telemetry semantics.

Closed through_000127.json retains actual per-family/per-opponent world-second
exposure. Across updates 118-127, acquisition starts accounted for 55.6231926810%
of 983,040 world-seconds, distinct from their nominal 50% source-bank share.

The permanent checkpoint is written before its scheduled probe, so its retirement
metadata names the previously completed probe 117. The running campaign subsequently
records probe 127 with streak zero and retired=false. This expected ordering does
not imply the probe evaluated the wrong policy.

The process was running, STOP/failure were absent and both campaign stderr files
were empty at inspection. No extra GPU evaluation or optimization was launched;
the saved raw evidence was validated without interrupting healthy training. This
publication contains only the immutable checkpoint, completed result and receipt,
CPU audit, closed prefix and this report, not live logs or unrelated work.
