# Completed acquisition probe 137: further substantial regression

Unchanged read-only development protocol: 1,024 fixed acquisition starts, active
native-v5 Nexto, deterministic Rival, and native focal contacts in the original
episode. This is not a new full-match evaluation.

| Metric | Previously reported 127 | Newly completed 137 |
| --- | ---: | ---: |
| Easy contact within 5 seconds | 223/512 (43.5546875%) | 57/512 (11.1328125%) |
| Varied contact within 8 seconds | 120/512 (23.4375%) | 104/512 (20.3125%) |
| Easy median first contact, successes only | 0.783333 s | 2.808333 s |
| Varied median first contact, successes only | 2.125000 s | 2.641667 s |

Easy success fell by 166 attempts and varied success by 16. This is the second
consecutive probe with lower success in both groups, with a particularly severe
loss of easy-contact performance. Both conditional medians increased; because
their successful subsets differ, this is not a paired speed comparison. All
failures remain in the original tick arrays. Easy cases had 92 contacts within
the full 8-second diagnostic, but only 57 within the required 5-second deadline.

The temporary acquisition curriculum stays active and has no passing streak.
There is no new full-match evidence of recovery from the already published
update-100 regression. These results warrant pausing for a focused diagnosis;
they do not establish its cause. The monitor has not imposed a new performance
stop threshold, changed training semantics, or interrupted a healthy process.

## Verification and exposure

- Checkpoint SHA: `ABAF4DB992E867B5E676F0C4C3E8A7E211EBB21956B8D0E79A0EFB4523412FCE`.
- Raw probe SHA: `6362EDFB849AAE0B363E3840FCBB2A970EE9BD62F1F21EEAC3A1066ABC8E99E7`.
- Scenario SHA: `E331A5A5EEB6BE101945647E246FF9FDEAEEE7B5F081CB722B2248FB99B2925F`.
- Specification SHA: `3D7565AA4BE4A12352FF8D909513FCF03ADD096FC4B5BCFF6EA037A36526D1B2`.
- Completed UTC: 2026-09-08T15:43:49.530392Z.
- Original and stationary-kickoff authorities verified against local and remote
  source/evidence bytes. Checkpoint amendment identity remains correct from 76.
- All 16 CPU lineage, contracts, finite-state, Adam/counter, exposure and closed
  update-prefix checks passed. KL remains telemetry only.
- Result, receipt and actual checkpoint hashes agree. All 1,024 native-contact
  tick values, counts, fractions and conditional medians were checked on CPU.
- Evaluation reports zero optimizer steps and unchanged checkpoint/model/Nexto.

Across updates 128-137, acquisition-started episodes received 57.4671529132% of
983,040 world-seconds. Actual per-family/opponent exposure is retained in the
closed through_000137.json and is distinct from the nominal 50% reset-bank share.

The snapshot is saved before its scheduled probe; retirement metadata therefore
names the previously completed probe 127, as expected. No STOP/failure file was
present and current/original campaign stderr logs were empty at inspection.
Only this report, the immutable snapshot, completed result/receipt, CPU audit and
closed prefix are published. No extra GPU evaluation, optimizer work or training
configuration change was performed by the monitor.
