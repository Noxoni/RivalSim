# Completed acquisition probe 147: easy-contact regression continues

Same fixed 1,024 development starts, active native-v5 Nexto, deterministic Rival,
and native focal first contact in the original episode. No new full-match result
is included; at inspection the scheduled update-150 match evaluator was running.

| Metric | Previously reported 137 | Newly completed 147 |
| --- | ---: | ---: |
| Easy contact within 5 seconds | 57/512 (11.1328125%) | 39/512 (7.6171875%) |
| Varied contact within 8 seconds | 104/512 (20.3125%) | 104/512 (20.3125%) |
| Easy median first contact, successes only | 2.808333 s | 2.583333 s |
| Varied median first contact, successes only | 2.641667 s | 2.433333 s |

Easy acquisition lost another 18 successes, its third consecutive decline. Varied
success count was unchanged; equal counts need not be the same successful states.
Slightly lower medians among different successful subsets are not evidence of
recovered acquisition. Of 512 easy starts, only 51 contacted at all within the full
8-second diagnostic and just 39 met the 5-second deadline. Failures remain in the
saved first-contact tick arrays. This is not evidence of reliable possession,
kickoff performance, scoring or recovery in normal matches.

The removal thresholds remain unmet and the temporary curriculum stays active.
A pause for focused diagnosis remains the recommendation; no new performance
stop threshold or autonomous training-setting change was imposed by this monitor.

## Verification and actual exposure

- Checkpoint SHA: `639C702B4046EDDB100844826666C18F60CAE539D99A09C96000C3985932311D`.
- Raw probe SHA: `C7AAE22DF8234C68C73E748EED00A13CEAC6CF891A3CF19F5A01900B359000F6`.
- Scenario SHA: `E331A5A5EEB6BE101945647E246FF9FDEAEEE7B5F081CB722B2248FB99B2925F`.
- Specification SHA: `3D7565AA4BE4A12352FF8D909513FCF03ADD096FC4B5BCFF6EA037A36526D1B2`.
- Completed UTC: 2026-09-08T15:49:07.773714Z.
- Original and stationary-kickoff authorities reverified against local/remote
  bytes; checkpoint amendment identity remains correct from update 76.
- All 16 CPU lineage/contracts/finite-model-and-Adam/counters/closed-prefix audit
  checks passed. KL is still telemetry only.
- Result, receipt and actual checkpoint hashes agree. All 1,024 contact ticks,
  success fractions, failure fractions and conditional medians verified on CPU.
- Model, checkpoint and Nexto unchanged; zero evaluation optimizer steps.

During updates 138-147, acquisition-started episodes occupied 62.0236477320% of
983,040 world-seconds, not the nominal 50% source-bank fraction. The closed prefix
records actual per-family/per-opponent exposure for every update.

No STOP/failure file was present; both original and resume stderr files were empty.
Healthy training and its scheduled evaluation were not interrupted, and no extra
GPU work was launched. Only this report, immutable checkpoint, completed probe and
receipt, CPU audit and closed prefix are published. The later full-match evaluation
is not treated as complete or silently rerun by this publication.
