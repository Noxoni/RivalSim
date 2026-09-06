# Corrected-runtime kickoff continuity at +675

This read-only replay exactly reproduced the saved ten complete match results,
summary and hidden-reset counts. It did not change the model, checkpoint,
controls, physics, rewards or optimizer. The authority was published in
`bdba262ff171e0a2ef587b4e21e43effcb249e7b` before execution.

The archive contains ten initial and 205 post-goal kickoff origins, each with
eight 30 Hz decisions. All **1,640 same-layout/side/local-age Rival action pairs
are identical**. Hidden state is zero at every origin. Initial position,
velocity, boost and all four wheel bits are exact. Quaternion rounding differs
by at most 5.96e-8; the four affected orientation observation fields differ by
at most 2.384e-7. The previously repaired stale-wheel observation issue is absent.

Subsequent motion is not identical. Maximum absolute differences, considering
both cars and all paired origins:

| Physics age | Position (uu) | Velocity (uu/s) | Observation |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 2.384e-7 |
| 4 | 2.776676 | 77.71756 | 0.47910637 |
| 8 | 5.008535 | 60.26073 | 0.39326668 |
| 12 | 6.601688 | 60.84106 | 1 |
| 16 | 7.535364 | 81.68230 | 1 |
| 20 | 8.453613 | 114.92923 | 1 |
| 24 | 11.88428 | 145.65762 | 1 |
| 28 | 17.33643 | 174.00250 | 1 |

At age four, self angular velocity differs by up to 0.598592 rad/s and opponent
angular velocity by 2.874638 rad/s. Later unit-sized observation differences
include opponent ground/contact/dodge bits. This is a discrepancy to isolate,
**not proof of another reset bug or an explanation of all learning regression**.

The tap samples every four physics ticks. Nexto's previous-action buffer matches
from age four onward, but that alone cannot prove all intervening applied
controls identical. Initial old-history and global neural-cadence differences
are not by themselves wrong opponent behavior. Layout is an event field that
clears to -1: the reducer binds it at each `(world, goal count)` origin, never
guesses later layout from the transient field.

Static follow-up found two distinct possible explanations to separate:

- Native startup intentionally uses special tick-zero quaternion and suspension
  solver semantics (`SOLVER_INITIAL_DT=1/60`, normal physics `1/120`). A fresh
  simulator is not automatically a physics-equivalent post-goal reference.
- Standard kickoff reset clears wheel-contact bits but leaves some other
  vehicle caches. The scenario reset clears handbrake, contact count and wheel
  world-contact caches too. Chassis world-contact normal is not explicitly
  cleared in either path and is read in the wheel/pre-solve code. This static
  finding is not yet causal evidence that it alters kickoff motion.

The next bounded diagnostic controls both players' exact per-tick actions and
separates cold-start clock semantics, quaternion rounding and cache groups in
private replay. No production reset, reward, Nexto, policy or training change
has been justified or made here.

## Evidence

- `trace/kickoff_reset.npz`: SHA256
  `EDF5523209EDC6B662781B97F033AF947B4EB97A73C817E9CF069DBB079E2FD1`.
- `trace/manifest.json`: complete source/runtime identities and exact replay.
- `report.json`: every paired index, age, field, native state and action delta.
- `tests.xml`: four focused CPU tests passed, including exact report rebuild,
  transient-layout handling and rejecting missing/invalid origins.
- Checkpoint `plus_000675.pt`: SHA256
  `77E6EF076F7073549F45168FA6FB34BE9A2D4C25E1580F17F7434A59195B32F8`.

Rebuild: `.venv/Scripts/python.exe benchmarks/report_direct_skills_kickoff_continuity.py results/rival2/direct_skills_v1/kickoff_continuity_000675`.
The +675 review STOP remains. No PPO update beyond +675.
