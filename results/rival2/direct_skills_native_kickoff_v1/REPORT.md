# Rival's opening approach: slower, not inactive

This is a completed read-only development diagnostic, not a gameplay promotion.
The original prospective method is commit `52f996caf38a663d59b8936ff67933358be7248d`.
Its narrow serialization-recovery amendment is commit
`328b5cbb2c421c6ae3eeb76530a2f462329b1175`; no simulator, controller,
reward, policy, optimizer or observation implementation was changed.

## Result

Ten standard kickoff starts (five layouts, both sides), six seconds each,
for the parent650, corrected child10 and corrected child25. The corrected
native-v5 Nexto controller, seeds, policy argmax and lifecycle match the already
completed full evaluations. All three lose all ten initial first contacts.

| Measurement | Parent650 | Child10 | Child25 |
| --- | ---: | ---: | ---: |
| Rival pre-contact mean speed, uu/s |1456.625|1456.659|1456.449|
| Nexto pre-contact mean speed, uu/s |1637.292|1637.292|1637.292|
| Full-throttle command fraction |100%|100%|100%|
| Boost command fraction |100%|100%|100%|
| Reverse-throttle command fraction |0%|0%|0%|
| Rival native flips before first contact |0/10|0/10|0/10|
| Nexto native flips before first contact |10/10|10/10|10/10|
| Rival contact lag, physics ticks |5–19|5–19|5–18|
| Changed pre-contact decisions vs parent |—|3/670|12/670|

Speed is first averaged over the same pre-first-contact ticks for both cars in
each world, then across worlds. It is not a whole-match movement average.
At 120 Hz, the parent lag is 41.7–158.3 ms. This measures Rival's own first
contact after Nexto has already moved the ball, not hypothetical arrival at a
stationary ball. The speed deficit is about 11.0% during this opening race.

Rival is largely aligned with the ball: per-world mean heading-dot-ball-direction
is about 0.962–0.995. At Nexto's first contact it remains about 360–911 uu from
the ball. Rival has no early native flip in any of the 30 starts; Nexto's first
native jump is at physics tick61 and native flip at73 (roughly0.5/0.6 seconds).
Several Rival starts stay grounded until contact; others jump only near the end
of the approach. These are native state flags, not new mechanic classifications.

Holding boost does not mean boost is available throughout. In the parent,
six of ten Rival cars have exhausted it by the first-contact boundary. The trace
retains reserve, ground state, every wheel's contact, full rotation and angular
velocity so those distinctions are inspectable.

The +25 block barely changed the opening control sequence:12 of670 pre-contact
decisions differ, while641 of1800 decisions differ across the full six seconds.
This is closed-loop sequence similarity, not a comparison of logits on identical
observations, and not evidence the weights or optimizer failed to update.

## Interpretation and next intervention

The observed opening failure is an acceleration/approach deficit. It is not a
failure to press throttle/boost, persistent reversing, or failure to aim toward
the ball in these starts. It does not by itself explain every full-match loss.
Nexto's early flip is associated with the faster trajectory; no intervention was
performed to prove that copying that input alone solves Rival's kickoff.

The current kickoff bonus called `first_touch` is each player's first personal
contact, not the first contact in the race. `SkillEvents.calculate()` in
`rivalsim/direct_skills_v1.py` pays that bounded bonus even after the opponent
has already touched. Later physical-control and goal outcomes remain distinct.
This is a documented design, not another lifecycle bug. The trace motivates a
small outcome-focused kickoff learning arm rather than more payment for
throttle, boost, speed, or a prescribed flip.

Recommended next concrete experiment: retain the unified entity policy,
corrected native Nexto, natural play, shooting and all other skill roles. For
the kickoff role only, prospectively test a one-shot **race-first-contact**
bonus in place of the personal-first-contact bonus, with the existing larger
control/advancement and true goal objectives intact. Include physically valid
near-tie/advantaged kickoff approach starts as a bridge to the unchanged five
standard starts. No scripted controller prefix, fake-kickoff label, blanket
action penalty, named-mechanic reward or inference router. A good second-touch
control outcome must still be worth more than merely being first.

Freeze the exact modification, bounded start distribution, parent, sample budget
and early evaluation before that learning arm. Keep shooting20% and preserve
normal-play checks. Use original parent650 as the controlled comparison root;
do not promote the regressed child25 or call child10 a champion on this small
development sample. This document recommends the next arm; it is not its final
training authority and starts no learning.

## Integrity and scope

- All three captures:720 physics ticks,180 policy decisions per world,10 worlds.
- Exact replay of all5400 captured observation/hidden-input pairs reproduces
  every controller output and hidden output at the original batch size.
- All emitted native controls equal scheduled controls; four-tick holds exact.
- Both policy models and all three checkpoint files unchanged; all captured
  values finite; no optimizer steps or reward edits.
- All goal prefixes agree with prior full evaluations, **but each prefix is
  empty**. Do not overstate this as an independent full-trajectory proof.
- Parent attempt1 completed raw capture but summary serialization failed on a
  NumPy Boolean. Its raw archive/receipt remain in Git. The explicit permitted
  attempt2 reproduces **every array byte** from attempt1. No silent retry or
  partial result was reported as complete.
- Twelve focused capture/evaluator tests and two reducer tests passed. CPU
  reduction was run twice with byte-identical `analysis.json`.
- No closed-loop training, model selection, new physics or Nexto investigation.

## Files

`protocol.json`, `serialization_recovery.json`: frozen method and operational
amendment, including checkpoint/source/prior-evaluation hashes.

`parent650_attempt2`, `child10`, `child25` each have `.npz`, `.json`, and
`.started.json`. NPZ contains every pre/post native field and actual controls;
JSON binds every array's dtype, shape and raw-byte SHA-256, model integrity,
contact timing, and concise half-second physical samples. `parent650.npz` is the
preserved incomplete-report attempt, not a separate candidate.

`analysis.json`: deterministic CPU reduction, per-world native timing and
cross-checkpoint controls; `report_tests.xml`, `recovery_tests.xml`, `tests.xml`.

Rebuild the report without GPU or learning:

```powershell
.venv\Scripts\python.exe -B benchmarks/report_direct_skills_native_kickoff_v1.py
```

Do not rerun capture commands as routine monitoring: these cases are complete.
