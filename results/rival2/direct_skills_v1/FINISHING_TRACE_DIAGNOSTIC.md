# Bounded read-only finishing diagnosis

Prepared after the scheduled +200 review, without modifying any of the twenty
frozen campaign source files, reward/curriculum authority, policy or optimizer.
The learner remained active. **Live replay has not yet been run or validated.**

## Why this exists

The fixed finishing test declined from 9 goals / 42 concessions at +150 to
5 / 52 at +200, while the existing on-target task proxy rose from 48 to 56.
Previously scoring cases 5, 11, 16, 33 and 43 all still touched the ball at +200;
four then conceded and one timed out. The saved aggregate evaluation cannot
identify their contact geometry, controller decisions or goalkeeper response.
It is not evidence that more first touches or more proxy successes solved shooting.

The +200 review permits continuing the unchanged run to the scheduled +250
review because full-match scoring/concessions improved again. Persistent
finishing regression is a reason for focused trajectory diagnosis, not a reason
to invent a cause or silently retune rewards. This helper prepares that diagnosis.

## Method and boundaries

`benchmarks/trace_rival2_direct_skills.py` calls the **unchanged existing**
`skill_evaluation(model)` once: all five fixed 64-case families, identical seeds,
batching, deterministic Rival actions, Nexto cadence and recurrent reset handling.
Only the finishing environment is observed. A temporary subclass delegates every
transition to the frozen environment. A temporary wrapper delegates each native
`world.step(1)` call, reading before and after it, before automatic episode reset.
Both bindings are restored even on exceptions. Neither script modifies training
files, checkpoint bytes, reward parameters or physical state.

The output contains all 64 original finishing cases, not a favorable selection:

- At each 30 Hz decision: the exact policy observation, proposed Rival controls
  for both perspectives, original-case alive mask, pre-reset transition
  observation, termination/truncation/reset, native counters and existing task
  event/reward accounting.
- At each 120 Hz physics tick: the actual applied eight-channel controls after
  Nexto substitution; ball position/velocity/angular velocity; both cars'
  position/quaternion/linear and angular velocity, boost, grounded state,
  jump/dodge timers and flags, individual wheel contacts, demolition state;
  native contact flags/counters and goal/scoring flags. No mechanic labels are added.
- Original case index and focal side, preserving Blue/Orange identities. Raw
  coordinates remain simulator-native; they are not silently canonicalized.
- Original-episode and pre-goal validity masks. Inactive post-reset rows and
  the remainder of an already goal-ended four-tick action interval are **not**
  additional valid samples. Success does not create an extra episode boundary.

The maximum is 360 decisions / 1,440 physics records, each holding 64 worlds.
`finishing.npz` is lossless, dtype-preserving, non-pickle NumPy storage. The JSON
manifest binds the checkpoint, saved evaluation, frozen runtime sources, helper,
archive hash, array schema and complete replay result. Raw physical contact flags
are not interchangeable with debounced native touch counters.

All five replay evaluation results must match the saved evaluation **exactly**,
including per-case outcomes and float summaries. No numeric tolerance is used.
The trace independently reconciles original-case goals, endings and contacts,
including per-tick contact counter reconstruction and absorbing-goal masks.
Archive write/read equality and checkpoint/model immutability are also checked.
A mismatch blocks interpreting that replay as the original result; it does not
authorize reward tuning, model changes or a new training lineage.

## Safe execution

Do **not** run this beside the learner or start a second GPU job. The helper uses
the same mandatory exclusive GPU lease, acquired **before** CUDA model creation
or output directory creation. It refuses while the worker holds the lease. It
does not stop the learner, remove STOP, restart training, or kill user apps.

If the +250 review calls for diagnosis, request a clean accepted-boundary pause,
preserve/hash the latest rolling checkpoint, verify the learner has exited, then
run the helper against the scheduled checkpoint being investigated. For example:

```powershell
.venv\Scripts\python.exe benchmarks/trace_rival2_direct_skills.py --update 200 --output results/rival2/direct_skills_v1/finishing_trace_000200
```

An existing output directory is never overwritten. A separate +150 replay can
provide the paired before/after comparison, but each must first reproduce its own
saved evaluation. This is a diagnostic replay, not another selection test or a
replacement for the scheduled +250 evaluation. No optimizer is constructed or
stepped. Resume only the preserved current rolling model/Adam/RNG afterward,
never the diagnostic +150/+200 checkpoint; the campaign's already-declared fresh
physical-episode/hidden-state resume semantics remain in force.

## Current validation status

**CPU-only: 38 tests passed**, including the adjacent existing report tests.
`trace_cpu_tests.xml` holds the result. New tests cover exact comparison, bounds,
buffer-copy/dtype/negative-zero archive roundtrip, tick ordering and duplicates,
original-episode and absorbing-goal accounting, nonfinite rejection, lease
refusal before model/output creation, and real tap delegation on a fake CPU
environment including per-tick opponent controls and pre-reset state. Exception
paths restore the original instance/class methods.

These tests validate diagnostic mechanics, **not live CUDA trajectory parity**.
No new shooting capability, improvement or causal explanation is claimed yet.
