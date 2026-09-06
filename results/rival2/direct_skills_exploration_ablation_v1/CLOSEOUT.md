# Paired closeout procedure

The running training/evaluation source and frozen authorities are unchanged.
This additional CPU-only packaging tool is not imported by either worker and
does not collect rollouts, construct optimizers, update weights or run matches.

After both arms and the supervisor have actually exited, run:

```text
.venv/Scripts/python.exe benchmarks/finalize_direct_skills_exploration_ablation_v1.py
```

It requires both accepted +30 boundaries, complete contiguous curves, the
exact common750 parent, frozen coefficients and paired exposure. It reruns
only the existing CPU file/checkpoint audits, verifies final rolling/permanent
checkpoint equality, compares both +10/+30 evaluations, and checks external
logs/state stay unchanged during capture. It refuses to finalize if a live
campaign process or failure artifact exists. KL remains telemetry only.

Artifacts: completion.json; comparison_000010.json/comparison_000030.json;
each arm's final state/latest/closed curve/stdout/stderr; supervisor logs/state.
Final checkpoint artifacts and all four match records/integrity files must
also be committed and remotely read back. A complete-review status is not a
gameplay promotion or SSL verdict; interpret the actual scorelines and
contact evidence before deciding what training to do next.

The isolated finalizer tests pass13/13, including rejection of premature,
wrong-parent, cross-arm, inconsistent, extra-update and live-worker closeout.
This is validation of packaging behavior, not a new capability evaluation.
