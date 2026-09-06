# Prospective650 action-selection diagnostic

Question: sampled training scoring improves while greedy evaluation scoring
regresses. Does the exact distribution PPO trained perform differently from
greedy execution in the same fixed evaluation situations?

Use only accepted650 SHA256
`4FCD41C2C8305B0EFF448ED2E78ECAD4BDC447D87C740EAE79EEBBD4621EF09F`.
Compare the already completed deterministic650 results with **three** predefined
sampled evaluations: action seeds20260906651,20260906652,20260906653. Each runs
the same64finishing starts and same ten regulation matches with original bounded
overtime. Physics/scenario/Nexto seeds, cadence and reset method stay unchanged.
The private action generator restarts from the declared seed for each mode.

Only action selection changes: categorical sampling of raw logits divided by2,
exactly the distribution currently used by PPO. No temperature sweep, new model,
reward change, task indicator, rollout-training update or optimizer. All three
results must be retained. No checkpoint/seed selection or automatic inference
promotion follows. This is a development diagnosis, not an SSL test.

The diagnostic subclass uses the harness's historically named `deterministic`
method slot to return sampled native table controls. Its name is not a claim of
greedy execution. Forward logits/hidden and checkpoint state schema remain exact;
the original production policy/evaluator are not modified. Explicit output
action-mode metadata and a different runtime-package/authority identity prevent
these outputs from passing the ordinary same-method learning comparator.

Focused tests verify exact use of the trained categorical distribution and
action table, private RNG replay, unchanged weights/forward/hidden/schema,
nonfinite refusal and fixed no-selection/no-learning scope. Full native outputs
must additionally verify unchanged model/checkpoint, complete score accounting
and recurrent goal-reset consistency. No observer timeout authorizes rerunning
a completed seed. Any failed attempt is retained and investigated explicitly.

```powershell
.venv\Scripts\python.exe benchmarks/diagnose_direct_skills_sampling_000650.py verify
.venv\Scripts\python.exe -u benchmarks/diagnose_direct_skills_sampling_000650.py run
```

Run only after publishing and remotely verifying the authority, source, tests
and650review/checkpoint. The same exclusive GPU lease prevents overlap with a
learner. Training remains paused at650. The existing STOP is not cleared for
this read-only diagnostic. After all three seeds, report aggregate and per-seed
scoring/contacts, finishing goals versus projected on-target proxies, and the
limits of the comparison before deciding on a prospective learning change.
