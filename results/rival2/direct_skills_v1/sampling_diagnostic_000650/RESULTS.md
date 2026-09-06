# Sampling does not rescue the650 checkpoint

All three prospectively frozen seeds completed once, with zero optimizer steps.
The worker exited at2026-09-06T10:22:31Z. Training remained intentionally paused
at650 throughout; the agent review STOP was not cleared. No failure artifact,
nonfinite error, checkpoint mutation, reward change or deployment change occurred.

Authority/source/tests and650 checkpoint were published at
`9ea1e437ee1574ffa136a54e2eafa0ec96ce631f`. All29 files from that commit were read
back from origin/main before the diagnostic completed (Git-normalized text
identities and binary checkpoint identity). Its entry
verifier also compared prospective authority/source/test bytes to origin/main.

## Fixed regulation matches

Same ten starting layouts/sides, original physics seeds, corrected native reset,
Nexto and overtime rules. The only intervention is the private650 evaluation
copy's action selection. Categorical logits/2 is the exact distribution trained
by the frozen T2 PPO collector; no temperature search was performed.

| Action mode / seed | Rival goals | Nexto goals | Wins | Contacts |
|---|---:|---:|---:|---:|
| Existing greedy650 |42|139|0/10|577|
| Sampled20260906651 |37|149|0/10|617|
| Sampled20260906652 |35|146|0/10|641|
| Sampled20260906653 |39|145|0/10|689|
| Sampled mean |37|146.667|0/10|649|

Each sampled seed has fewer Rival goals and more concessions than greedy650.
Sampling increases contacts: mean12.98/minute versus11.54/minute. Pooled
same-player follow-ups are43.86% versus39.21%; this is not possession duration.
All matches contain Rival contacts. No-touch resets are disabled by this match
protocol; their zero count is not a learned success.

## The same64 finishing starts

| Action mode / seed | Goals | Concedes | Timeouts | Contacts | On-target proxy |
|---|---:|---:|---:|---:|---:|
| Existing greedy650 |5|41|18|102|53|
| Sampled20260906651 |5|40|19|130|54|
| Sampled20260906652 |7|33|24|114|52|
| Sampled20260906653 |8|35|21|116|54|
| Sampled mean |6.667|36|21.333|120|53.333|

Every mode touches all64 balls. Shooting sees modestly more goals and fewer
concessions with sampling, but remains weak in every seed. In particular, roughly
53 projected on-target contacts are not53 successful shots. The current proxy
projects toward the goal plane without predicting the keeper or wall bounces.
It can be paid on a ball that Nexto saves. This is a documented shaping proxy,
not evidence of a broken goal reward or measured shot quality against a keeper.

## Integrity and interpretation

`report.json` records all raw-file hashes, all three per-case comparisons,
finishing outcomes, aggregate ranges/means and45 checks (15 per seed), including
native score conservation and recurrent goal resets. The existing same-method
learning comparator correctly REFUSES these sampled-versus-greedy comparisons.
Checkpoint SHA remains
`4FCD41C2C8305B0EFF448ED2E78ECAD4BDC447D87C740EAE79EEBBD4621EF09F`.
The original frozen runtime source hashes and baseline hashes reverify unchanged.
The CPU reducer rebuilds exactly;15 focused tests pass and their XML is retained.
An initial test invocation named a nonexistent comparator test file and collected
zero tests; its collection-error XML is preserved. Correcting the command to the
existing `tests/test_direct_skills_matches.py` required no production code edit.

This rules out a simple claim that the650 policy already plays well but greedy
execution is hiding it on these cases. It does not establish why sampled
training rates improved while regulation outcomes regressed. The three action
seeds reuse the same fixed development cases, not30 independent starting
populations. There is no statistical or SSL capability claim and no selection
of the best seed, older checkpoint or new deployment mode.

## Next development direction

Shooting already occupies20% of source starts. Those starts put an active
opponent between the ball and goal; they do not include a graduated empty-goal
and recovering-defender progression. Adding a duplicate shooting family is not
the missing change.

Next, compare the fixed650 policy on a bounded, prospectively specified shooting
difficulty ladder (unpressured finish, recovering defender, established keeper)
using actual goals and contact-to-goal outcomes. This should determine whether
Rival cannot finish an accessible shot or mainly cannot beat/recover possession
from a set keeper. Only then choose a small shooting-curriculum correction with
an unchanged full-match reference. Preserve650 as the source; do not silently
change rewards, select600 after this diagnostic, add another exploration sweep,
or launch another unchanged50-update block. The SSL goal remains active and unmet.

Rebuild without new simulation:

```powershell
.venv\Scripts\python.exe benchmarks/report_direct_skills_sampling_000650.py
.venv\Scripts\python.exe -m pytest -q tests/test_direct_skills_sampling_report.py
```
