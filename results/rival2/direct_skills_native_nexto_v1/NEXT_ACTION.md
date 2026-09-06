# Current handoff: completed block, inspect Rival's failed kickoff approach

The25-update block and both corrected-opponent evaluations are COMPLETE.
Worker43740 exited normally after `complete_review` at2026-09-06T15:14:24Z.
Do not resume this completed directory, remove STOP markers, or interpret the
absent worker as an operational failure needing an automatic restart.

Results were published at `91d5eca6aa7b695edb9203a4586635310fccbd25`; all14 commit
files were read back, with exact raw SHA checks for checkpoint/JSON/JSONL/logs.
Sixteen completion/reporting tests passed. The goal turn made real progress by
completing the experiment and preserving evidence that changes the next action;
the bot's playing strength did not demonstrate sustained improvement.

| Checkpoint | Wins | Goals for/against | Contacts/min | Same-player next contact |
| --- | ---: | ---: | ---: | ---: |
| Parent650 |0/10|7/196|9.12|27.70%|
| Child10 |0/10|10/198|10.80|33.59%|
| Child25 |0/10|2/203|9.48|24.04%|

All three have zero kickoff first contacts. Preserve all three; do not promote
child10 as a champion simply because it is the least bad of this small set.
The final checkpoint is `checkpoints/rival2/direct_skills_native_nexto_v1/child_000025.pt`,
SHA `D53603E3B3DBD990EBC06E44179932C33F03358F4B4300098E2CAD1BD523B29D`.
See `REVIEW_000025.md`, `completion.json` and both progress files for full detail.

## Next bounded action, not another broad compatibility investigation

Implement and freeze a six-second (720physics-tick), ten-world kickoff trace
of **Rival's own policy behavior** for the parent650, child10 and child25.
Reuse the existing corrected evaluator, its five standard layouts/both sides,
native-v5 Nexto mode/seed and raw deterministic Rival argmax. Record actual
applied controls, pre/post car/ball state, heading/angular motion, boost,
ground/jump/flip state and contact/goal timing. Capture real model observations
at the30Hz decision boundary; do not change them or prepend scripted controls.

The question is why Rival loses the initial race/approach: weak throttle/boost,
steering/heading error, mistimed jump/flip, or another physically observed cause.
Do not label a mechanic or invent a reward from the trace. Bound interpretation
to these initial six seconds, excluding post-goal restarts when describing the
initial kickoff. Check goal-event prefixes against the existing full match and
model/action provenance so instrumentation is not silently changing the policy.
Commit the exact method, checkpoint/source hashes and finite scope before running.
Run one case per checkpoint under the existing GPU lease, no simultaneous PPO.

Use the result to choose a specific learning intervention. Do not spend another
large sample block solely because tensor guards passed, and do not redefine the
SSL goal around this diagnostic. No learning intervention is frozen here yet.

## Already complete; do not repeat

Native games, recorder/bridge/observation/timer audits, installed Nexto identity,
controller timing comparisons, production controller oracle/GPU verification,
parent baseline selection and kickoff **source-start admission** check.
`KICKOFF_START_CHECK.md` confirms the starts are present and initial admission
is enabled; it is not proof Rival makes good decisions. The native-v5 opponent
correction stays intact. Do not reuse legacy Nexto scores as learning evidence.

The existing monitor's external cursor records child25 as reported and points
here. Keep notifications quiet unless a new result, problem or required user
action exists. The SSL development goal remains active, not complete or blocked.
