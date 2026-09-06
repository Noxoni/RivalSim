# Finishing-only25 block: mixed capability, healthy training

Completed exactly25accepted PPO updates from preserved recovery child25/direct625,
under prospective commit3b7cae5c85d12c0a8cfbb50b7df5a48be9c8e955. Worker41004
exited normally at2026-09-06T12:15:03Z; campaign_state is complete_review.
No further PPO is running in this branch. No live deployment or SSL claim.

Final checkpoint: checkpoints/rival2/direct_skills_finishing_goal_v2/child_000025.pt

SHA256: `939213BF57FFC751C7C173FFCC8AC34DE84D2CAA66CFECB79C59792CBABB0F6D`

Research ancestry625->650, distinct from old direct650/675files. Added3,540Adam
steps, final144,636; added110,592,000learner decisions and294,912,000physics
world-ticks. Model/Adam/fourRNG/opponent state/counters/authority/contracts saved.
Source625checkpoint,600reference, prior675 and all other candidates preserved.

## Same fixed V3 Nexto development matches

Ten five-minute matches, five original layouts/both sides, deterministic joint90
argmax. Same handbrake-reset V3 runtime and original seeds, not easier starts.

| Metric | Reference600 | Immediate parent625 | Finishing child650 |
|---|---:|---:|---:|
| Wins/losses |8/2|7/3|8/2|
| Goals for/against |169/122|124/112|131/96|
| Goal difference |+47|+12|+35|
| Rival contacts |486|805|667|
| Rival contacts/min |9.72|16.10|13.34|
| Same/resolved followups |50/311|413/684|268/538|
| Same-player followup fraction |16.08%|60.38%|49.81%|
| Rival first kickoff contacts |0|0|0|
| Matches without Rival contact |0|0|0|

Seven paired goal differences improve and three worsen versus immediate625.
Against600, three improve, one ties, six worsen. This candidate scores more and
concedes less than625; it has fewer repeat contacts. It concedes less than600
but also scores less, with a lower aggregate goal difference. Neither clearly
dominates all metrics. Repeat contacts do NOT measure possession duration.
No-touch resets are disabled by the full-match protocol; their zero is not a
learned achievement. Zero native Rival demolitions are recorded in this set.

Per-case goals/concedes, immediate625 -> new650:

| Side/layout |Parent625|New650|
|---|---:|---:|
|Blue0|14/12|15/10|
|Blue1|14/12|17/9|
|Blue2|13/12|15/10|
|Blue3|13/11|15/9|
|Blue4|13/11|16/10|
|Orange0|15/10|11/9|
|Orange1|12/11|14/11|
|Orange2|10/11|14/9|
|Orange3|10/11|7/10|
|Orange4|10/11|7/9|

## Original skill cases: actual native outcomes, not proxies

Each64-case family ends at first goal or timeout. The evaluation reward/event
implementation is intentionally unchanged; old projected-shot success counters
remain diagnostics, NOT this new finishing training reward.

| Family |Parent goals/concedes/timeouts|Child goals/concedes/timeouts|
|---|---:|---:|
|Natural|38/22/4|37/26/1|
|Challenge|0/38/26|1/38/25|
|Finishing|6/36/22|7/44/13|
|Defense|7/23/34|6/21/37|
|Kickoff|64/0/0|64/0/0|

Finishing: all64still contacted, on-target raw projections53->45, totalcontacts
114->94. One extra goal alongside eight extra concedes is NOT a convincing
finishing improvement; ten paired outcomes improve, eighteen worsen. No-touch
acquisition is not the main hard-shot bottleneck.

Natural contains35kickoffstarts that all score, not64independent ongoing games.
The29ongoing-ground cases worsen from3goals/22concedes/4timeouts to2/26/1,
with22touched cases and55contacts. Aggregate natural results must not hide this.
Defense touches62->64 and shot-clear proxies36->37, but attack goals7->6.
Challenge control-gain proxies11->8 and forward-control6->7; no broad possession
competence established. These fixed development cases do not establish ranked
performance or transfer to native Rocket League.

## Training and integrity

All31checkpoint/curve/reward invariants and all11fullmatch integrity checks pass.
28focused final report tests pass (in addition to prelaunch13native/30runner
tests,13no-step checks and24actual first-update audit checks). All25curve rows
are retained. No nonfinite/corruption failure or KL rejection occurred.
Maximum completed-update mean KL0.0024422061285090907; maximum individual sample
KL52.548606872558594 is telemetry, NOT a newly imposed safety failure. Peak
allocated GPU memory16,321,048,064bytes. No changed hard limit or rollback rule.

The live new finishing reward is verified in every row: approach0,
direct_reward=-failed_attempt, success_endings=goals. First-touch and projected
shot counts remain raw/unpaid. Late updates13..25 finishing-vs-Nexto include
12,939episode endings:1,331goals,8,177concedes,3,431timeouts. All success endings
are those1,331actualgoals, not10,259projected-shot events. Early1..12 includes
the declared fresh-episode transient and is not an identical sampling window.
Learner contacts/min18.015->18.777 and speed1189.203->1194.749uu/s are training
telemetry, not deterministic skill improvement. No outcome-dependent rows omitted.

## Decision

Preserve this candidate and600/625; do not overwrite a best/official/live model.
Do not automatically extend the expired25-update block or retune its reward.
The goal-only finishing graduation has not solved defended shooting, although
the full-match result recovered relative to625. A single trajectory cannot prove
which changes caused that recovery.

Next, build the missing native entity/joint90 recurrent30Hz export/runtime path
and audit its observation semantics before relying further on simulator wins.
LIVE_READINESS.md and the read-only loader probe identify a real compatibility
gap: old V5 hybrid/120Hz exporter/runtime cannot run this model faithfully.
This does not mean the checkpoint is corrupt or that all simulator training was
invalid. Keep this phase no-optimizer and no automatic live installation; prepare
an honestly labeled native-play candidate with source hashes and sequential
action/hidden/cadence tests. The continuing SSL goal remains active and unmet.

Detailed31checks, same-method comparisons, every raw goal/contact, training-role
counts and checkpoint identities are in review_child_000025.json, both evaluation
JSONs, integrity_child_000025.json and the complete25-row training curve.
