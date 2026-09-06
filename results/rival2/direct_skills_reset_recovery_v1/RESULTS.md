# Child25: more repeat contacts, but no finishing breakthrough

The separately frozen branch completed25accepted PPO updates and both original
evaluations. Process18848 exited after `complete_review` at2026-09-06T11:45:38Z;
stderr is empty. No learning beyond child25. This is a normal review boundary,
not a KL failure or user cancellation. The SSL goal remains unmet.

## Checkpoint and training integrity

`checkpoints/rival2/direct_skills_reset_recovery_v1/child_000025.pt`

SHA256 `8EBC09AF738CA89554854D9854EDAE46E01083A9B2841F9E0428D5A054E85AA3`.

This child has cumulative direct ancestry625, NOT the old675lineage. Exact600
model/Adam/fourRNG was preserved at entry. The branch adds110,592,000learner
decisions,294,912,000physical world ticks and3,538Adam steps; cumulative direct
exposure2,764,800,000decisions/7,372,800,000ticks and141,096Adam steps. Earlier
entity-stage exposure remains separately recorded in the checkpoint.

All27final checkpoint/curve/contract checks,11full-match integrity checks and
35focused CPU tests pass. Parent600, old675 and the observation/action/architecture/
PPO/opponent/reward identities are unchanged. New scenario geometry and corrected
runtime are explicitly bound, not hidden changes to historical packages.
Max completed-update mean KL0.0029863838; max sample estimate10.75134087.
KL is telemetry only, zero KL rejections. All model/Adam/PPO outputs are finite.
Peak CUDA allocation16,321,086,976bytes. Full25-row curve is committed, no missing
or rejected update disguised as accepted. These are integrity results, not SSL.

## Original full Nexto matches, same V3 method

| Metric | Parent600 | Child25 |
|---|---:|---:|
| Wins / losses |8 /2|7 /3|
| Goals / concedes |169 /122|124 /112|
| Goal difference |47|12|
| Rival contacts |486|805|
| Rival contacts/min |9.72|16.10|
| Nexto contacts |1315|1141|
| Same-player / resolved follow-ups |50 /311|413 /684|
| Same-player follow-up fraction |16.08%|60.38%|
| Rival kickoff first contacts |0|0|
| Matches without Rival contacts |0|0|

Three matched goal-difference cases improve, seven worsen. The child touches the
ball more and follows its own contacts much more often, but scores less. This is
mixed behavioral adaptation, not proof that nothing learned and not proof of
better match performance. Same-player follow-ups are NOT measured possession
duration, intentional fakes, advanced mechanics or shot quality. Fewer total
goals also mean different numbers of kickoff-origin segments; do not treat
aggregate counts as equal-opportunity conversion rates.

| Side/layout | Parent goals/concedes | Child goals/concedes |
|---|---:|---:|
|Blue0|17/12|14/12|
|Blue1|18/11|14/12|
|Blue2|11/14|13/12|
|Blue3|18/11|13/11|
|Blue4|10/13|13/11|
|Orange0|15/12|15/10|
|Orange1|21/12|12/11|
|Orange2|20/12|10/11|
|Orange3|19/12|10/11|
|Orange4|20/13|10/11|

All games finish regulation, no unresolved overtime. No-touch resets are disabled
by protocol; their zero count is not a learned achievement. Both policies are
deterministic. Ten fixed development matches do not establish SSL, ranked skill
or native Rocket League compatibility. No RLBot deployment occurred.

## Original skill cases,64 each

| Family | Parent goals/concedes/timeouts | Child goals/concedes/timeouts |
|---|---:|---:|
|Natural|38/25/1|38/22/4|
|Challenge|0/38/26|0/38/26|
|Finishing|6/42/16|6/36/22|
|Defense|8/21/35|7/23/34|
|Kickoff|64/0/0|64/0/0|

Natural's35kickoff starts all score in both. The29ongoing-ground starts remain
3goals; concedes25->22, timeouts1->4, touched cases21->23, contacts47->73.
This is not a scoring breakthrough. Finishing reaches all64balls in both and
scores6in both; projected on-target events47->53, actual concedes42->36.
Defense regresses slightly in goals and concedes. Challenge controlled gains
remain11, forward-controlled advances4->6, but actual outcomes do not improve.
Kickoff controlled-acquisition and advancement events each rise0->12 while all64
still score; repeated standard layouts are not64independent generalization tests.

## Training signal and concrete next action

All rows are retained. Early child1..12 versus late13..25: contacts17.506->17.946
per learner-minute, movement1189.22->1194.39uu/s, entropy0.6823->0.6715, ended
episode touch fraction87.39%->89.08%. Early includes the declared fresh-episode
transient; this is descriptive, not a controlled attribution to one ingredient.

Late finishing-vs-Nexto has12,641ended player episodes:1,125goals and8,266concedes,
but10,111`success_endings` under the old declared shot-projection proxy. A prior
on-target touch remains a proxy success even if Nexto subsequently saves and
scores. This is the documented V1 definition, not a new lifecycle bug. The
actual goal +/-10 reward still arrives correctly. Nevertheless, rewarding the
already solved touch/projection subtask is now poorly aligned with the missing
competency: beating a keeper and converting the attempt.

Keep600as the stronger measured full-match reference, and preserve child25as a
research candidate with more repeat contacts. Do not claim it supersedes600on
match results. Next, make a narrowly versioned finishing-only graduation experiment
as described in `NEXT_FINISHING_STAGE.md`. Do not blindly extend this completed
block or build another mechanic classifier. Publish and verify the next exact
authority/code/tests before learning; all current artifacts remain immutable.
