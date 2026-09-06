# Direct Skills +300: scoring regresses despite more follow-up contacts

## Identity and unchanged training

- Checkpoint: `checkpoints/rival2/direct_skills_v1/plus_000300.pt`.
- SHA256: `B7C8F0FC66582B5B2E71FC01195B9681C9318945E42F2AB070029A3ABC18CFA8`.
- 300 accepted updates; 1,327,104,000 new learner decisions;
  3,538,944,000 physical world ticks; 42,674 new Adam steps (94,824 cumulative).
- All fourteen checkpoint/evaluation checks and eleven full-match integrity
  checks pass. All twenty frozen runtime source hashes still match the package.
  Parent, contracts, finite model/Adam, counters, exact one-third learner exposure
  to Nexto and zero optimizer steps during evaluation are verified.
- Completed-update mean KL across updates 1–300 ranges from
  0.0018078048026256024 to 0.0061284025456337776. Maximum sample KL is
  33.80954360961914. KL remains telemetry only; there were no KL rejections.
- This block continued the exact +250 model, Adam and RNG after the published
  read-only finishing diagnosis. Physical episodes and recurrent hidden state
  restarted as declared by the existing resume protocol. No reward, model,
  scenario, observation, optimizer-setting or evaluation-method change occurred.

## Ten complete regulation matches against Nexto

| Metric | Parent +0 | +150 | +200 | +250 | +300 |
|---|---:|---:|---:|---:|---:|
| Wins / losses | 0 / 10 | 0 / 10 | 0 / 10 | 0 / 10 | 0 / 10 |
| Rival goals | 1 | 24 | 33 | 44 | 29 |
| Concessions | 266 | 203 | 187 | 183 | 197 |
| Rival contacts | 170 | 368 | 352 | 402 | 407 |
| Contacts/minute | 3.40 | 7.36 | 7.04 | 8.04 | 8.14 |
| Same-player follow-ups / resolved next contacts | 0 / 166 | 25 / 339 | 26 / 317 | 34 / 356 | 71 / 370 |
| Kickoff first contacts | 0 | 1 | 3 | 1 | 5 |
| Matches with no Rival contact | 0 | 0 | 0 | 0 | 0 |
| Concessions with at most one contact since reset | 52 | 6 | 6 | 0 | 2 |

The +300 result is worse than +250 on both scoring and concessions. It is also
worse than +200 on these quantities. This is not an improved match checkpoint,
despite higher contact/follow-up counts. All ten matches remain decisive losses;
neither competitive Nexto parity nor SSL capability has been demonstrated.

| Match | Side | Initial layout | +250 Rival / Nexto | +300 Rival / Nexto | Same follow-ups +250 → +300 |
|---|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 4 / 19 | 4 / 22 | 4 → 2 |
| 1 | 0 | 1 | 7 / 18 | 1 / 21 | 1 → 16 |
| 2 | 0 | 2 | 6 / 16 | 1 / 22 | 3 → 16 |
| 3 | 0 | 3 | 3 / 19 | 4 / 19 | 3 → 2 |
| 4 | 0 | 4 | 4 / 18 | 2 / 19 | 1 → 4 |
| 5 | 1 | 0 | 3 / 20 | 4 / 13 | 3 → 14 |
| 6 | 1 | 1 | 6 / 15 | 3 / 21 | 12 → 4 |
| 7 | 1 | 2 | 4 / 18 | 3 / 19 | 5 → 2 |
| 8 | 1 | 3 | 4 / 20 | 5 / 19 | 1 → 4 |
| 9 | 1 | 4 | 3 / 20 | 2 / 22 | 1 → 7 |

Three matches score more, six score fewer, one is unchanged. Goal differential
improves in three and worsens in seven. Follow-up gains are concentrated in
matches 1, 2 and 5: the first two score substantially worse, while match 5 improves.
Extra same-player contacts therefore do not establish useful possession or better
finishing. They may represent several different physical outcomes; no new causal
claim or mechanic label is inferred from these aggregate counters.

Contact conservation is 71 same-player + 299 opponent follow-ups + 31 goal-ended
intervals + six unclosed intervals = 407 Rival contacts. The 19.19% follow-up
fraction excludes goal-ended intervals and is not possession duration. A
goal-ended interval does not identify the scorer. Nexto records 1,926 contacts.

Canonical velocity-change bins are 10 backward / 12 neutral / 385 forward.
Net ball-y displacement until the next contact or goal is 48 backward / 168
neutral / 185 forward. These are not interchangeable shot-quality measures.
Two native Rival demolition events are recorded; purposeful demo seeking is not
established. No-touch resets are disabled in this full-match protocol; zero such
resets is not a learned capability. All ten matches do contain Rival contacts.

## Fixed 64-case skill tests

| Metric | +150 | +250 | +300 |
|---|---:|---:|---:|
| Natural goals / concessions | 15 / 49 | 32 / 32 | 32 / 31 |
| Challenge goals / concessions | 2 / 52 | 1 / 52 | 2 / 49 |
| Challenge control gains | 3 | 7 | 5 |
| Challenge forward-control events | 3 | 3 | 3 |
| Finishing goals / concessions | 9 / 42 | 6 / 51 | 5 / 45 |
| Finishing on-target task events | 48 | 44 | 54 |
| Defense goals / concessions | 5 / 30 | 7 / 29 | 6 / 33 |
| Defensive clears | 37 | 38 | 36 |
| Kickoff goals / concessions | 26 / 0 | 52 / 0 | 52 / 0 |
| Kickoff controlled acquisitions | 0 | 0 | 0 |
| Failed kickoff timeouts | 38 | 12 | 12 |

Shooting remains twenty percent of episode-source starts. At +300 all 64
finishing cases reach the ball, with conditional first contact at 0.533724 seconds,
but only five score. Fourteen time out, compared with seven at +250. An increase
from 44 to 54 projected on-target task events has not produced more goals. These
events are explicitly bounded geometric proxies, not goalkeeper-aware shot
success. This is the same conversion weakness examined in the completed +250
physical trace diagnosis, not evidence that the shooting scenario was omitted.
The five scoring cases are 22, 37, 41, 43 and 61. Only case 22 also scored at
+250; cases 37/41/43 return from the +150 scoring set. The near-flat aggregate
therefore hides changes in which situations succeed, rather than an unchanged
deterministic policy.

Challenge has 53 touched cases, 86 contacts, five control gains and three forward
control events. Defense has 62 touched cases, 97 contacts, four control gains and
36 clears. These are small and mixed changes, not broad mastery.

### Kickoff versus ongoing-ground starts

The short kickoff test is unchanged case-for-case: layouts 0/1/2/3 score, layout 4
times out after twelve seconds. Layout 4 still concedes in the longer natural
test. Repeated layouts and sides are not 64 independent generalization cases.
Zero controlled kickoff acquisitions and only five first contacts in full matches
do not establish good kickoff possession or intentional fakes.

Of 29 ongoing-ground cases, +250 → +300 changes are: touched cases 18 → 19,
contacts 20 → 35, goals 3 → 3, concessions 26 → 25 and timeouts 0 → 1. The increase
in ongoing-play contact does not yet improve scoring. Kickoff-start outcomes in
the natural test are unchanged.

## Training telemetry and resume caveat

| Metric | Updates 201–250 | 251–300 | 252–300 |
|---|---:|---:|---:|
| Contacts / learner-minute | 10.7870 | 11.7650 | 11.5596 |
| Mean speed, uu/s | 1,176.39 | 1,171.28 | 1,170.64 |
| Categorical entropy | 0.7756 | 0.6377 | 0.6400 |
| Ended player episodes with a contact | 85.01% | 84.63% | 85.33% |

Update 251 starts fresh physical episodes after the diagnostic pause and is not
stationary evidence of an instantaneous policy gain. The complete block and the
block excluding that first update are both retained; their unequal exposure is
not hidden. `review_reduction_000300.json` contains role samples and raw totals.

Natural Nexto goals per million learner decisions increase from 628.40 in 201–250
to 783.73 in 251–300 (795.94 in 252–300); concessions decrease from 2,224.48 to
2,038.18 (2,070.43 excluding 251). Finishing Nexto goals per million change from
319.58 to 338.08, but only 325.98 excluding 251. Thus the stochastic training
signal is not uniformly bad, but it does not override the deterministic match
regression. Entropy continues declining; finite-state checks do not prove that
exploration is sufficient or that strategic behavior is healthy.

## Decision

Preserve +250 and +300 separately. Do not promote +300 as better, deploy it,
restart from a different lineage or silently retune rewards. The +250 physical
diagnosis already verified exact replay and actual goal/reset penalties; no new
operational failure appears in this block.

Continue the authorized unchanged campaign to the next scheduled +350 review,
not a long unchecked extension. This is a bounded observation decision, not a
claim that the regression will self-correct. Compare full-match goals/concessions
against +250, finishing against +150/+250, and useful control outcomes separately
from repeated contacts. If +350 still fails to recover match performance, pause
at an accepted boundary for a focused learning/transfer diagnosis before another
large unchanged block. Do not rerun the already completed +150/+250 finishing
trace merely because another review occurred. Numerical/corruption failures and
user stop instructions remain immediate stop conditions; KL remains telemetry.

All data reductions in this review are CPU-only and introduce no new policy
evaluation or optimizer step. The learner automatically resumed after its
scheduled evaluation and remains monitored. The SSL goal is still incomplete.

## Review validation

The focused audit/report/full-match reducer suite passes all 22 tests. The first
invocation encountered three fixture-setup errors because Windows denied access
to the default `pytest-of-patri` temporary directory; its other nineteen tests
passed. Repeating the identical suite with a new, path-checked isolated temporary
directory passed all 22 without source changes. Both JUnit files are retained:
`review_tests_000300.xml` (initial operational error) and
`review_tests_000300_localtemp.xml` (successful retry). This was not a learner,
checkpoint or evaluation failure and did not require a training restart.
The failed-test XML retains pytest's original traceback whitespace; the staged
whitespace check excludes only that raw generated failure artifact.

Subsequent CPU-only reduction narrowed a transfer question: Rival scored eight
of ten opening goals, but only 21 of the 216 goals after the first goal in each
match. At +250 those figures were eight opening goals and 36 of 217 later goals.
`learning_transfer_review_000300.json` preserves exact partitions and action/reward
accounting. `KICKOFF_RESET_TRACE_DIAGNOSTIC.md` prospectively specifies a bounded
read-only input/reset/opponent-cadence trace if the +350 review warrants the
already described pause. This preparation is not a native replay result or an
assertion of a reset bug; no live training source or setting was changed.
