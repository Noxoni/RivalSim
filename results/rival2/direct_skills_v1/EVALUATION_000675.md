# +675: shooting curriculum did not improve the fixed tests

The650->675 curriculum-only block completed and stopped normally for review at
2026-09-06T10:54:43Z. Worker20016 exited; stderr is empty. No KL/nonfinite stop.
Do not extend the current block automatically: finishing did not improve and
the full-match regression is broad. This is not SSL-level gameplay.

## Preserved checkpoint and training

`checkpoints/rival2/direct_skills_v1/plus_000675.pt`

SHA256 `77E6EF076F7073549F45168FA6FB34BE9A2D4C25E1580F17F7434A59195B32F8`

-675 accepted direct-stage updates,2,985,984,000 direct learner decisions,
  7,962,624,000 physical world ticks. Earlier-stage exposure remains separate.
-25 new updates,110,592,000 learner decisions,294,912,000 physics ticks and
  3,538 Adam steps;148,186 cumulative Adam steps. Model/Adam/fourRNG preserved.
-14 root and27 curriculum-specific integrity checks pass.32 focused CPU tests
  pass. Initial broad test invocation had29passes/3fixture permission errors in
  Windows shared pytest temp. A fresh workspace-local base resolves it with no
  test/production semantics change; raw errorXML is retained separately.
-Maximum update-mean KL .0022629100, maximum sample estimate9.951499; telemetry
  only, zero KL rejections. Peak allocated CUDA16,321,118,208bytes.
-Source650, old root/T2/follow-up authorities, all runtime contracts, rewards,
  models' architecture and evaluation sources are unchanged. Only the previously
  published intermediate shooting bank differs from650; no fresh optimizer.

## Ten original deterministic Nexto matches

Same corrected-reset runtime, five layouts on both sides. No evaluation case,
action-selection mode or opponent change. This is a development comparison,
not an untouched ranked benchmark or a capability promotion.

| Metric |600|650|675|
|---|---:|---:|---:|
|Wins|0|0|0|
|Rival goals|65|42|37|
|Nexto goals|144|139|168|
|Goal difference|-79|-97|-131|
|Rival contacts|508|577|520|
|Nexto contacts|1357|1443|1760|
|Rival contacts/minute|10.16|11.54|10.40|
|Same-player/resolved follow-ups|128/437|209/533|148/474|

Against650:1 matched case improves and9 worsen. Against600:all10 worsen.
Against corrected451:0improve,1tie,9worsen. All10 still improve over entering
corrected0, which does not excuse the recent regression.

Blue25/64 ->25/76; Orange17/75 ->12/92. Blue contacts270 ->309, Orange307 ->211.
Eight opening goals remain eight; second goals4 ->1; later goals34 ->29 and
later concessions137 ->166. These are counts with unequal segment exposure,
not equal-opportunity conversion rates. Follow-up fraction39.21% ->31.22% is
not possession duration. Forward ball-displacement bins329 ->235, backward50
->73, neutral193 ->209. Native demo1 is not evidence of intentional demo play.
Every game contains Rival contacts; no-touch resets are disabled by protocol.

| Side/layout |650 Rival/Nexto|675 Rival/Nexto|
|---|---:|---:|
|Blue0|5/14|5/16|
|Blue1|4/12|7/16|
|Blue2|4/13|3/13|
|Blue3|7/13|7/15|
|Blue4|5/12|3/16|
|Orange0|1/13|1/12|
|Orange1|4/16|3/21|
|Orange2|3/13|3/17|
|Orange3|4/16|3/21|
|Orange4|5/17|2/21|

## Original fixed64 cases per family

|Family|650 goals/concedes/timeouts|675 goals/concedes/timeouts|
|---|---:|---:|
|Natural|32/25/7|31/30/3|
|Challenge|1/43/20|2/33/29|
|Finishing|5/41/18|5/43/16|
|Defense|8/17/39|7/17/40|
|Kickoff|51/0/13|51/0/13|

Finishing touches all64 again, on-target proxies53 ->53, actual goals5 ->5.
Challenge control14 ->12, forward-controlled advances6 ->10, contacts135 ->165.
Defense controls18unchanged, clears40 ->37, contacts171 ->186. These are task
proxies, not independent possession/save adjudication or whole-game success.

Natural29 ongoing-ground cases:3/24/2 ->2/25/2, touched cases21 ->23 andcontacts
48 ->72. The other35 kickoff cases:29/1/5 ->29/5/1. Blue layout0 changed from
five timeout cases to five concessions; these are repeated deterministic layout
cases, not five independent configurations. Separate kickoff acquisition is0.

## Training rates versus transfer

Full curves1..675 and651..675 are saved. The rate summary excludes only the
declared fresh-resume buffers601/651, preserving them in full data. Use actual
role decisions, not source-bank percentages, as denominators.

602-650 ->652-675:contacts17.3235 ->17.9690 per learner-minute;speed1191.16
->1185.16uu/s;ended-episode contact89.91% ->89.26%;entropy.63031 ->.54605.
Natural Nexto goals958.72 ->915.50 andconcedes1706.93 ->1660.96 per million
role decisions;finishing goals397.84 ->328.09,concedes2410.06 ->2363.97.
These are different curricula, not a controlled causal estimate of learning.

Within the unchanged new block,652-663 ->664-675:finishing goals317.92 ->338.66,
concedes2352.61 ->2375.78 per million role decisions. Natural goals981.72
->852.53,concedes1589.48 ->1728.95. Reward/proxy improvement is insufficient to
claim transfer. The fixed original matches and shots show no improvement.

## Next bounded action

Before more learning, examine the first-versus-later kickoff transition on the
current corrected runtime with the existing read-only tap. The recent results
still show8/10 opening goals, but only29 later Rival goals versus166 concessions;
the pattern predates this curriculum and cannot by itself prove its cause.
Fullmatch675 records7 Rival versus205 Nexto first kickoff contacts; first touch
is NOT equivalent to winning a kickoff, and intentional fakes are not inferred.

Replay675 exactly with the existing bounded first32-physics-tick kickoff tap.
Require bit-identical complete raw match results before interpretation. Compare
same-side/layout initial and post-goal observations, recurrent reset, actions,
native state and Nexto scheduler state. This is a post-correction continuity
check, not a repeat of the old wheel-only patch or permission to change physics.
See `kickoff_continuity_000675/AUTHORITY.md`. Do not change rewards, PPO or policy
from an input discrepancy alone. If no discrepancy appears, investigate later
behavior instead; do not invent a reset fix. The user's SSL goal remains active.
