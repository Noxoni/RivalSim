# Direct Skills +450: stalled match improvement; diagnostic pause at +451

The unchanged ten-match test records 60 goals / 175 concessions and 0/10 wins,
versus +400's 58 / 167. Rival contacts increase from 458 to 477 (9.54/min), and
same-player next contacts from 107/393 to 132/416 resolved intervals. More
contacts do not establish possession mastery: goal difference worsens from
-109 to -115. Concessions with at most one contact since reset rise from 5 to 14.
Every match has Rival contacts; no-touch resets are disabled by this protocol.

Contact accounting: 132 same-player + 284 opponent + 60 goal-ended + 1 unclosed
= 477. Net displacement bins are 33 backward / 185 neutral / 258 forward;
velocity-change bins are 25 / 49 / 403. These are not equivalent shot metrics.
There is one native Rival demolition event, not proof of intentional demo play.

## Fixed skill outcomes

| 64-case family | Goals | Concedes | Timeouts | Relevant task telemetry |
|---|---:|---:|---:|---|
| Natural | 37 | 26 | 1 | 54 touched cases, 75 contacts |
| Challenge | 4 | 37 | 23 | 8 control gains, 7 forward-control events |
| Finishing | 5 | 41 | 18 | All 64 touch; 45 projected on-target events |
| Defense | 5 | 24 | 35 | 16 control gains, 34 clears |
| Kickoff | 64 | 0 | 0 | 0 control gains; repeated five layouts/both sides |

Finishing is already 20% of the reset-source distribution. Its +400 result was
7 goals / 38 concedes, so the shooting problem is not omission of a scenario.
All cases touch in about 0.535 seconds, but only five score. The 45 geometric
on-target events are not 45 successful shots or a goalkeeper-aware measure.
Natural ongoing-ground cases alone score 2 / concede 26 / timeout 1 across 29
cases, down from 3 / 25 / 1. The other 35 natural starts are successful kickoffs.
There is still no SSL or Nexto-parity evidence.

## Integrity and pause

+450: `plus_000450.pt`, SHA256
`BD5AD1C2D727A602EA48381B76C08A7A3E1556B05277F30452A13FC05E34F6C6`.
1,990,656,000 learner samples; 5,308,416,000 physical world ticks;
116,246 cumulative Adam steps. Fourteen checkpoint checks and eleven match
checks pass; source evidence and all nine baseline comparisons are retained.

The previously published +400 rule called for a diagnostic pause on a stall or
regression. STOP was requested after +450 evaluation; the already-started +451
update completed safely. Latest resumable state is `paused_000451.pt`, SHA256
`B9D1BE5F75D22A9751832982A44B4A50F5BEE521DD4A1EE04C7640101EA8F324`:
1,995,079,680 learner samples and 116,388 cumulative Adam steps. Model/Adam are
finite, all RNG streams are present, and the external rolling checkpoint matches.
No KL rejection occurred. This is an agent-owned diagnostic pause, not user stop.

The prospectively committed +300 kickoff/reset trace was then run without an
optimizer. It exactly reproduces the saved raw match, summary and hidden-reset
counts. Its immutable 37,980,686-byte archive and per-input intervention are in
`kickoff_reset_trace_000300/`. Resume must use +451, never the diagnostic +300.

## Measured reset-input problem

Across 226 same-layout/same-side post-goal age-zero observations, all recurrent
states are zero. Eight wheel-contact observation flags carry pre-reset contacts;
other observed differences are at most 2.384185791015625e-7 in forward vectors.
70 first actions differ. CPU inference reproduces every recorded GPU action.
Changing only wheel fields in copied post-goal inputs back to their initial values
restores all 70 initial actions, with no remaining action difference. Boost changes
in all 70, steer/yaw in 28, pitch in 16 and roll in 58.

This proves first-action dependence on stale reset contacts. It does not establish
how much full-match scoring is caused by them. Initial states and training
curriculum resets already invalidate these contact caches; the standard native
kickoff reset omitted that invalidation. Training rewards, optimizer settings,
curriculum, weights and Nexto cadence have not been changed by the diagnosis.

## Additional CPU reduction

The initial goal is Rival's in all ten +450 matches, just as at +400. Later
Rival goals increase 48 to 50 while later concessions increase 167 to 175.
Thus the slight scoring increase is not merely initial-kickoff improvement.
Unequal segment exposure still prevents comparing these counts as success rates.

Over updates 351-400 versus 401-450, training contacts/minute increase
12.947900390625 to 13.8086669921875; movement speed 1169.1454566876446 to
1179.9278998119212 uu/s; ended-player-episode contact fraction 0.8747884430312154
to 0.8827977115304821. Categorical entropy declines 0.44863318954573733 to
0.39073902188407045. Mean completed-update KL is 0.002349214511110117 in the
latest block; maximum sample KL 60.38947296142578 is finite telemetry, not a
rejection condition. These acquisition changes do not cancel the worse match
goal differential or demonstrate finishing. Full role/exposure totals are in
`review_reduction_000450.json`; no optimization occurred during reduction.
