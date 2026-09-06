# +150 versus +250: physical finishing diagnosis

## Verified source, not a replacement evaluation

Both read-only replays reproduced **every field** of their saved five-family
evaluations exactly, including case outcomes, contact counts, event totals and
float summaries. Checkpoint/model hashes stayed unchanged; no optimizer was
constructed or stepped. These are diagnostic replays, not newly selected results.

- +150 archive: `finishing_trace_000150/finishing.npz`, 40,584,398 bytes,
  SHA256 `CD702E5C62344287E7ED289D4DADA8F2755217B7FB11580E3585A4E1C02D75F5`.
- +250 archive: `finishing_trace_000250_v2/finishing.npz`; its exact size/hash,
  complete schema and replay checks are in the adjacent manifest.
- Each records 64 original finishing cases at all 1,440 available physics ticks,
  with 360 exact policy observations/actions and original-episode validity masks.
- The first +250 helper attempt failed before capture because wheel-contact
  storage is flat. That helper-only error and the explicit-width correction are
  preserved. The retry and +150 capture passed; neither changed simulation.

## What is actually happening

| Physical observation | +150 | +250 |
|---|---:|---:|
| Goals / concessions / timeouts | 9 / 42 / 13 | 6 / 51 / 7 |
| Boost pressed at first contact | 64 / 64 | 64 / 64 |
| Jump pressed at first contact | 1 / 64 | 15 / 64 |
| Native flip active at first contact | 0 / 64 | 0 / 64 |
| Cases with a later Nexto contact | 56 / 64 | 58 / 64 |
| Rival follow-up before Nexto's next contact | 8 / 64 | 8 / 64 |
| Goals without a subsequent Nexto contact | 8 / 9 | 6 / 6 |
| Final conceded-step reward range | -10 to -9.962477 | -10 to -9.962477 |

Rival is not failing these near-ball attempts because it sits still or refuses
to boost. Native `is_boosting` is true at all 64 first contacts in both versions;
mean car speed there is 1,119.18 versus 1,139.73 uu/s. It reaches every ball.
Many contacts are too lateral, while others
travel toward goal but remain reachable by Nexto. The subsequent-touch evidence
and actual case trajectories expose the missing quality: placing/controlling the
ball beyond the goalkeeper and continuing effectively after the initial contact.

The increased jump-button incidence is descriptive, not proof that jumping caused
the six lost scores. None of those six lost scoring cases had jump pressed at
first contact. There is no basis here for a generic jump/flip penalty.

The visible terminal penalties confirm that this is **not** the prior missing
goal-reward/reset failure. The tiny magnitude variation is the already-tested
within-four-tick goal discount. This does not prove optimal PPO credit assignment,
only that the actual concession consequence reaches the learner's reward path.

## Paired examples of lost scores

Velocity below is the native ball velocity at the end of the first uninterrupted
collision episode (two physics ticks in these cases), canonicalized toward the
opponent goal. X is lateral, Y is forward. Geometry is not a new reward rule.

| Case | +150 first-contact velocity X/Y/Z, uu/s | +250 X/Y/Z | Observed continuation |
|---|---|---|---|
| 5 | 971 / 785 / 375 | 1070 / 234 / 253 | +150 follows up at 1.650 s and scores; +250 is much more lateral, Nexto contacts at 3.875 s, then Rival concedes |
| 11 | 680 / 1541 / 432 | 662 / 1699 / 387 | +250 is faster forward but Nexto contacts at 1.325 s; speed alone did not preserve the finish |
| 16 | 542 / 1005 / 320 | 396 / 1097 / 333 | Both have a Rival follow-up, but +250 then gives Nexto contact at 1.592 s and concedes |
| 37 | 558 / 1425 / 410 | 1088 / 1054 / 350 | +250 becomes substantially more lateral, loses the Rival follow-up, and Nexto contacts at 3.958 s |
| 41 | 169 / 1418 / 433 | 228 / 1358 / 353 | Nexto contacts in both versions, earlier at +250 (0.850 versus 0.908 s); only +150 still scores |
| 43 | -184 / 1884 / 527 | 152 / 1702 / 332 | +250 is lower/slower and redirected more centrally; Nexto contacts at 1.250 s instead of being bypassed |

For case 37 the simple constant-velocity goal-plane X projection changes from
about 657 to 1,745 uu. For case 5 it changes from about 2,448 to 11,864 uu. These
numbers describe the initial direction only: +150 case 5 **still scores through
a follow-up**, which is exactly why the ray must not be treated as a ground-truth
shot-success detector. Bounces, posts and opponent actions are not in that ray.

There are also real gains: cases 14, 19 and 52 become goals. In each +250 case
Rival's initial contact is followed by a goal without another Nexto contact,
where the older version allowed Nexto a subsequent touch. The shared policy
is changing useful contact geometry, but not consistently preserving it across
the fixed cases. This is a mixed learned-behavior result, not total collapse.

All exact actions, positions, quaternions, wheel/flip state, contact timing and
outcomes are in `finishing_trajectory_comparison_000150_to_000250.json` and the raw
archives. Its CPU reducer does not modify or reevaluate either policy.

## Decision

Resume the **same +250 model and Adam**, unchanged, to the next scheduled +300
review. No alternate checkpoint, new reward, additional shooting family, task ID,
architecture change or preservation objective is introduced. The bounded
diagnosis is complete: it found behavior-level placement/follow-up limitations,
not an active implementation fault warranting a training restart or immediate
reward retune. This conclusion is limited to the measured paths, not proof that
every part of the simulator/trainer is bug-free.

The continuation is justified by three successive full-match scoring gains
(12 to 24 to 33 to 44 from +100 through +250), improving concessions, new
ongoing-ground scores and improved challenge/defense events. Finishing remains
a specifically tracked weakness. At +300 compare actual finishing goals and
concessions, follow-up contacts, ongoing-ground scores and the same full matches.
Do not report proxy successes as learned possession or SSL capability.

## Reproduce CPU report

```powershell
.venv/Scripts/python.exe benchmarks/report_rival2_finishing_trace.py --baseline results/rival2/direct_skills_v1/finishing_trace_000150 --candidate results/rival2/direct_skills_v1/finishing_trace_000250_v2 --output results/rival2/direct_skills_v1/finishing_trajectory_comparison_000150_to_000250.json
```

This verifies archive hashes, positive replay-parity/immutability verdicts,
native outcome accounting and deterministic report rebuild. It runs no GPU work
and does not overwrite a changed report. The authoritative checkpoint selection
remains the scheduled +250 checkpoint, not either diagnostic outcome subset.
