# Direct Skills +250: match gains, persistent finishing weakness

## Identity and integrity

- Checkpoint: `checkpoints/rival2/direct_skills_v1/plus_000250.pt`.
- SHA256: `657EBB91830C3602495BDF71FD9EE6E66645C63C61159B6B63BE0F6BEA4FF93E`.
- 250 accepted updates; 1,105,920,000 new learner decisions;
  2,949,120,000 physical world ticks; 35,556 new Adam steps (87,706 cumulative).
- All fourteen checkpoint/evaluation audit checks and all eleven full-match
  integrity checks pass. Checkpoint hashes, parent, counters, finite model/Adam,
  exact one-third Nexto exposure and zero learning during evaluation are checked.
- Same entity-aware recurrent joint90 policy, Adam lineage, reward, scenario
  bank, 30 Hz decisions / 120 Hz physics, three-second rollout, two PPO epochs,
  learning rates, evaluation seeds and methods. No training source changes.
- Across 1–250: completed-update mean KL ranges from 0.001973527937893277 to
  0.0061284025456337776; maximum sample KL is 33.80954360961914. KL is telemetry
  only; no KL rejection. Finite/corruption checks remain enabled.

## Ten complete regulation matches against Nexto

| Metric | Parent +0 | +150 | +200 | +250 |
|---|---:|---:|---:|---:|
| Rival wins / losses | 0 / 10 | 0 / 10 | 0 / 10 | 0 / 10 |
| Rival goals | 1 | 24 | 33 | 44 |
| Concessions | 266 | 203 | 187 | 183 |
| Rival contacts | 170 | 368 | 352 | 402 |
| Contacts/minute | 3.40 | 7.36 | 7.04 | 8.04 |
| Same-player follow-ups / resolved next contacts | 0 / 166 | 25 / 339 | 26 / 317 | 34 / 356 |
| Kickoff first contacts | 0 | 1 | 3 | 1 |
| Matches with no Rival contact | 0 | 0 | 0 | 0 |
| Concessions with at most one contact since reset | 52 | 6 | 6 | 0 |

Five matches score more than +200, one scores fewer, four are unchanged. The
previous highest-scoring match falls from ten to three goals; the aggregate
gain is not just that same favorable match improving again. There are still
ten decisive losses: this is progress, not competitive parity or SSL capability.

| Match | Side | Initial layout | +200 Rival / Nexto goals | +250 Rival / Nexto goals |
|---|---:|---:|---:|---:|
| 0 | 0 | 0 | 4 / 19 | 4 / 19 |
| 1 | 0 | 1 | 2 / 19 | 7 / 18 |
| 2 | 0 | 2 | 2 / 19 | 6 / 16 |
| 3 | 0 | 3 | 10 / 14 | 3 / 19 |
| 4 | 0 | 4 | 0 / 21 | 4 / 18 |
| 5 | 1 | 0 | 3 / 19 | 3 / 20 |
| 6 | 1 | 1 | 4 / 20 | 6 / 15 |
| 7 | 1 | 2 | 1 / 19 | 4 / 18 |
| 8 | 1 | 3 | 4 / 18 | 4 / 20 |
| 9 | 1 | 4 | 3 / 19 | 3 / 20 |

Contact accounting: 34 next-Rival contacts + 322 next-opponent contacts + 45
goal-ended intervals + one unclosed interval = 402 Rival contacts. Follow-up
fraction is 34/356 = 9.55%, not possession duration. Goal-ended intervals do not
by themselves identify the scorer. Both players' goal-ended intervals reconcile
to every physical goal. Nexto contacts decline from 2,066 to 1,877.

Canonical ball-velocity-change bins are 18 backward / 3 neutral / 381 forward.
Net ball-y displacement until next contact or goal is 35 backward / 193 neutral /
173 forward. These are different quantities; neither is a shot-success count.
There are no Rival demolition events. No intent or named-mechanic claim is made.

## Fixed 64-case skill tests

| Metric | +150 | +200 | +250 |
|---|---:|---:|---:|
| Natural goals / concessions | 15 / 49 | 21 / 41 | 32 / 32 |
| Challenge goals / concessions | 2 / 52 | 2 / 50 | 1 / 52 |
| Challenge control gains | 3 | 2 | 7 |
| Challenge forward-control events | 3 | 2 | 3 |
| Finishing goals / concessions | 9 / 42 | 5 / 52 | 6 / 51 |
| Finishing on-target task events | 48 | 56 | 44 |
| Defense goals / concessions | 5 / 30 | 4 / 31 | 7 / 29 |
| Defensive clears | 37 | 35 | 38 |
| Kickoff goals / concessions | 26 / 0 | 39 / 0 | 52 / 0 |
| Kickoff controlled acquisitions | 0 | 0 | 0 |
| Failed kickoff timeouts | 38 | 25 | 12 |

The existing twenty-percent finishing family is present throughout training.
At +250 all 64 finishing cases touch the ball, but only six score. Seven time
out and 51 concede. First-touch mean is 0.535807 seconds. Reaching these nearby
balls is not the missing step; effective contact and subsequent play need diagnosis.

Scoring case indices are:

- +150: 5, 7, 11, 16, 22, 33, 37, 41, 43.
- +200: 7, 22, 37, 41, 52.
- +250: 7, 14, 19, 22, 33, 52.

Only three +150 scoring cases remain at +250; three new cases replace part of
the loss. Aggregate projected on-target events are not proof of a good finish.
They are documented task proxies, not goalkeeper-aware shot adjudication.

### Kickoff versus ongoing-ground starts

- Layout 2 joins layouts 0/1/3 as scoring kickoff starts. Layout 4 still fails
  twelve-second attempts and concedes in the longer natural test.
- Zero controlled kickoff acquisitions and one full-match first contact do not
  prove intentional fakes, kickoff possession or good first-contact technique.
- Repeated five-layout/two-side starts are not 64 independent generalization cases.
- Of 29 ongoing-ground starts, touched cases rise 16 to 18; goals rise 1 to 3,
  with 26 concessions and no timeouts. Contacts fall 21 to 20. Nine of the eleven
  extra natural scores come from kickoff layout 2; two come from ongoing starts.
- Challenge control improves but remains only seven cases, with three forward
  control events and three later lost-control events. Acquisition is not mastery.

## Stochastic training block (context, not evaluation replacement)

| Metric | Updates 151–200 | 201–250 |
|---|---:|---:|
| Contacts / learner-minute | 9.9465 | 10.7870 |
| Movement speed, uu/s | 1,150.70 | 1,176.39 |
| Categorical entropy | 0.9993 | 0.7756 |
| Ended player episodes with contact | 84.04% | 85.01% |
| Natural Nexto goals / concessions | 12,418 / 62,083 | 16,553 / 58,596 |
| Challenge Nexto control / forward control | 1,398 / 705 | 1,939 / 1,027 |
| Finishing Nexto goals / concessions | 3,655 / 38,332 | 4,079 / 37,573 |
| Kickoff Nexto control / forward control | 372 / 99 | 425 / 123 |
| Defense Nexto control gains | 1,280 | 2,016 |

Entropy is declining but no numerical guard fired. Training has many successful
events; that does not erase deterministic missed finishes or demonstrate SSL.
All 250 rows are preserved in the immutable training-curve prefix.

## Review action

The promised focused finishing diagnosis is triggered: +250 remains below +150
on finishing goals and above it on concessions. A clean accepted-boundary pause
completed after the entire scheduled evaluation, at 05:33:07 UTC on 2026-09-06.
The rolling and permanent +250 checkpoints are byte-identical. No update was
rolled back; no safety guard fired; this is not a campaign termination.

The bounded read-only helper, published before the pause, replays the same fixed
evaluation and records finishing trajectories. Its first live attempt found a
flat wheel-contact-buffer assumption in the diagnostic, not the trainer. That
operational error and its focused correction are preserved. The retry must pass
exact saved-evaluation parity before any causal interpretation. See
`FINISHING_TRACE_DIAGNOSTIC.md`, `diagnostic_pause_000250.json`, and trace manifests.

Do not resume from an old diagnostic snapshot. Preserve +250 weights, Adam,
counters and RNG; any subsequent continuation uses only that same lineage and
the declared fresh physical-episode/zero-hidden resume semantics. No automatic
reward retune, new architecture, specialist router, deployment or SSL promotion.

The bounded diagnosis subsequently completed with exact +150/+250 replay parity.
It identified inconsistent contact placement and goalkeeper/follow-up outcomes,
not a missing-goal/reset fault. Continue from the preserved +250 checkpoint
unchanged toward +300; details and exact trajectory evidence are in
`FINISHING_DIAGNOSIS_000250.md`.
