# Entity +250: mixed fixed-scenario result, still no Nexto scoring

The regular deterministic evaluation completed at 2026-09-05T23:34:38Z. Training
continued automatically from the same model and optimizer. No learning settings,
reward, physics, architecture, scenario bank or opponent schedule were changed.

## Identity and exposure

- Checkpoint: `checkpoints/rival2/ssl_entity_continuation_v1/plus_000250.pt`.
- SHA-256: `22C3D4762784F7AC9D48DD429DFDFCA7E88333DFC2DABE2B11A9C01ED4C53878`.
- 250 entity-phase updates; 150 continuation updates after the immutable +100 pilot.
- 44,702 cumulative Adam steps; 1,439,805,700 cumulative entity-phase trainable
  decisions and 2,949,120,000 physics world ticks. Not total lifetime training.
- Updates 201–250 added 276,281,906 trainable decisions, including 18,630,094
  current-Rival-versus-Nexto decisions, and 8,684 Adam steps. Nexto is never trained.
- `boundary_000250_audit.json` reproduces all counters, hashes, finite model/Adam,
  unchanged optimizer configuration, runtime sources, contracts, action table and
  entity buffers. Large KL remains telemetry, never rejection. No safety fault.
- Prefix `training_curve_through_000250.jsonl`, SHA-256
  `A7D623CA24F7D9EB5E13C1B6900E91DD9AB5C370777E950CFEE7A5592C3DEB22`, contains
  the 150 complete rows 101–250. No racing live log is part of this evidence.

## Fixed 64-world scenarios: +200 to +250

| Metric | Acquisition +200 / +250 | Finishing +200 / +250 | Nexto kickoff +200 / +250 |
| --- | ---: | ---: | ---: |
| Worlds with a focal contact | 61 / 60 | 64 / 64 | 25 / 30 |
| Contact events | 85 / 101 | 92 / 93 | 63 / 43 |
| Goals for | 40 / 42 | 53 / 55 | 0 / 0 |
| Goals against | 4 / 9 | 2 / 5 | 64 / 60 |
| No-touch truncations | 13 / 7 | 7 / 2 | 0 / 0 |
| Contacts per player-minute | 6.400 / 7.812 | 12.093 / 14.406 | 4.513 / 3.275 |
| Positive canonical ball-y velocity at contact decision end | 92.94% / 96.04% | 100% / 96.77% | 80.95% / 39.53% |

All scenario hashes match +200 exactly. `evaluation_000250_comparison.json`
contains full-precision values. Acquisition and finishing are bilateral current
self-play, so the opposing policy evolves too. They show fewer inactivity endings
and more activity, not an isolated win-rate improvement: focal concessions also
rose. First-contact medians exclude failed acquisition attempts.

Nexto is fixed. More distinct cases acquired a contact, but contact frequency and
the positive-ball-velocity fraction declined. Those are important counter-signals;
do not summarize this boundary as uniformly better. Positive velocity at decision
end is the existing metric, not contact impulse, shot quality, possession duration,
or evidence of a named mechanic.

The four Nexto cases without a conceded goal are **30-second time-limit endings**,
not wins or completed drawn matches. This is derived from the existing exclusive
one-episode-per-world accounting: 64 minus 60 goals against, zero goals for, zero
no-touch truncations, zero surviving-at-horizon. `evaluate()` removes a world only
on reset; `FreshGroundEnv` and the native reward kernel permit only goals, no-touch
timeouts or episode time limits. The same derivation gives six acquisition and two
finishing time-limit endings. These are derived counts, not an extra recorded field.

## Interpretation and next check

No Nexto scoring breakthrough is established. At the time of this short report,
the last full-match comparison was +200 versus +100 (zero Rival goals/wins,
fewer concessions); the completed follow-up is appended below.
The mixed +250 short result warrants a bounded natural-match follow-up using the
same ten-match method. `FULL_MATCH_000250_PLAN.md` fixes +250 versus already recorded
+200 before any +250 full-match inference. This is development diagnosis, not a
new test-based checkpoint selection or permission to tune rewards.

Training update times ranged from 35.7 to 169.1 seconds in this interval. Later
updates slowed while the GPU was shared with other applications. Contention is a
plausible contributor, not a controlled causal benchmark; no applications were
stopped or resource/capability settings changed. All scheduled updates completed.

## Follow-up result

The separately published natural-match follow-up is now complete. See
`FULL_MATCH_000250.md` and `full_match_000250_integrity.json`: contacts increased
238 to 302, but concessions increased 243 to 317 and same-player follow-ups fell
22/237 to 6/301. Rival scored zero and lost all ten. This is a gameplay regression
relative to +200, not a positive result inferred from fewer scenario no-touch
endings. The original short-evaluation measurements above remain unchanged.
