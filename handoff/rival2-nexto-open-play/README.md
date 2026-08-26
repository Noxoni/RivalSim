# Completed Handoff — Rival 2.0 vs Nexto Kickoff-Free Open Play

**Status:** `COMPLETE / PASS_GREEN`

**Completion commit:** `9807da8b3c404beb63a5426959132de549332128`

This handoff is historical and must **not** be executed again. The complete implementation and evidence are published at:

- `docs/RIVAL2_NEXTO_OPEN_PLAY_RESULTS.md`;
- `results/rival2/nexto_open_play/`.

The original detailed v1 handoff remains recoverable in Git history. This file is intentionally reduced to a completion boundary so an automated worker cannot mistake the old benchmark for active work.

## Final reference result

Exactly 4,096 physically continuous base states produced 16,384 deterministic kickoff-free first-goal duels.

- Overall: Rival 8,786, Nexto 7,255, draws 343; decisive Rival win rate `54.772%`.
- Rival as Blue: 3,753-4,241 with 198 draws; decisive win rate `46.948%`.
- Rival as Orange: 5,033-3,014 with 145 draws; decisive win rate `62.545%`.
- Orange-minus-Blue decisive win-rate difference: approximately `15.597` percentage points.
- Original vs mirrored decisive win rates: `55.158% / 54.384%`.
- Rival inheriting original Blue vs original Orange physical car: `54.678% / 54.866%`.

The team-side asymmetry remains unresolved and is an explicit target of the active RocketSim reciprocal-validation work.

## Active successor

Use:

`handoff/rival2-rocketsim-crosscheck/README.md`

The successor begins by building and validating a RocketSim -> `RIVAL2_OBS_V1` adapter for the same frozen Rival policy, while keeping pinned public Nexto on its native/source RocketSim/RLGym-style semantics. It then runs both normal 5-minute matches with kickoffs and the controlled kickoff-free comparison in RocketSim.

No training occurred in this completed handoff, and no training is authorized by this archival status file.
