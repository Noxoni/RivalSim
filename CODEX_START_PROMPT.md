# Closed Codex Boundary — Rival 2.0 Campaign 04

Rival 2.0 Campaign 04 is complete. It resumed the exact Campaign 03 100M checkpoint with its
optimizer, RNGs, counters, opponent assignments, and historical-policy state intact; continued
the unchanged Reward V2 training line; stopped at update 120 / 1,006,632,960 cumulative agent
decision samples; and published the four authorized evaluations and final checkpoint.

## Completed result

- resume checkpoint SHA-256:
  `A0F2E554448B31A373BD73254125AC0ADFDB541EE3B695AD9D040B2CCFA68991`;
- resume authority and loaded state: `PASS_GREEN`;
- continuation updates: 108, covering updates 13 through 120;
- update integrity: 108 / 108 `PASS_GREEN`;
- authorized evaluation integrity: 4 / 4 `PASS_GREEN`;
- final update/sample boundary: 120 / 1,006,632,960;
- update 121 run: no;
- final checkpoint exact reload/continuation: `PASS_GREEN`;
- final checkpoint SHA-256:
  `DB5AA09B2CAD40D4C1F5DB1014FDE245C58994A6948458212751724F782BE6B0`;
- frozen primary-axis behavioral classification at 1B: `CONTINUING`.

The published 100M -> 250M -> 500M -> 750M -> 1B curve is in
`docs/RIVAL2_CAMPAIGN04_RESULTS.md` and `results/rival2/campaign04/`. The exact final resumable
checkpoint is `checkpoints/rival2/campaign04/rival2_campaign04_1b_resume.pt`.

## Boundary

There is no active follow-on authorization in this file. Do not continue training, build the
viewer, change the reward/PPO/model/simulator, or begin v0.6 RocketSim/RLBot transfer work
without a new explicit handoff.
