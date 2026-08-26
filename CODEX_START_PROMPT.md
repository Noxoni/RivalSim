# Closed Codex Handoff — Rival 2.0 Overnight Curriculum Complete

The Rival 2.0 overnight curriculum is complete. The user prospectively extended the final
Reward V1 wall-clock continuation from three hours to six hours before that final phase
completed; the clean official lineage therefore stopped at the first completed PPO update
crossing 21,600 elapsed Phase C seconds.

Completed boundary:

- acquisition completed at update 420 / 3,523,215,360 cumulative samples after two consecutive
  4,096-world held-out evaluations reached no-touch truncation <=1%;
- the reward-only `RIVAL2_REWARD_V2` -> `RIVAL2_REWARD_V1` transition passed exact state-custody
  checks;
- Phase B completed exactly 239 PPO updates / 2,004,877,312 additional base-reward samples;
- Phase C completed at update 5,403 / 45,323,649,024 cumulative samples after 21,601.926
  elapsed seconds at update completion;
- all 5,283 continuation updates passed integrity;
- the final six-hour held-out evaluation recorded 85.483708 touches/minute, 2.496403
  goals/minute, and 0.003418 no-touch truncation;
- the final full resumable checkpoint has SHA-256
  `4DC158DC2A9D16B79FB5FE7D868E3B50928AB113B55DFCC753F3734F8D87372E` and passed exact reload.

Published evidence:

- `docs/RIVAL2_OVERNIGHT_RESULTS.md`;
- `results/rival2/overnight/`;
- `checkpoints/rival2/overnight/`.

There is no active continuation authorized by this file. Do not resume training, build the
viewer, alter the frozen training contracts, or begin v0.6 without a new explicit handoff.
