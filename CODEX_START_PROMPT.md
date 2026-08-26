# Closed Codex Boundary — Rival 2.0 Campaign 02 Complete

RivalSim v0.5 remains complete with `PASS_GREEN`. Rival 2.0 Campaign 02 has completed its exact
controlled entropy-off execution and is now closed.

Campaign 02 reproduced Campaign 01's seed-`20260826` initialization model SHA-256 and every
substantive initialization-evaluation value exactly. It changed only the campaign-layer PPO
entropy coefficient from `0.01` to `0.0`; every other model, contract, PPO, self-play, seed,
world-count, rollout, checkpoint, and evaluation semantic remained fixed.

The 131,072-world, horizon-32 run stopped at update 12 with 100,663,296 agent decision samples,
the first completed update crossing 100M. Initialization plus the first 10M/25M/50M/100M
threshold checkpoints and evaluations were preserved and published.

Independent closeout statuses:

- execution status: `COMPLETE`;
- behavioral result under the prospectively fixed rule: `IMPROVED`;
- initialization control: `PASS_GREEN`;
- final exact checkpoint continuation: `PASS_GREEN`;
- frozen v0.5 trainer: unchanged (`PASS_GREEN`).

Campaign 02 improved two of the three primary metrics relative to initialization and was not
worse than Campaign 01 final on any of them. Ordinary self-play touches/minute reached `0.291182`
versus `0.272091` at initialization and `0.175624` at Campaign 01 final. Stochastic touch
differential versus initialization reached `+35` versus `+15` initially and `-46` at Campaign 01
final. Stochastic goal differential was `-3`, slightly worse than the initialization value `-2`
but better than Campaign 01 final `-16`.

No Campaign 02 update crossed the diagnostic instability thresholds. Maximum approximate KL was
`0.008194` at update 2 and maximum clip fraction was `0.087534` at update 6. Representative final
analog standard deviation was approximately `1.015`, far below the `exp(1)` ceiling; the
Campaign 01 update-4 instability did not recur. This is controlled behavioral improvement, not a
claim of learned Rocket League competence or external transfer.

The full resumable final checkpoint is published at
`checkpoints/rival2/campaign02/rival2_campaign02_100m_resume.pt`. Its SHA-256 is
`4A9B366CD3A04222D639252EB2E3EBAD194AF2154D9DBFF213B1AF89A3909FA0` and its exact size is
21,126,324 bytes.

Read the completed evidence in:

- `docs/RIVAL2_CAMPAIGN02_RESULTS.md`;
- `results/rival2/campaign02/summary.json`;
- `results/rival2/campaign02/comparison_campaign01.json`;
- `results/rival2/campaign02/optimizer_diagnosis.json`;
- `results/rival2/campaign02/initialization_control.json`;
- `results/rival2/campaign02/evaluation_000m.json` through `evaluation_100m.json`;
- `results/rival2/campaign02/training_curve.json`;
- `handoff/rival2-c02/README.md`;
- `handoff/rival2-c02/DIAGNOSIS.md`;
- `handoff/rival2-c02/CAMPAIGN.md`;
- `handoff/rival2-c02/ACCEPTANCE.md`.

Tracked v0.1-v0.5 results, Campaign 01 artifacts, the four Rival 2.0 contracts, and the frozen
v0.5 training implementation remain byte-for-byte unchanged.

No further work is authorized by this prompt. Do not continue Campaign 02 training, tune another
hyperparameter, change rewards, begin a curriculum, or begin v0.6 RocketSim/RLBot transfer. A new
explicit controlling handoff is required for any subsequent work.
