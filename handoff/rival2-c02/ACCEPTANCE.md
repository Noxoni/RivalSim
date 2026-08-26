# Rival 2.0 Campaign 02 — Acceptance and Closeout

Campaign 02 is complete only when the bounded controlled rerun has been executed, evidenced, published, and stopped.

## A. Parent and immutability

Required:

- starting HEAD is `1ce5932cadd66b14032e61750836763499567bc9`;
- published `results/v0.1/` through `results/v0.5/` remain byte-for-byte unchanged;
- all Campaign 01 tracked artifacts remain byte-for-byte unchanged;
- frozen Rival 2.0 observation/action/reward/episode contracts remain unchanged;
- frozen v0.5 trainer implementation remains unchanged except for new campaign-layer code/evidence outside the frozen implementation files.

## B. Controlled-variable proof

Before training, evidence must show:

- same campaign seed `20260826`;
- same evaluation seed `920260826`;
- fresh initialization model SHA-256 equals Campaign 01 initialization:
  `890F224879DB6E458472985B226A664D8AE49B8303C21CFB0FD83A485CF42848`;
- Campaign 02 PPO config differs from Campaign 01 only in `entropy_coefficient`, with `0.0` instead of `0.01`;
- the unchanged initialization evaluation is reproduced under the same held-out protocol.

Any other learning-semantic difference blocks the controlled Campaign 02 interpretation.

## C. Training execution

Required unless a genuine correctness/integrity hard stop occurs:

- 131,072 worlds;
- horizon 32;
- first completed update crossing 100M samples;
- expected nominal final sample count 100,663,296 if 12 ordinary updates complete;
- all observations/actions/rewards/values/log-probabilities/advantages/returns/losses/gradients/parameters/optimizer state finite;
- analog actions bounded and buttons exactly binary;
- done/reset accounting valid;
- historical policies frozen/gradient-free;
- policy/sample version accounting exact;
- zero ordinary hot-path H2D/D2H state traffic.

## D. Snapshot custody

Required snapshots:

- initialization;
- >=10M;
- >=25M;
- >=50M;
- >=100M.

Final checkpoint must be full and resumable and must pass the frozen v0.5 continuation checks. Publish the final checkpoint if repository policy permits; otherwise publish exact size/hash and retain local custody with explicit path and reproduction instructions.

## E. Evaluation

Every snapshot must be evaluated under the same Campaign 01 held-out protocol. Publish separate machine-readable results for all five snapshots.

The final report must compare:

1. Campaign 02 initialization -> final;
2. Campaign 02 final -> Campaign 01 final;
3. Campaign 02 final -> frozen initialization head-to-head.

At minimum report touches, goals, no-touch truncations, wins/losses/draws, goal/touch differentials, action magnitudes, analog standard deviations, button probabilities/activation, and episode duration.

## F. Optimizer diagnosis

Publish per-update PPO metrics and explicitly report:

- whether analog standard deviations trend toward the `exp(1)` ceiling;
- maximum approximate KL and which update produced it;
- maximum clip fraction and which update produced it;
- any update with KL >=0.1 or clip fraction >=0.3;
- whether the Campaign 01 update-4 style instability recurs.

The diagnostic entropy metric may be logged, but Campaign 02's optimization contribution from entropy must be exactly zero.

## G. Behavioral closeout

Apply the prospectively defined `IMPROVED` / `DEGRADED` / `INCONCLUSIVE` classification from `CAMPAIGN.md` without changing the rule after results are known.

Execution may be `COMPLETE` even if behavioral outcome is `DEGRADED` or `INCONCLUSIVE`.

Do not claim learned Rocket League competence solely from optimizer metrics.

## H. Quality and regression

Run relevant campaign tests, repository tests, Ruff, Python compilation, and `git diff --check`. Confirm prior frozen evidence differences are zero.

A complete full simulator parity rerun is not required solely because the campaign changes no simulator/trainer implementation behavior, but any touched shared implementation file must trigger the appropriate inherited regression gates.

## I. Boundary

After Campaign 02 closeout:

- do not continue training past the authorized first completed update crossing 100M;
- do not tune another hyperparameter;
- do not change rewards;
- do not begin a curriculum;
- do not begin v0.6 RocketSim/RLBot transfer.

Publish what happened and stop for review.