# Active Codex Handoff — Rival 2.0 Campaign 03

Campaign 02 is complete at commit `816c66b455d253b0f563bb378e53316a09ffd48e` with behavioral result `IMPROVED`. Campaign 03 is now authorized as a **direct training run**, not another preflight/validation exercise.

Read `handoff/rival2-c03/README.md` in full and execute it as the controlling specification.

## Mission

1. Preserve all published v0.1-v0.5 and Campaign 01/02 evidence unchanged.
2. Preserve `RIVAL2_REWARD_V1` and introduce Campaign 03 `RIVAL2_REWARD_V2` by adding one per-agent dense term:

   `approach = (car_ball_distance_before - car_ball_distance_after) / 4096.0`

   Distance is true 3D Euclidean distance in unreal units across one four-tick/30-Hz decision interval, measured before physics and at the final pre-reset transition state. Each agent receives its own approach term; it is intentionally not forced to zero-sum.
3. Keep the Campaign 02 PPO baseline, especially `entropy_coefficient=0.0`. Do not change any other PPO/model/observation/action/episode/self-play setting.
4. Run only the tiny targeted GPU reward-sign/reset-leakage smoke described in the handoff. **Do not run capacity preflight, initialization evaluation, inherited parity/regression gates, world-count sweeps, or repeated held-out evaluations before training.**
5. Immediately train from scratch at 131,072 worlds / horizon 32 through the first completed update crossing 100,000,000 agent decision samples.
6. Save resumable checkpoints at the first updates crossing 25M, 50M, and 100M.
7. After training, run one 4,096-world ordinary stochastic self-play evaluation using evaluation seed `920260826`. Compare final touch rate, goal rate, goal termination, and no-touch truncation directly with Campaign 02 final.
8. Publish compact Campaign 03 evidence and the final resumable checkpoint, then stop.

The point of this run is to see whether a dense car-to-ball approach signal materially reduces the roughly 99% no-touch timeout rate. Do not add curricula, another reward term, action masks, hyperparameter tuning, or v0.6 RocketSim/RLBot work in this run.
