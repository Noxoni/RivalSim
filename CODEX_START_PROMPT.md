# Active Codex Handoff — Rival 2.0 Overnight Curriculum

Campaign 04 is complete at update 120 / 1,006,632,960 cumulative samples. This handoff authorizes one uninterrupted overnight curriculum.

Start from current `origin/main` and read `handoff/rival2-overnight/README.md` in full. Treat it as controlling.

Required completed Campaign 04 checkpoint:

`checkpoints/rival2/campaign04/rival2_campaign04_1b_resume.pt`

Expected SHA-256:

`DB5AA09B2CAD40D4C1F5DB1014FDE245C58994A6948458212751724F782BE6B0`

Mission:

1. resume Reward V2 unchanged from the exact 1B checkpoint;
2. evaluate every 30 updates until no-touch truncation is <=1% on two consecutive 4,096-world held-out evaluations;
3. preserve that acquisition-complete checkpoint;
4. explicitly transition only the reward from `RIVAL2_REWARD_V2` to the existing base `RIVAL2_REWARD_V1`, preserving learned/optimizer/RNG/counter/historical state and recording the authorized reward-only migration;
5. train 239 additional PPO updates under Reward V1 (2,004,877,312 additional samples), with evaluations at +60/+120/+180/+239 updates;
6. save the required full resumable 2B-base-reward checkpoint;
7. immediately continue Reward V1 training for three real elapsed wall-clock hours, checkpoint/evaluate at approximately 1h/2h/3h, and stop after the first completed PPO update at or after 10,800 elapsed seconds;
8. publish the compact overnight evidence package and final resumable checkpoint.

Do not run preflight, smoke, parity/regression, world-count, pytest, Ruff, compileall, or other release-certification ceremony. Do not build the viewer. Do not alter PPO/model/observation/action/episode/self-play settings. Do not make any reward change other than the single authorized V2 -> V1 approach-removal transition. Do not begin v0.6.

Execute all phases without returning for approval unless a genuine failure makes continued training invalid.
