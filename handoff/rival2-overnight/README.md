# Rival 2.0 Overnight Curriculum — Acquisition -> Base Reward -> 3h Run

This handoff authorizes one uninterrupted overnight curriculum beginning from the completed Campaign 04 1B checkpoint.

Required parent commit:

`4c121fab8c4bfe38fbf60f1c81a47d2dce898235`

Required starting checkpoint:

`checkpoints/rival2/campaign04/rival2_campaign04_1b_resume.pt`

Expected SHA-256:

`DB5AA09B2CAD40D4C1F5DB1014FDE245C58994A6948458212751724F782BE6B0`

Preserve the current 131,072-world / horizon-32 / entropy-0 PPO, model, observation, action, episode, and self-play settings throughout. Do not build the viewer and do not begin v0.6.

## Phase A — finish ball acquisition under Reward V2

Resume the exact Campaign 04 checkpoint and continue `RIVAL2_REWARD_V2` unchanged.

Do not run preflight, smoke, parity/regression, lint, compile, or unrelated tests.

Beginning after update 120, save and run the existing 4,096-world ordinary stochastic self-play evaluation every 30 PPO updates (about 251,658,240 additional agent samples per point). Use the same held-out protocol/seed used by Campaign 04.

The acquisition curriculum is complete only when:

`no_touch_truncated_fraction <= 0.01`

on **two consecutive evaluation checkpoints**.

Do not require literal zero. If a checkpoint rises above 1%, the consecutive count resets. Continue Reward V2 until two consecutive checkpoints satisfy the rule. There is no arbitrary sample cap on Phase A.

Preserve the confirming checkpoint as the **acquisition-complete checkpoint** and publish the Phase A touch/no-touch curve.

## Reward transition — remove only approach shaping

After Phase A is confirmed, remove only Reward V2's per-agent approach term.

The active reward becomes the already-existing preserved `RIVAL2_REWARD_V1` exactly:

- goal: +10 scoring side / -10 conceding side;
- canonical ball progress;
- unique touch: +0.05 / -0.05;
- unique demolition: +0.10 / -0.10;
- no approach-distance reward.

Do **not** invent `RIVAL2_REWARD_V3`; V1 is already the desired base reward.

Perform the transition at a PPO-update boundary. Preserve the learned model, optimizer state, CPU/CUDA RNG state, policy/opponent generator state, counters, historical-policy pool, and self-play state. The post-transition checkpoint must truthfully bind to `RIVAL2_REWARD_V1` and record the exact Reward V2 parent checkpoint/hash and the authorized V2 -> V1 curriculum transition. Do not silently bypass general checkpoint contract checking; implement this as an explicit reward-only curriculum migration/transition path.

No fresh initialization and no parameter reset are authorized.

## Phase B — 2B additional samples under base Reward V1

From the acquisition-complete policy after the explicit V2 -> V1 transition, train **239 additional PPO updates**, which is the first completed update crossing 2,000,000,000 additional agent decision samples:

`239 * 8,388,608 = 2,004,877,312 additional samples`

Keep every non-reward training setting unchanged.

Save and evaluate at these Phase B offsets from the reward switch:

- +60 updates / +503,316,480 samples;
- +120 updates / +1,006,632,960 samples;
- +180 updates / +1,509,949,440 samples;
- +239 updates / +2,004,877,312 samples.

Use the same lightweight 4,096-world stochastic self-play evaluation. Track at minimum touches/min, goals/min, goal fraction, no-touch fraction, episode duration, policy action distribution, and PPO stability.

At +239 updates, save a full resumable **2B-base-reward checkpoint** and verify it can be loaded. This is a required durable boundary, but do not stop the overall overnight run there.

## Phase C — immediately continue for 3 real wall-clock hours

Immediately after the Phase B 2B checkpoint/evaluation, continue training the same Reward V1 policy for **three real elapsed wall-clock hours**.

Start a monotonic wall-clock timer immediately after the Phase B boundary. Continue PPO updates until the first completed update at or after **10,800 seconds** elapsed.

Checkpoint and run the same lightweight evaluation at the first completed update at or after approximately:

- 3,600 seconds;
- 7,200 seconds;
- 10,800 seconds.

Checkpoint/evaluation overhead counts inside the three-hour real-clock interval; do not convert this phase into a predetermined sample target. Record the exact update, cumulative sample count, and elapsed seconds for each hourly point.

At the 3-hour boundary, save and publish the exact final full resumable checkpoint and final evaluation, then stop.

## Runtime behavior and stop conditions

This is a training run, not a release-certification exercise. Do not run the old preflight/regression/parity/lint/compile/test ceremony before or after the run.

Normal per-update finite/integrity checks already present in the trainer may remain. Stop early only for a genuine non-finite training failure, checkpoint corruption, impossible reward transition, or other condition that makes continued training invalid. Otherwise execute Phase A -> reward transition -> Phase B -> Phase C without returning for approval between phases.

## Evidence

Publish a compact overnight package containing:

- Phase A evaluation curve and exact acquisition threshold crossing;
- acquisition-complete checkpoint identity;
- explicit V2 -> V1 reward-transition record;
- Phase B 0.5B/1B/1.5B/2B evaluation curve;
- required 2B-base-reward checkpoint identity;
- Phase C 1h/2h/3h evaluation points with exact elapsed wall time and cumulative samples;
- final 3h checkpoint identity;
- compact PPO stability/throughput summary.

Do not build the viewer during this run. Do not change rewards again after the V2 -> V1 switch. Do not begin v0.6.
