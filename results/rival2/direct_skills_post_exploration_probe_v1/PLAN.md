# One no-update post-run gradient check

Run only after the exploration campaign completes at +30 and its actual worker
exits. Checkpoint: `checkpoints/rival2/direct_skills_log_barrier_v1/child_000030.pt`,
SHA `16BF2B904785D49E0363B869AF953CFCCC62FF754286A307D386729AC691DDAD`.

Question: has the frozen beta 0.01 exploration regularizer become large relative
to the PPO reward-learning gradient after 30 updates? No assumption that it has.

Reuse the original calibration implementation unchanged: 1024 worlds, 90
decisions, same frozen kickoff/scenario/reward/Nexto definitions, seed 2026090601,
Nexto seed 2026090691, eight deterministic 128-sequence microbatches chosen with
seed 2026173602 (scenario seed +83001), T2 on-policy sampling and likelihood.
Measure actor parameter gradient norms and cosine alignment; exclude the critic.

Construct no optimizer, take no optimizer step, change no policy/reward, and
rerun no match evaluation. Use the exclusive GPU lease. Retain checkpoint byte
identity. Save new evidence here; never overwrite original calibration evidence.

The original initialization diagnostic criterion (median added/PPO norm <=0.5,
maximum <=1.0) is a reference for this probe, NOT a new runtime rejection guard.
No coefficient or next campaign is selected from the raw candidate table.
The eight new microbatches are not the historical update-30 minibatches, and a
gradient ratio alone is not a causal explanation of gameplay performance.

Commit/push this plan and implementation with the completed +30 evidence before
the probe. One bounded rollout and backward-only probe; no repeated search.
