# Rival 2.0 Campaign 02 — Entropy-Off Controlled Rerun

Campaign 02 is authorized on top of completed Campaign 01.

The single intended learning change is `entropy_coefficient=0.0` instead of Campaign 01's `0.01`. All other Rival 2.0 simulator, observation, action, reward, episode, model, PPO, self-play, seed, world-count, rollout, checkpoint, and evaluation semantics are held fixed as specified by this handoff.

Read `DIAGNOSIS.md`, `CAMPAIGN.md`, and `ACCEPTANCE.md` as controlling documents. Do not treat this as permission for reward tuning, curricula, model changes, action changes, or v0.6 transfer work.

The purpose is to produce a clean controlled answer to whether Campaign 01's entropy bonus drove the policy toward excessive randomness. Preserve Campaign 01 evidence unchanged, train a fresh policy through the same bounded 100M sample target, compare directly, publish the outcome, and stop.