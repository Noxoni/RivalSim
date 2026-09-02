# Rival 2 Codex autonomous campaign v1

This lane starts from the strongest measured human-derived closed-loop policy,
`human_bc_ppo_10h`, rather than the lowest offline-RMSE checkpoint. The fixed first
stage uses equal world probability for current-policy self-play and frozen Nexto,
native 120 Hz observations/actions, Gameplay 120 V2, coherent fixed exploration, and
eight small reviewed-human replay steps per accepted PPO update.

KL is recorded but never rejects or rolls back an update. Nonfinite loss, gradient, or
parameters remain a hard stop. Every five updates, a fresh deterministic 256-episode
Nexto evaluation and read-only human validation decide whether the checkpoint can
replace the preserved best model. The human test split is not loaded.

The pre-step authority is `authority.json`; `preflight.json` records the exact-scale
32,768-world no-optimizer validation.
