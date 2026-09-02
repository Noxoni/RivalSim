# Rival 2 Codex autonomous campaign v2

V2 freezes the only demonstration-replay dose that improved both read-only human
validation and deterministic closed-loop play: eight supervised steps from the
functional `human_bc_ppo_10h` policy. Repeating the same replay improved offline RMSE
but destroyed scoring, so competitive PPO performs no further human optimizer steps.

The exact selected human anchor is trained at native 120 Hz against equal-probability
current self-play and frozen Nexto. Exploration and likelihood use the same fixed
distribution. KL remains telemetry only; nonfinite state is a hard failure. Every PPO
update is checked by a deterministic 256-episode Nexto evaluation and read-only human
validation, while the best previously functional checkpoint remains preserved.
