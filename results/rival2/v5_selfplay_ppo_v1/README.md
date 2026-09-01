# V5-rooted self-play PPO V1

This directory holds the prospective authority and bounded evidence for the
pure current-policy, two-sided 120 Hz PPO campaign rooted in Human BC V5.

The training entry point is
`benchmarks/run_rival2_v5_selfplay_ppo_v1.py`. It verifies the committed
authority, source checkpoint, reward contract, empty historical pool, fresh
optimizer, full-model trainability, and pure-current assignments before the
first rollout. The first rollout then proves that every trainable sample comes
from the current policy on both canonical perspectives.

The authoritative checkpoints are written to
`checkpoints/rival2/v5_selfplay_ppo_v1/`; a rolling operational checkpoint is
kept outside Git under `G:/dev/RivalSim-runs/v5-selfplay-ppo-v1/`.
