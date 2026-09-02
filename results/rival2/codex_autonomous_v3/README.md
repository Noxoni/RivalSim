# Rival 2 Codex autonomous campaign v3

V2 proved that one nominal PPO boundary—192 minibatch optimizer steps—was too coarse
for the selected deterministic human anchor even at a 5e-7 policy learning rate. V3
keeps the exact same parent, reward, opponents, full policy boundary, and KL-as-
telemetry policy, but accepts exactly one optimizer minibatch step per fresh exact-
scale rollout. The pilot evaluates every microstep and preserves the anchor unless
deterministic closed-loop Nexto play genuinely improves.
