# Rival Human Behavior Cloning V2

This lane continues only the selected Human BC V1 policy with a fresh AdamW
optimizer over `actor.weight` and `actor.bias`. The shared trunk and critic are
frozen and audited for exact tensor identity at every accepted validation
boundary. The native recordings, frozen split manifest, observation bridge,
observation adapter V2, action targets, supervised objective, mechanic-aware
sampling, and simulator actor-retention limits are inherited unchanged.

The human test split is loaded once only after validation checkpoint selection.
This lane performs no PPO, reward, detector, mechanic-definition, demonstration,
dataset, split, adapter, or observation/action-contract changes.
