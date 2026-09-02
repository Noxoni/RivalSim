# Rival 2 Codex autonomous campaign v6

V6 is a bounded four-branch policy-gradient search from the same promoted V4 model.
The branches differ only in their prospectively listed rollout/curriculum seed. Each
uses a fresh optimizer, 50/50 current self-play and Nexto worlds, an analog-only
training boundary, one optimizer minibatch per fresh rollout, and at most three
microsteps. The global selection order is deterministic Nexto goal differential,
goals, touches, then first touches; all human-validation and frozen-boundary checks
remain active.
