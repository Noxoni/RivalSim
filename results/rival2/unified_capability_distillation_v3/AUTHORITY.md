# Unified Capability Distillation V3 Authority

V3 starts only from the selected V2 single-network checkpoint. The V2 physical
evaluation proved controlled demo/dash transfer, strong but slightly incomplete
aerial transfer, and a natural closed-loop failure caused by compounding error.

Before optimization, V3 freezes a DAgger-style correction: collect natural
trajectories from the V2 student itself against pinned Nexto, label every state
with the side-correct V23 teacher, and rehearse those sequences jointly with
fresh natural-teacher, aerial, demo, floor-landing, and wall-landing sequences.

Only the unified recurrent context modules may change. The capability-baked
feed-forward base remains frozen. Teacher identity is training metadata only;
the deployed policy receives one observation stream and emits one actor output.
There is no runtime router, task id, scenario id, or expert action splice.

This is supervised correction only. PPO, reward changes, simulator changes,
mechanic detectors, and official promotion are not authorized by this file.
