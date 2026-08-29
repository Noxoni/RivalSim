# Rival human BC continuation V1

This is the prospectively frozen continuation of the selected human-BC V1
checkpoint. It resumes the checkpoint's fresh supervised AdamW state and keeps
the frozen adapter, native 120 Hz targets, human splits, gameplay/mechanic family
balance, label -> attempt -> frame mechanic sampling, hybrid supervised loss,
simulator retention objective, validation score, and hard retention limits
unchanged.

The continuation has no normal fixed-step stopping target. It stops on a clear
validation plateau, exhaustion of the inherited transactional hard-guard retry
budget, or a genuine failure. The 8,192-additional-step ceiling is an emergency
resource boundary; reaching it while validation remains materially improving is
a blocked result rather than successful convergence.

No closed-loop mechanic framework or Rocket League-to-RivalSim mechanic-state
reconstruction is part of this task. The human and simulator test splits are
opened once only after a final validation checkpoint has been selected.
