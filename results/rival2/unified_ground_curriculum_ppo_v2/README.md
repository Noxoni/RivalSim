# Unified Ground Curriculum PPO V2

V1 demonstrated that starting directly with the zero-sum gameplay reward can
produce a reward-free symmetric opening while a large number of stale-rollout
minibatch steps moves the near-deterministic policy. V2 restarts from the exact
Unified V5 checkpoint and does not load that update.

Phase A remains pure current-policy self-play but uses the existing immutable
120 Hz acquisition reward. Its approach term supplies a physical, dense signal
before contact; its contact terms end when the deterministic evaluation shows
routine acquisition at two consecutive boundaries. Phase A is capped and
fails closed if the policy cannot acquire the ball.

Phase B creates a fresh optimizer and uses Gameplay 120 V2. Possession is the
existing proximity-times-velocity-match control score. Shooting is learned
from signed goalward progress and goals. A useful 50 is learned from retained
control, progress, recovery, and scoring consequences, not an inferred event
bonus. No named mechanic receives reward.

The policy-gradient learning rate is deliberately small, PPO uses one epoch,
and the critic-head learning rate is separate. Value loss cannot enter the
actor/shared trunk. KL remains telemetry only; nonfinite loss, gradients, or
parameters still trigger transactional rollback.
