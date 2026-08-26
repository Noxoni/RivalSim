# Campaign 01 Diagnosis — Controlling Hypothesis for Campaign 02

Campaign 02 is authorized because Campaign 01 exposed a specific optimizer pressure that can be isolated without changing Rival 2.0's environment, reward, action, or model contracts.

## Observed Campaign 01 behavior

Campaign 01 finished all 12 updates correctly but degraded behaviorally:

- ordinary stochastic self-play touches/minute: `0.272091 -> 0.175624`;
- goal-terminated fraction: `0.006348 -> 0.003418`;
- no-touch truncation fraction: `0.993652 -> 0.996582`;
- final stochastic result vs initialization: `7-23`, goal differential `-16`;
- final stochastic touch differential vs initialization: `-46`;
- final deterministic result vs initialization: `0-819`;
- final deterministic touch differential: `-819`.

The final stochastic analog policy standard deviations were approximately `2.64-2.65`, close to the configured `exp(1) = 2.71828...` ceiling implied by `log_std_max=1`.

## Source-level mechanism

The frozen policy implementation reports hybrid entropy using:

- analytic entropy of the five **pre-tanh Gaussian** channels; plus
- Bernoulli entropy of the three digital controls.

The PPO objective is:

`policy_loss + value_coefficient * value_loss - entropy_coefficient * entropy`

With Campaign 01 `entropy_coefficient=0.01`, increasing Gaussian `log_std` directly decreases total loss through the entropy term until the clamp is reached. This entropy diagnostic does not account for the tanh-squashed action distribution's bounded support, while the actual controller receives `tanh(u)`.

Campaign 01 metrics are consistent with that pressure:

- update 1 entropy approximately `9.216`;
- update 12 entropy approximately `14.162`;
- analog standard deviations climbed from approximately `1.0` at initialization to approximately `2.65` at the final checkpoint;
- final stochastic mean absolute analog actions were approximately `0.80` across throttle, steer, pitch, yaw, and roll;
- button probabilities remained close to the maximum-entropy region around `0.5`.

The entropy term therefore became much larger in magnitude than the measured policy-loss term and strongly rewarded exploration magnitude.

## Additional instability marker

Campaign 01 update 4 recorded:

- approximate KL approximately `1.085`;
- clip fraction approximately `0.617`.

This was an unusually large policy movement. Campaign 02 does not add KL early stopping or tune PPO around this event because the purpose is to isolate the entropy coefficient first. It must, however, record the same metrics so the event can be compared directly.

## Campaign 02 hypothesis

Set `entropy_coefficient=0.0`, leaving everything else unchanged.

The existing entropy calculation remains a diagnostic only and must not contribute to gradients/loss weighting in Campaign 02.

A positive Campaign 02 result would show that removing this pressure prevents the runaway analog standard-deviation trend and improves behavioral metrics relative to Campaign 01 and preferably relative to initialization.

A negative Campaign 02 result is also useful: it would indicate that entropy pressure was not the only material training problem, and subsequent work should then examine reward sparsity, PPO update scale, or other learning design choices separately rather than changing multiple variables at once.