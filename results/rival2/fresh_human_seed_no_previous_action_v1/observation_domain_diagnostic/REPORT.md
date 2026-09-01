# No-previous-action observation-domain diagnostic

Verdict: **MISMATCH**

This is a read-only diagnosis. No optimizer, reward calculation, physics step, or checkpoint mutation occurred.

## Group discrepancies

| Group | Mean absolute | RMSE (normalized contract units) | Max | Largest field |
|---|---:|---:|---:|---|
| ball | 0.000000 | 0.000000 | 0.000000 | `ball.position.x` |
| self_car | 0.021825 | 0.073977 | 0.968960 | `self.dodge_available` |
| opponent | 0.020278 | 0.062285 | 0.954534 | `opponent.dodge_available` |
| relative_state | 0.000000 | 0.000000 | 0.000000 | `relative.opponent_velocity.x` |
| boost_pads | 0.025464 | 0.038994 | 0.180614 | `boost_pad.14.active` |
| previous_action_zeros | 0.000000 | 0.000000 | 0.000000 | `previous_action.throttle` |
| lifecycle_timers | 0.143972 | 0.378003 | 1.000000 | `lifecycle.kickoff_reset` |

## Human-pipeline versus matched-native deterministic actions

Complete-action RMSE: `0.153282`; mean absolute difference: `0.067095`; maximum absolute difference: `1.000000`.

## Five native RivalSim kickoff outputs

| Layout | throttle | steer | pitch | yaw | roll | jump | boost | handbrake |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 diagonal_left | 0.417588 | 0.153481 | -0.024391 | 0.024973 | 0.076503 | 0.000000 | 0.000000 | 0.000000 |
| 1 diagonal_right | 0.399257 | 0.074898 | -0.115113 | -0.027016 | 0.064794 | 0.000000 | 0.000000 | 0.000000 |
| 2 off_center_left | 0.427156 | 0.142573 | -0.089591 | 0.019882 | 0.074176 | 0.000000 | 0.000000 | 0.000000 |
| 3 off_center_right | 0.392952 | 0.131489 | -0.096256 | 0.009001 | 0.074900 | 0.000000 | 0.000000 | 0.000000 |
| 4 center | 0.471742 | 0.140136 | -0.085923 | 0.019461 | 0.074991 | 0.000000 | 0.000000 | 0.000000 |

## Conclusion

The bounded evidence indicates both observation-domain mismatch and temporal/action ambiguity. Material observation groups: ['lifecycle_timers', 'self_car', 'opponent']. The five initial native kickoff outputs do not issue a strong throttle-plus-boost/jump kickoff command.

The complete field table, per-state inputs/actions, provenance, and integrity checks are in `diagnostic.json`.
