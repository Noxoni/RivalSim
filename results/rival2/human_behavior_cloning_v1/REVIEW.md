# Rival human behavior cloning V1 review

Verdict: **PASS**. The selected checkpoint is the accepted step-160 boundary.
The bounded campaign stopped safely when the next 32-step boundary exceeded the
frozen critic maximum-absolute-drift guard. The selected checkpoint predates that
rejected boundary; its simulator validation and test retention both pass every
frozen hard limit.

## Selected artifact

- Checkpoint: `checkpoints/rival2/human_bc_v1/rival2_human_bc_v1.pt`
- SHA-256: `560C2414C17039DC920126EA148BF73FE6CC4677EE440F043599A7E1C76D2874`
- Model tensor SHA-256: `872731CA5FD5FE148242A709D954CD1FFEF5249075811E79A15B93420AF0DAD7`
- Accepted supervised optimizer steps: 160
- PPO-resumable: no

## Open-loop imitation

| Split/family | Bootstrap complete-action RMSE | Selected RMSE | Relative change |
| --- | ---: | ---: | ---: |
| Validation gameplay | 0.621598 | 0.560036 | -9.904% |
| Validation mechanics | 0.616808 | 0.535502 | -13.182% |
| Test gameplay | 0.599077 | 0.541930 | -9.539% |
| Test mechanics | 0.646498 | 0.559705 | -13.425% |

All 13 mechanic labels represented in validation improved. Full per-channel,
per-label, button BCE/accuracy, analog MAE/RMSE, and log-standard-deviation drift
tables are in `training_curve.json`, `final_test_metrics.json`, and `evidence.json`.

## Retention and gameplay sanity

- Selected simulator-validation actor mean KL: 0.003291.
- Final simulator-test actor mean KL: 0.003320.
- Simulator-test actor maximum sample KL: 0.603020 (hard limit 2.0).
- Simulator-test critic RMSE: 0.019568 (hard limit 0.075).
- Simulator-test critic maximum absolute drift: 0.482265 (hard limit 0.5).
- Mixed-opponent touch rate: 8.318710 -> 8.148766 per simulated minute
  (0.979571 ratio).
- Mean normalized movement speed ratio: 1.001594.
- Analog saturation fraction: 0.0000521 -> 0.0000467.
- The 256-tick/world sanity horizon observed no goals, concedes, or truncations in
  either arm; those zero event counts are reported without extrapolation.

## Closed-loop mechanic boundary

No held-out native mechanic start can be mapped exactly into RivalSim. Native
recordings lack source-exact RivalSim lifecycle/contact history and cross-engine
contact-manifold/integrator state. Consequently all mechanic labels are reported
`NOT_EVALUABLE_EXACTLY`; no approximate start was invented, no human action was
replayed, and no mechanic is claimed learned solely from open-loop improvement.

## Guard-stop disclosure

The proposed step-192 boundary failed only the critic maximum-absolute-drift
guard. Transactional rollback preserved step 160 and no hard limit was weakened.
The first retry used the configured 1.5e-5 LR; rollback restored the interval-start
LR before retries two and three, so those retries duplicated the first. This did
not alter the selected checkpoint, which existed before all rejected step-192
attempts. The exact attempt telemetry is retained in `training_curve.json` and
the finalization audit in `evidence.json`.

The next safe research step is an explicitly separate, no-learning native-to-
RivalSim starting-state fidelity investigation. It should determine whether any
mechanic family can support an exact closed-loop evaluation before authorizing
further BC or any PPO transition.
