# Requested update-20 evaluation

Update 20 was evaluated with the existing deterministic 1,024-world,
3,600-tick protocol: 512 worlds each against Nexto and frozen Unified V5.
These are aggregate scenario goals, not full-match scores or win rates.
The exact pre-amendment update-10 evaluation is the comparison baseline.

| Opponent / metric | Update 10 | Update 20 |
| --- | ---: | ---: |
| Nexto goals for / against | 185 / 923 | 174 / 946 |
| Nexto goal differential | -738 | -772 |
| Nexto touches per minute | 13.609375 | 13.56640625 |
| Nexto no-touch resets | 0 | 0 |
| Nexto goalward-touch fraction | 0.87772675 | 0.88828103 |
| Nexto mean speed (uu/s) | 1192.313647 | 1194.205447 |
| Frozen V5 goals for / against | 362 / 375 | 400 / 386 |
| Frozen V5 goal differential | -13 | +14 |
| Frozen V5 touches per minute | 16.0078125 | 15.50390625 |
| Frozen V5 no-touch resets | 1 | 2 |
| Frozen V5 goalward-touch fraction | 0.82259639 | 0.85941043 |
| Frozen V5 mean speed (uu/s) | 1184.560331 | 1186.038270 |

## Interpretation

Mixed short-interval results, not a demonstrated broad improvement. The
Nexto goal differential worsened by 34; the frozen-V5 differential improved
by 27. Ball acquisition and movement speed show no major gain. Goalward-touch
fraction improved against both opponents, but that is not proof of improved
possession, kickoff technique, aerial competence, or reduced backward driving.
Those behaviors are not directly measured by this evaluation.

The ten new updates used 169,122,582 trainable agent samples and 117,964,800
world physics ticks, in 923.471 seconds of recorded update time. Cumulative
training samples reached 227,954,380. Each new update has 2.8125 times the
physics exposure of a previous 128-tick update. This sequential comparison
does not isolate a causal benefit of longer rollouts/advantage traces and
does not establish statistical significance from a single aggregate run.

## Integrity and continuation

- Exact saved update/policy version: 20; checkpoint SHA-256:
  `8CD78E5AE377158F133DA07B59D42A1C19A615F2A5C44505939EAD7B1852D8D8`.
- Immutable artifact: `checkpoints/rival2/ssl_foundation_v5_long_trace_v1/evaluation_u0020.pt`.
- Model/optimizer finite checks, 7,220 additional Adam steps, contiguous
  accepted boundaries, sample/tick accounting, and unchanged PPO/contracts
  all pass; see `evaluation_u0020_checkpoint_audit.json`.
- No reward, opponent mix, learning rate, rollout length, or GAE changes were
  made for this evaluation. No KL rejection was introduced.
- The existing accepted-boundary stop/final evaluation/resume path was used.
  Resume preserves saved model, Adam, counters and RNG; as in the existing
  runner, simulator episodes restart with zero recurrent hidden state.
- The controller resumed the exact immutable update-20 artifact under the
  original deadline. Regular update-50 and update-100 evaluations remain;
  the campaign still stops at total update 100 or its original deadline.

Full precision is in `evaluation_u0020_comparison.json` and the snapshot
manifest. The controller's two focused tests passed; Ruff lint/format and
`git diff --check` passed. Tests cover exact in-flight boundary selection and
matched-protocol comparison; checkpoint auditing separately verifies the real
production artifact. Execution logs remain in the external campaign directory.
