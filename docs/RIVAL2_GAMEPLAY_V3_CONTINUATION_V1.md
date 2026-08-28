# Rival2 Gameplay V3 production continuation (+120 target)

Status: `STOPPED_HARD_SAFETY_GUARD_AT_PROPOSED_UPDATE_581`.

Source: iteration `489` / `10D97428B3F1CC2E307040314D1DD1A924BD82975D4B88C0F73C3FC2716DCF54`.

Last accepted model: iteration `580` / `F3CBD88FE611FC7D8C69FDD6E7D1203336167BC96A92E0E0111F9C8B64916FC6`.

The configured hard PPO safety boundary fired on proposed update `581`. Training stopped immediately; the guard was not weakened and no later update ran.

## Hard-stop diagnostic

- Reason: `minibatch_kl_limit_exceeded`.
- Post-step minibatch KL: `0.199769646` (hard limit `0.100000000`).
- Retention mean KL: `0.081041917`.
- Transactional rollback completed: `True`.
- Parameters restored exactly: `True`.
- Optimizer restored exactly: `True`.
- Adam counters restored exactly: `True`.

## Accepted-update PPO safety

- Accepted updates: `91`.
- Maximum accepted minibatch KL: `0.019802354`.
- Maximum completed-update mean KL: `0.018605746`.
- Maximum retention mean KL: `0.019957421`.
- Retention-budget early stops: `17`.

## Reward scale across accepted updates

- Mechanics / absolute gameplay: `0.001363766`.
- Bad flip / absolute gameplay: `0.003001068`.
- Mechanics / progress: `0.012301644`.
- Bad flip / progress: `0.027070669`.
- Maximum single-update mechanics / gameplay: `0.163548959` at iteration `580`.
- Maximum single-update bad flip / gameplay: `0.052452665` at iteration `552`.

The mechanics maximum occurred on the first fresh-simulator rollout after the +90 boundary. That rollout had no ball touch, no progress component, and no completed episode, but did detect 7,580 pogo events. This boundary transition is an investigation signal; it is not presented as ordinary steady-state reward composition.

## Completed fixed-context Gameplay V3 shadows

| iteration | touches/min | flip touches/min | bad/min | bad/flip | mechanics/progress | bad/progress | Rival goal share | no-touch |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 489 | 18.949230 | 12.816724 | 6.012260 | 0.469095 | 0.013446 | 0.027325 | n/a | n/a |
| 519 | 20.473921 | 11.818386 | 5.564273 | 0.470815 | 0.013183 | 0.024286 | 0.521008 | 0.000000 |
| 549 | 19.445931 | 12.413009 | 5.995717 | 0.483019 | 0.013671 | 0.026936 | 0.449580 | 0.000000 |
| 579 | 19.045746 | 12.732792 | 6.312953 | 0.495803 | 0.014629 | 0.028879 | 0.520661 | 0.000000 |

Scheduled checkpoint/evaluation boundaries at iterations 519, 549, and 579 completed green. The iteration-609 boundary was not reached.
