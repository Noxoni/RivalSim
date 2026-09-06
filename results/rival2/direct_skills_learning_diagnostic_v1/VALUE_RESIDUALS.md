# Secondary analysis: critic residual is not an established sole cause

This reuses the committed 2048-world matched-trajectory NPZ. Zero new simulator
rollouts, optimizer steps, or model changes; CPU-only analysis. Six focused
tests passed, including the distinction between bias and explained variance.
Running the analysis twice produced byte-identical JSON.

The data compares parent 650 with the **previous kickoff child +15**, not any
checkpoint from the current log-barrier experiment. Both saw the same recorded
parent-generated trajectories. Values below are against the same bootstrapped
GAE targets, not independent full-episode Monte Carlo outcomes.

For the 13,716 recorded kickoff/Nexto samples (correlated trajectory frames):

| Statistic | Parent | Previous kickoff child +15 |
| --- | ---: | ---: |
| Mean value prediction | 1.35201 | -1.15750 |
| Mean fixed target | -0.73016 | -0.73016 |
| Target minus prediction | -2.08217 | 0.42734 |
| Fixed-target RMSE | 2.75182 | 0.91043 |
| Explained variance | -2.71352 | 0.25861 |

The initial prediction was too optimistic relative to these targets. However,
the previous child substantially adapted its critic while still failing to
improve full matches (previous child: 5 scored / 209 conceded; parent: 7 / 196).
This observation argues against treating an unchanged or stuck critic as the
established sole explanation, or claiming that a critic-only upgrade solves it.

Opponent-family advantage normalization pools the five reward roles. In these
saved parent samples, mean normalized advantage ranges from -0.63156 for
kickoff/Nexto to +0.39211 for challenge/Nexto. That is a measured difference, not
proof that role-specific centering would improve policy learning. A negative
advantage is relative to the baseline, not an intrinsic judgment of an action.

Important limits: these targets bootstrap from the parent; fixed-target child
error is not an unbiased estimate of the child's own return-prediction error.
No current log-barrier child's value estimates were measured. Neither this
analysis nor the earlier saturation finding establishes a complete causal
explanation for failed gameplay transfer. No training setting was changed.

Reproduce with:

```powershell
.venv\Scripts\python.exe -B benchmarks/summarize_direct_skills_value_residuals_v1.py
```

`value_residuals.json` contains all ten role/opponent groups, the exact source
NPZ hash and checkpoint identities. Historical raw records remain unchanged.
