# Human BC V4 blocked review

Human BC V4 stopped before accepting a supervised validation boundary and before
opening either frozen test authority. No V4 checkpoint was selected or promoted.

## Frozen outcome

- Parent: Human BC V1 step 160, SHA-256
  `560C2414C17039DC920126EA148BF73FE6CC4677EE440F043599A7E1C76D2874`.
- Proposed actor-only optimizer steps: 256 across four exact transactional retries.
- Accepted supervised optimizer steps: 0.
- Retry learning rates: `3e-5`, `1.5e-5`, `7.5e-6`, `3.75e-6`.
- Every retry was restored to the exact interval-start model, optimizer, RNG, and
  hard-tail state.
- Human test accesses: 0.
- Untouched V4 simulator-test student evaluations: 0.
- PPO updates and reward changes: 0.

## Stop condition

Every proposal passed complete and stress retention, finite-output, critic-parity,
selection-margin, frozen trunk/critic, and human distribution-health checks. Across
the retry sweep, complete all-perspective maximum sample KL ranged from
`0.0077860784` to `0.0275878996`; stress ranged from `0.0094721379` to
`0.0310448941`. The largest corresponding all-perspective mean KL values were
`0.0004693181` complete and `0.0004609338` stress.

Every proposal failed only the prospectively frozen simulator boost-saturation
check in all-perspective, current-policy-applicable, counterfactual-opponent,
teacher-low-variance, and orientation-sensitive groups. The fixed threshold was
`0.95`; the highest-learning-rate proposal remained at `0.9613189697` complete
all-perspective and `0.9603033066` stress all-perspective, with the
orientation-sensitive fractions at `0.9831181765` and `0.9806162119`.
Historical-opponent boost saturation remained below the limit.

The original BC V1 parent also fails this absolute simulator boost-saturation
threshold at zero KL. The pretraining admission bug was corrected before the first
proposal so training could attempt to improve the inherited condition. The trained
candidate and test guard itself was not weakened. Four retries did not cross it, so
the authority correctly rolled everything back and stopped.

## Interpretation and next authority

This V4 run is blocked by the frozen candidate distribution-health authority, not
by actor KL instability, critic drift, nonfinite state, human data, or the V4 tail
coverage machinery. Because candidate validation outcomes have now been observed,
this V4 authority must not be retuned and training must not resume under modified
V4 thresholds.

If another campaign is authorized, define a new prospective version before any
optimizer proposal. Its simulator distribution-health contract should be explicitly
BC-V1-parent-relative (or should separate inherited boost saturation from newly
created saturation) while preserving the unchanged all-perspective maximum
per-sample actor KL limit of `2.0`. Do not use any V4 test outcome: both V4 test
authorities remain sealed.

Machine-readable details are in `evidence.json`, `training_curve.json`,
`pre_optimizer_baseline_diagnostic.json`, and `artifact_manifest.json`.
