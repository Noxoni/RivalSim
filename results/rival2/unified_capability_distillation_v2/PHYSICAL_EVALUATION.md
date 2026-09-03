# Unified Capability Distillation V2 Physical Evaluation

V2 is one recurrent policy with one actor output. It uses no runtime router,
expert action splice, scenario id, or task id.

## Outcome

The capability actor transfer is physically successful. Against identical
controlled seeds, the unified policy closely reproduces the specialist:

- actual demos: 394 unified / 394 specialist;
- demo follow-up touches: 378 / 376;
- demo follow-up goals: 289 / 290;
- productive floor landings: 78 / 83;
- productive wall landings: 124 / 121;
- productive landing chains: 178 / 182.

The aerial transfer is strong but incomplete. Across 2,048 attempts, the
unified policy recorded 43.60% elevated follow touches, 20.70% high follow
touches, 4.10% second airborne touches, and 9.03% productive continuations.
It scored 1.22% goals within the contact budget, below the frozen 1.50% gate;
the specialist control scored 2.25% on the same seeds.

The natural closed-loop result is not promotable. Rival touched the ball in all
256 bounded Nexto episodes, produced 1,400 touches and 1,099 forward contacts,
and had zero no-touch truncations, but lost every episode 0-256. The tiny
teacher-rollout natural RMSE is therefore insufficient to prevent compounding
closed-loop error.

## Verdict

`PARTIAL_NOT_PROMOTED`

Preserve V2 as evidence that one neural network can retain the controlled
demo/dash behavior and most aerial behavior. The next prospective correction
must rehearse V23 actions on states induced by the unified student's own
closed-loop natural trajectories. It must not introduce a runtime router.
