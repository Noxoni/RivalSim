# Corrected native-v5-controller baseline and selected learning parent

Both full ten-match comparisons completed normally under prospective commit
`02c78cb30b7273014db3b4ceba985d7be665c675`, before any optimization. Each ran
36,000 regulation ticks per world with five standard layouts on both sides.
Neither required overtime. Both policies and pinned Nexto weights stayed exact.
All existing match-integrity checks passed. Full per-case telemetry and score
events are in the named JSON files and their integrity reductions.

| Policy | Wins/losses | Goals for/against | Rival contacts | Contacts/min | First kickoff contacts |
|---|---:|---:|---:|---:|---:|
| Reference600 | 0/10 | 5/204 | 475 | 9.50 | 1 |
| Finishing650 | 0/10 | 7/196 | 456 | 9.12 | 0 |

Every world recorded a Rival contact. No-touch resets are disabled in this
regulation protocol; their absence is not proof Rival learned persistence.
Same-player followups were 130/466 (27.90%) and 123/444 (27.70%), respectively;
these are contact identities, **not** possession-duration measurements.

The frozen ranking selects **finishing650**, with goal difference -189 versus
-199 after equal wins. The difference is small; neither policy is competitively
adequate and no SSL/native-play promotion is made. This is development parent
selection, not a held-out claim or a trained improvement between these runs.

Selected source remains:

`checkpoints/rival2/direct_skills_finishing_goal_v2/child_000025.pt`

SHA-256 `939213BF57FFC751C7C173FFCC8AC34DE84D2CAA66CFECB79C59792CBABB0F6D`.

These numbers differ from the previous deterministic timing-only comparison
because this evaluator explicitly uses native-v5 kickoff sampling and production
admission semantics. They must not be presented as a regression caused by
learning: there were no optimizer steps in either baseline. The old immediate-
action simulator's 8/10 wins remain legacy-opponent evidence only.

## Training readiness

The selected parent passed all 14 zero-step preflight checks on a real
1,024-world/90-decision rollout: exact source/model/Adam/counters, finite loss
and gradients, isolated critic, exact action targets and learner masks,
corrected controller, unchanged finishing reward and absent mechanics hot path.
No optimizer step was taken by this preflight.

The final focused CPU tests passed 36 tests in 6.32s. Three pre-existing GPU
reward tests were explicitly deselected while the baseline GPU worker ran;
they were not claimed as newly run. The actual new training integration has
the completed GPU preflight plus the prior 32,768-world controller validation.

`training_authority.json` and `training_package.json` freeze a real +25-update
learning block with evaluations at +10 and +25, unchanged rewards/scenarios/
PPO/T2 exploration, and the corrected Nexto opponent. Model/Adam/four RNG start
from the selected parent; new physical episodes/zero hidden and fresh seeded
native-opponent RNG are explicit. Every accepted update is durably saved.
This finite block is a measurement/review boundary, not completion of the SSL goal.
