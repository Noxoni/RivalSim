# Standard kickoff contact-cache correction

The source bug and measured input/action effect are documented in the +450
review and immutable +300 trace. The prospective correction plan was published
and all 40 milestone files remotely read back at
`a1509658ce3b2e46542f4f621e3b8481fd4a990f` before kernel editing.

Only the existing standard native reset receives the wheel-contact buffer and
zeros four wheel flags per selected car. Non-reset cars are untouched. This
invalidates pre-teleport contacts, matching construction and curriculum reset;
it is not a fabricated measured grounded state or a change to contact physics.
No rewards, coefficients, task events, opponent schedule, action/observation
schema, architecture, PPO settings or model weights change.

Native validation passes all five kickoff layouts, both policy perspectives,
poisoned contact flags, repeated resets, unchanged unselected observations and
exact deterministic +300 action parity with fresh kickoff observations. The
existing direct-skills goal-on-each-of-four-ticks, reset, reward, truncation,
bootstrap, Nexto-sample-mask and ground-shot checks pass: 17 focused tests total.

A pre-edit native fixture captures 246 training observation/state/vehicle/
lifecycle arrays across 32 worlds (16 resetting and 16 unaffected). Every array
is bit-identical after correction. Direct Skills training already cleared the
wheel cache through its scenario reset; the measured training-reset semantics
are unchanged. The fixture collector initially referenced wheel_contact on
GpuState rather than vehicle; that helper-only AttributeError occurred before
capture and was corrected before the old-kernel fixture was frozen. It was not
a training failure or a failed physical parity test.

Full-match standard resets do change. Preserve all old results through +450.
Future results carry `RIVAL2_STANDARD_KICKOFF_CONTACT_CACHE_RESET_V2` and runtime
package identity. Run corrected fixed parent+0 and latest+451 full matches to
establish a comparable baseline. Never attribute old-to-corrected score changes
solely to learning. The original reward/PPO authority remains unchanged and the
package explicitly binds this operational correction and preserved prior package.

Resume only the intact latest +451 model/Adam/RNG, not +300 used for diagnosis.
The correction does not imply shooting or possession is solved and is not an
SSL or deployment promotion. Independent Nexto cadence/history differences
remain possible influences; they have not been modified speculatively.
