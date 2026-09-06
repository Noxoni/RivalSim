# Narrow follow-up: clear residual handbrake at native kickoff

This plan follows the completed fixed-action causal isolation in RESULTS.md.
Do not rerun that completed prefix/trace, or the old550/650 diagnoses. Keep675
policy/Adam/checkpoint unchanged and the agent675 review STOP during this work.

1. Add `vehicle.handbrake_value` to the native standard-kickoff reset arguments
   and set selected cars' values to exactly0 alongside wheel cache invalidation.
   Do not clear unrelated buffers, change collision physics, first-tick timing,
   quaternion construction, controls, Nexto cadence or rewards.
2. Focused native validation: all five layouts, both cars, selected/unselected
   worlds, residual handbrake values including0..1. Verify selected cache clear,
   unselected arrays unchanged, and fixed-action replay matches a private
   handbrake-only-cleared reference. Check no observation/contract shape change.
3. The scenario template already clears handbrake. Compare complete native
   curriculum-reset output arrays before/after for the same states, masks and
   source bank, including unselected worlds; require exact parity. If not exact,
   stop and investigate rather than claiming training semantics unchanged.
4. Publish code/tests and a new versioned fullmatch method/updated runtime
   authority that explicitly supersedes the old source hashes. Preserve old
   authorities/results, never rewrite their identities to make them pass.
5. Run only a bounded same-checkpoint corrected675 natural evaluation and a
   corrected600 reference (the better recent prior checkpoint) under the SAME
   new method. Include goals/concedes, first contacts, follow-ups and continuous
   play. No optimizer, selection trick, extra reward or large new benchmark.
   Runtime-vs-learning effects must be reported separately. Skill training
   objectives and original64case skill reference remain unchanged.
6. Decide the next finite training block prospectively from actual corrected
   natural outcomes. The650to675 keeper-pressure curriculum did not demonstrate
   transfer; do not blindly extend it or treat an intentional review pause as
   a crashed worker. Do not resume older expired runners or change lineage
   without an explicit prospective decision. The ongoing SSL goal remains unmet.
