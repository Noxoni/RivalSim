# Verified continuation launch

Authority/source/preflight commit `10ad0f53d5d6ee61ee50debb64e2ca7992206020`
was pushed and read back from `origin/main` before this campaign launched at
2026-09-05 21:10:30 UTC. The hidden launcher PID was 616; actual Python learner
PID 7976. These are launch observations, not a claim those PIDs remain alive
indefinitely. Inspect the current command line, state and logs for live status.

`launch_integrity.json` independently loaded the zero-continuation checkpoint
on CPU and proved exact +100 model, Adam moments, parameter groups and 18,200
step counters. The original +100 file remained byte-identical. Physical
episodes/hidden state restarted as declared; no old optimizer was substituted.

`accepted_launch_integrity.json` preserves the first two real continuation rows
(+101 and +102), their PPO diagnostics and a CPU-only accepted checkpoint audit.
All audited model/Adam tensors were finite; the model changed; Adam counters
advanced to 18,564; the parent remained unchanged. No optimizer step was run by
the audit. Nexto probability remains zero and the competence streak is one.

Preserved accepted checkpoint:

`checkpoints/rival2/ssl_entity_continuation_v1/launch_accepted_000102.pt`

SHA256: `6B5F698E8FCADCBE94F731DE4E680031D799E1AF8A6B7E595AF51D7F8F3C280D`

Training continues from this same lineage with rolling checkpoints every update.
The next scheduled deterministic evaluation is +150. **The high contact/goal
rates in the first freshly reset rollout are not a before/after learning
claim:** synchronized easy scenario starts differ from steady-state rollouts.
Use the fixed development evaluations and completed full-match comparison for
behavioral progress. No continuation evaluation has completed at this launch
milestone.

The existing ten-minute monitor was updated, not duplicated. It points to this
runner/authority/external state, preserves user-stop and numerical protection,
and reports only newly completed evaluations or actionable findings/failures.
The notification cursor begins at100, including the already reported full-match
result. The original pilot and old negative experiments must not be resumed.

Current capability remains weak: the last fixed full-match result is zero
goals/wins against Nexto despite higher contact counts and fewer concessions.
Neither the launch audit nor training health constitutes competitive/SSL
acceptance or a deployment change.
