# Selected kickoff handbrake reset is implemented and validated

Native change: one array argument in `rival2_interval_reset`, one selected-car
assignment `handbrake_value[car]=0.0`, and one call-site argument. No other
physics/kernel body is changed; an AST comparison against732c008 verifies this.
No first-tick/solver, quaternion, collision, controller, reward, model or PPO edit.

The pre-edit native reference was captured with kernel SHA
`83CA8E172A08588C3CDA2E58DCD57F17EA3F3B93540D184D51DF6CEB9CA17C14`.
The corrected kernel is
`BC91F57F6A593E7B5D0A4ED8A16B7EA02465954DDD437CFD4764F77869F2B62F`.
All4,368 arrays/traces match exactly after correction:

- standard five-layout/two-car reset versus the independent old-reset-plus-
  handbrake-clear reference;
- original curriculum and shooting-pressure curriculum versus their unchanged
  pre-edit native transitions;
- even, odd, all and no-world reset masks;
- all captured state/vehicle/contact/pad/lifecycle/control/template arrays,
  final182field observations and four following physics ticks;
- all unselected world arrays unchanged.

`native_reference.npz` SHA256
`662461445A8DB57ED71460897002CA7571D88BEA192E9D0C821BB113CC72E06A`.
`before.json` and `after.json` retain exact mask/count/kernel identities.
Focused17tests passed, including native goal reward/reset on every hold tick,
time-limit bootstrap, fixed Nexto learner share, prior wheel-reset behavior,
native shooting physics and comparison-method guards. One existing TorchScript
deprecation warning, no failures or skipped tests. No optimizer step.

The prior runtime package remains unchanged as historical authority. The new
`authority.json` explicitly supersedes only its two runtime source hashes for
new V3-method matches. Old bounded runners deliberately cannot silently launch
against old hashes. No PPO resume is enabled by this fix alone.

Next: original fixed ten-match evaluation of675 and600 under the corrected
method. Keep the675checkpoint and agent review STOP. Do not credit runtime
effects to learning or use old-method600 as the new675 comparison reference.
