# Zero-step operational stream recovery

The first launch completed all five fixed skill baselines but failed while
constructing the full-match simulator, before any new PPO optimizer step.
The original failure and package are retained alongside this report. The
zero-offset immutable checkpoint and its model/Adam equality audit are preserved.

Cause: `Rival2Env._activate_torch_stream` repeatedly creates external Warp
wrappers for the current Torch stream. The installed Warp implementation registers
each external handle on construction and unregisters it on wrapper destruction.
An older wrapper's destruction can remove the registration shared by the current
wrapper. Native graph-capture checks its stream registry and raises `unknown
stream`. The match evaluator formerly inherited that borrowed stream.

Correction: the direct-skills runner now owns a fresh Warp stream for the complete
graph-captured match evaluation. Torch uses the same handle. It synchronizes only
at evaluation boundaries and restores both caller streams even on exceptions.
Physics, rewards, PPO, opponents, evaluation states, actions, counters, and all
learning settings are unchanged. No dependency or global simulator change.

Real CUDA regression constructs/steps every skill family, destroys them, runs a
captured ten-world match for 120 ticks, and returns to skill environments; it
repeats this twice and checks stream restoration on failure. The regression passed.
This is a runtime check, not a new gameplay evaluation or optimizer step.
As a negative control, substituting `contextlib.nullcontext` for the new scope
reproduced the original `unknown stream` at the same graph-capture line. This
confirms the test covers the actual fault rather than merely succeeding after
a process restart (Warp 1.16.0, CUDA toolkit12.9/driver13.3, RTX5090).

The original full-scale no-step preflight remains evidence for the original
reward/training implementation; this targeted test covers the operational delta.
The amended package retains/binds the original package, failure, checks and
unchanged semantic authority. Remote publication is verified before resume.
Resume uses rolling offset0 with exact SHA
`DCD2ACC6203CF83228B28FCEBCADD487F93ED2B446466F29D883FF77DE523CD4`.
The completed skill baseline is not repeated; only the pending natural full-match
baseline remains before the first accepted update.
