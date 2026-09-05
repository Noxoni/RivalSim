# Post-pilot CUDA interface diagnostic: stream setup

The pilot completed exactly +100 and its worker exited before this diagnostic.
The first dynamic CUDA attempt failed at `world.capture_graph`, before any
physics tick, with `Warp error: unknown stream`. Its exact traceback is retained
in `full_match_cuda_first_stream_failure.json` (also the initial contents of
`full_match_cuda_interface_failure.json`). This is not a training failure or a
gameplay result; the +100 checkpoint was independently verified unchanged.

The CPU-only diagnostic had not needed a CUDA stream. In CUDA mode,
`FreshGroundEnv` adopts Torch's current stream. The helper inherited Torch's
default-stream handle and then attempted Warp graph capture on that handle.
The production full-match runner uses a Warp-owned stream instead.

The operational correction creates an owned, nondefault Warp stream, exposes
it to Torch, and sets both libraries to that stream before moving the model or
constructing either world. The compared policy, observations, 48-tick fixture,
forced goal, physics, checkpoint, and full-match evaluation protocol do not
change. The idle-only test and lint pass. Dynamic CUDA verification must still
be executed after publication; do not count this document as a passing result.

The second CUDA attempt successfully executed all 48 ticks. All observations,
logits, actions, hidden states and reset masks matched exactly, but the deliberate
goal fixture produced zero goals. That failed result is preserved in
`full_match_cuda_pre_fixture_check.json`. Inspection showed the helper only
wrote exported ball telemetry; native integration uses `ball_world.position_bt`
and `velocity_bt`. The existing production forced-goal test already writes the
authoritative position. The helper is corrected to set both representations
for its same near-goal position and velocity. A focused CPU test confirms exact
unit conversion and that only the fixture world is changed. No production
physics or policy code is altered. Dynamic verification is still required.

Final execution after commit `73c76c063a6222cf832caa8fb5dfd03a989af2ab`
passes on CUDA: all 48 physics ticks have exact observation, actor-logit,
emitted-action, hidden-state and reset-mask parity; exactly one forced native
goal/reset occurs; checkpoint/model hashes are unchanged. The complete passing
result is `full_match_cuda_interface_check.json`. The separate failure files
above remain historical diagnostic evidence, not the current status. No forced
fixture result is counted as a gameplay goal. Full-match gameplay evaluation
is now running under the unchanged prospective protocol.
