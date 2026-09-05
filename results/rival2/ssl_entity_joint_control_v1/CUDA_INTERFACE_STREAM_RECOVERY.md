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
