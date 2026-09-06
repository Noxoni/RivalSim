# Prospective native kickoff cold-start/cache isolation

No learning, model selection, reward change or production physics change.
Use immutable Direct Skills +675 SHA256
`77E6EF076F7073549F45168FA6FB34BE9A2D4C25E1580F17F7434A59195B32F8`.
Keep agent675 STOP. The original runtime package remains unchanged. Publish
this authority and the diagnostic implementation before GPU execution.

Replay only the first 4,748 ticks of the same ten fixed matches. The previous
exact continuity archive identifies each world's first native post-goal reset
at ticks `[1756,856,644,628,668,4716,856,644,628,668]`. Capture the first32 ticks
of both initial and first-post-goal windows, all applied controls at120 Hz,
physical state, and complete mutable world state at origins. Compare the
common recorded decision rows exactly against the immutable previous trace.
No synthetic goal, controller script, new opponent or new evaluation metric.

Restore private snapshots in the diagnostic process. Compare each first
post-goal world with the initial world of the SAME actual layout and Rival side.
Use the captured post-goal 32-tick controls for every causal comparison arm;
separately verify whether actual initial and post-goal control tapes match.
Restore all captured per-world arrays before each arm. Verify unmodified
initial/post-goal fixed-action replays against captured motion exactly.
The scalar solver tick is0 for cold start and positive for warm start;
baseline replay must verify that warm absolute tick differences are irrelevant
over this window. Keep elapsed episode state separately, never reset a live run.

Fixed arms: actual initial cold; initial with warm solver clock; actual post-goal
warm; post-goal with only initial quaternion/solver quaternion; post-goal with
only initial handbrake cache; post-goal with only initial chassis contact count
and world-contact normal; post-goal with only initial wheel/contact-count caches;
post-goal with all listed caches; post-goal with all caches plus quaternion;
post-goal with all caches plus quaternion and cold solver clock.
Report all arms; no selective omission. Matching warm initial after cache
replacement may support cache causality. Matching cold initial only after
clock change supports initialization semantics, not a production reset defect.
If baseline replay fails, retain failure/raw evidence, do not interpret arms.

Use the existing exclusive GPU lease and owned match stream. No optimizer is
constructed. Verify model/checkpoint/source hashes before and after. Save raw
origin arrays, per-tick controls/states, intervention arms, deterministic CPU
reduction and focused tests to Git. The finite ten-world/32-tick arm workload
is a targeted physical-continuity isolation, not a new training campaign.
Any later production fix requires causal evidence and separate focused native
validation; do not blindly clear buffers, change time semantics or Nexto cadence.
