# Nexto integration: a large controlled timing effect, no learning

Frozen protocol committed before execution at
`00a9d3c9040f360555be56d297576d44fd9beb38`; four remote files read back before
launch. Same reference600 checkpoint, ten worlds (five standard layouts, both
Rival sides), seed and 2,400 physics ticks per arm. No rewards, weights,
architecture, production opponent code or physics were changed.

| Opponent intervention | Rival goals | Nexto goals | Rival touches | No Rival touch worlds |
|---|---:|---:|---:|---:|
| Current simulator baseline | 14 | 3 | 34 | 0 |
| Native kickoff yaw only | 14 | 3 | 34 | 0 |
| Native neural compute/emission timing only | 2 | 10 | 23 | 1 |
| Both | 2 | 10 | 23 | 1 |

All ten worlds remain unresolved because this is a **20-second development
comparison**, not ten completed matches. Touch rates are 10.2 versus 6.9 per
world-minute. The baseline's 6,000 Rival decisions exactly reproduce the
previously committed reference600 simulator capture.

## Controlled finding

Changing neural compute/emission timing alone substantially worsens Rival's
result on this fixed test. This provides causal evidence that the simulator
opponent's immediate-application timing made this particular comparison easier.
It is not proof of a general skill ranking, the entire native transfer cause,
or recovery after changing the adapter.

Correcting yaw at kickoff table rows 44-59 changes emitted controls but leaves
all sampled car/ball positions **bit-identical** within each timing arm. Thus
the yaw discrepancy had no measured physical effect on these trajectories.
It is not legitimate to attribute the score reversal to the yaw fix.

The timing intervention implements the installed controller's pending-action
and emitted-control separation: after startup it computes at frames 2,10,18,...
and emits those newly computed controls at 8,16,24,... . Its clock is checked
against the previously executed installed native method's symbolic trace,
not just against a second manually copied expected schedule. Focused tests
also cover previous-action history and kickoff script override dispatch.

## Deliberate scope limits

This is **not a full native Nexto implementation**. All other simulator
observation construction, argmax selection, kickoff admission/script lifecycle,
countdown handling and physical semantics remain unchanged. Native stochastic
kickoff selection is not added here. The diagnostic subclass requires all
worlds active; do not silently reuse it for training assignments or overtime.

No promotion, model selection or learning follows from this short comparison.
A separately frozen two-arm full-regulation comparison will check persistence
over actual simulator matches. Native 0-24 and 1-30 results remain valid negative
evidence and are not replaced by any simulator score.

## Artifacts and reproduction

Each arm stores all applied controls, sampled car/ball positions, complete
existing match telemetry, wall time, immutable-model checks and integrity hashes.
`results.json` summarizes the four arms; `protocol.json` binds exact inputs/code.
The shared GPU lease and owned evaluation stream excluded concurrent training.

`G:/dev/RivalSim/.venv/Scripts/python.exe -B benchmarks/compare_nexto_native_integration.py run`

Completed outputs are never silently overwritten. The three focused timing
tests passed before protocol publication; final test evidence is retained.
Nexto-derived behavior retains the CC BY-NC-SA 4.0 attribution documented in
`third_party/nexto/PROVENANCE.json` and the native source-binding audit.
