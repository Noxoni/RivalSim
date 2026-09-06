# Prospective native kickoff reset-compatibility test V2

No training. Exact reference600 checkpoint and export, each on both teams
against the unchanged installed native Nexto. Two standard five-minute games,
900-second wall limit each including loading/countdowns, at most120seconds of
overtime. This is separate from the expired V1 partial game, not its continuation.

The only input change is first-kickoff-only reconstruction of eight unavailable
wheel fields as the simulator's empty reset cache. The car's on-ground state,
all other observation fields, model, hidden-state rules, actions, physics and
rewards are unchanged. It is not a measurement that native wheels lost contact.
Every decision records both base and final observations, quality, and the explicit
projection flag. Every delivered packet records its phase. No projection is
applied without an observed Countdown, on arbitrary attachment/rebind, or on
ordinary pause/held packets. Native V1 and the user's legacy installed bot remain
unchanged; the new runner/configs live only in this repo.

The implementation, 27 focused test results, source hashes and `native_authority`
are published before launch. Tests include real exported-actor inference,
first/reset-only behavior, nonpromotion of quality masks, unchanged other fields,
midplay attach/rebind, ordinary pauses, and deliberate invalid capture examples.

The existing game may be reused only if it is the exact prior test-owned process
and a bounded read-only observer detects no active game. A different existing
game is not interrupted. No policy process is running at preparation time.

New outputs: `G:/dev/RivalSim-runs/entity-native-reset-v2`.
Run/reduce using `G:/dev/RLBot-Rival/.venv/Scripts/python.exe -B` with
`benchmarks/run_entity_native_reset_v2.py run` and, after a case closes,
`benchmarks/report_entity_native_reset_v2.py reference600_blue` (or orange).

Do not call this native recovery, improvement, a win or SSL before actual results.
If a time bound expires, preserve the partial game. Do not extend bounds, revise
inputs/rewards or select another checkpoint after seeing these outcomes.
