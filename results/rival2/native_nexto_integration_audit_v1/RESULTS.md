# Native Nexto identity and integration audit — no learning

This read-only audit ran alongside the unchanged bounded native reset V2 test.
It neither connects to the game nor changes its controls, simulator adapter,
checkpoint, physics, reward or optimizer. It inspects the installed executable
as a PyInstaller archive on disk, **not process memory**. No result here makes
Rival native-competitive or explains its entire simulator-to-native gap.

## What is established

The actual installed `nexto.exe` (SHA-256
`6371A5B9DD740AEA858D86532F928E0FDB13BF387D8379F76CCD679C9B33E845`)
embeds a 1,852,625-byte `nexto-model.pt`. Extracted file contents are byte-identical
to the pinned simulator model:
`BF5343B5EEACAC6BF7CDB75DAC4A5C14BA0F94D820EAE75F00A211B6119D69FA`.
Therefore a different set of Nexto model weights is not the explanation.

The installed `get_output`, `maybe_do_kickoff` and `update_controls` bytecode,
constants, names, local variables, arguments, flags and exception tables match
the [public v5 port at commit 0bdb6b49](https://github.com/VirxEC/NectoFamily/blob/0bdb6b49072f6f3829319e68bd6210a0ca4b24a2/nexto/bot.py).
This is a source binding for those methods, not an assumption based on the name
of the bot. Archive contents were read using the installed official
[PyInstaller archive reader](https://github.com/pyinstaller/pyinstaller/blob/develop/PyInstaller/archive/readers.py).
Extracted bytecode disassembly and exact file/source hashes are retained.

## Concrete fidelity discrepancies

1. **Kickoff yaw:** the actual vendored upstream `KICKOFF_CONTROLS` literal has
   neutral yaw during zero-based ticks44–59, with steer−1. The simulator's table
   supplies yaw−1 as well. All16 mismatches are recorded in `audit.json` and
   `kickoff_tables.npz`; every other table entry agrees. Neutral yaw is confirmed
   against the native `ControllerState` default. It is not legitimate to call
   the entire implemented table source-exact.
2. **The old table reference duplicated the error.**
   `benchmarks/run_nexto_fidelity.py:_source_kickoff` manually duplicates the
   simulator table, including the incorrect yaw. Its equality test therefore
   does not establish upstream fidelity. The old unit test only checked the
   first44 rows. The new audit interprets the actual checked-in upstream literal
   through a restricted AST parser instead of copying its expected values.
3. **Observation-to-action scheduling differs.** Executing the installed native
   controller method with symbolic numbered neural outputs, continuous active
   packets and no kickoff override shows neural computation at frames
   1,2,10,18,26. After startup, the action computed at2 is emitted at8, the one
   computed at10 at16, and so on: six ticks between these computation and
   emission points. The simulator adapter computes and immediately emits at
   1,9,17,25. Both have eight-tick steady inference intervals, but do not have
   the same scheduling semantics. The complete32-frame trace is preserved.
   This is not a real-policy output or physical delivery-delay measurement.

The public/native port also specifies stochastic neural selection during
Kickoff; the current simulator adapter always uses argmax. Hardcoded kickoff
controls can override these neural outputs. Its practical effect after the
script ends is not measured here. Do not silently change the evaluation
opponent or its random semantics mid-comparison.

## An apparent difference that is not a current native mismatch

The historical vendored observation source uses a different up-vector sign
than the simulator helper. However, inspecting **the actual installed v5
helper** rather than extrapolating from that old source disproves a material
current mismatch in this component. On18,342 car quaternions from the preserved
native V1 capture, maximum forward-axis difference is1.496e−7 and maximum
up-axis difference1.438e−7. None exceeds1e−5. These tiny differences reflect
normalization/rounding; no gameplay conclusion follows from this component
check. The exact source NPZ and helper bytecode hashes are in
`source_binding.json`.

## What to do with this evidence

Finish the already-running native reset V2 test unchanged. Do not revise its
authority, widen its bounds, swap opponents or resume PPO. Preserve its
complete/partial results first.

Future simulator-opponent fidelity work should compare against executable
upstream methods and actual sequential observations, not a second hand-written
copy of the adapter. The discovered table and scheduler differences justify a
separate bounded, no-learning comparison; they do not prove that fixing them
will repair Rival's native performance. Full Nexto observation/action parity,
remaining Rival input reconstruction and contact/airborne physics fidelity are
still open. In particular, a yaw difference during grounded steering is not
automatically a large physical difference, and earlier action application is
not automatically a weaker opponent.

## Validation and reproduction

Use `G:/dev/RLBot-Rival/.venv/Scripts/python.exe -B`:

- `benchmarks/audit_native_nexto_integration.py`
- `benchmarks/audit_native_nexto_source_binding.py`
- `-m pytest -q tests/test_native_nexto_integration_audit.py --basetemp .tmp/<fresh-directory>`

Audit outputs refuse overwrite. Source binding fetches only the specified
immutable public commit; it does not run downloaded module code. Bytecode from
the installed controller is invoked only against a synthetic scheduler stub;
the pure installed rotation function reads recorded arrays. No bot instance or
game connection is created.

An initial fixture put a class after an unconditional raise, so Python removed
the unreachable class before the lookup test. That failed test report is
preserved; moving the class before the raise tests nonexecution correctly.
The supplemental public-source compile initially inherited this diagnostic's
future-annotations flag; `dont_inherit=True` makes flags match the source's own
compilation semantics. Neither diagnostic fixture issue changed production.

Nexto source/model attribution remains Rolv-Arild/Necto contributors, with the
v5 port by VirxEC/NectoFamily. Nexto-derived disassembly/tables retain the
CC BY-NC-SA 4.0 terms documented in `third_party/nexto/LICENSE` and provenance.
