# Full native-game observation formatting check

No learning, policy/reward/physics change or game connection. Uses the completed
native reset V2 Blue game, exact reference600 and its unchanged inference export.

## Result

For all **10,793 recorded decisions**, the actual production
`Rival2TensorBridge.observation`, `_car_block`, `_basis` and `_timer` methods run
on CPU tensor views constructed from the recording. No second hand-written
observation formula is the comparison reference.

- Maximum normalized absolute difference: **2.384185791015625e−7**.
- Samples with any field difference above1e−5: **0**.
- Original recorded actions reproduce exactly in the simulator Python runtime.
- Full recurrent replay with the rebuilt inputs changes **0 actions**.
- Both sequential arms retain the recorded25kickoff reset boundaries.
- Source checkpoint/export unchanged; optimizer steps0.

This rules out a material mathematical formatting/field-order difference in
the tested reconstruction. It does **not** establish native state observability
or native/RivalSim physics equivalence, and does not explain away the0–24loss.

## Independent versus shared inputs

Packet position, linear/angular velocity, rotation, boost, jump/dodge/ground/
supersonic/demo flags, available native timers, and previous consumed controls
are decoded from the raw GamePackets. Rotations are converted to physical xyzw
quaternions using the previously tested native convention; the production
Torch builder independently produces forward/up vectors. There are85output
fields not marked as shared reconstruction.

**97 fields use the existing reconstructed inputs in both arms**: wheel-cache
projections, unavailable/reconstructed car timers, pad active/cooldown values
and lifecycle indicators. These are deliberately shared to isolate formatting.
They are **not independently validated native measurements**. Normalized clipped
timers remain clipped; reconstructing a value from them cannot recover its
unclipped native history. No availability/quality claim is promoted.

Pad cooldown semantics were also checked against the installed RLBot schema:
`BoostPadState.timer` is elapsed time since pickup, not remaining cooldown, so
the native builder's duration-minus-timer direction is consistent with that
API. This source-level fact does not replace an independent packet/pad-order
audit.

## Evidence and reproduction

`inputs.npz` preserves the raw-decoded and shared-reconstructed tensor views,
recorded observations/actions, frame identities and reset counters.
`comparison.npz` stores every rebuilt observation and both sequential action
streams. `audit.json` reports all182fields, shared-input flags and source/data
hashes. `export.json` binds raw capture and exact policy identities.

Export using the native Python interpreter:
`G:/dev/RLBot-Rival/.venv/Scripts/python.exe -B benchmarks/audit_entity_native_observation_math.py export`.

Compare using simulator Python:
`G:/dev/RivalSim/.venv/Scripts/python.exe -B benchmarks/audit_entity_native_observation_math.py compare`.
Both refuse overwriting their completed outputs. Computation uses CPU tensors
only; importing the production module initializes Warp and enumerates devices,
but no Warp world, physics kernel or CUDA tensor is constructed by this probe.

Focused tests verify hash bindings, explicit shared-field scope, deterministic
production-builder output and an independent raw ball-velocity perturbation.
The perturbation changes only the expected ball/relative-velocity fields; it
never changes a recording or policy. An initial export import referenced a
helper through the V2 reducer that does not re-export it; the import was corrected
before any output artifact was written. Production remained unchanged.
