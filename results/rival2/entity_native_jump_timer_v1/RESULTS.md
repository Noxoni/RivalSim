# Native jump-timer semantic mismatch: measured, limited action sensitivity

No learning, game launch, runtime patch, policy mutation, reward change or
physics change. Both already completed V2 native games are the data source.
Exact reference600 and its unchanged CPU inference export are retained.

## Concrete source discrepancy

`deployment/entity_native_v1/packet_observation.py`, `Rival2LiveAdapter.advance`,
increments `air_time_since_jump` during any airborne state with `has_jumped`,
including the held initial jump. The actual simulator vehicle kernel resets
that timer to zero while `is_jumping`; elapsed time starts after that phase.
The normalized observation scale is 1.25 seconds in both domains.

The installed RLBot schema provides an independent remaining `dodge_timeout`
while an unspent jump/dodge window is active. It is negative in other cases,
including during the initial Jumping phase in these recordings. Its documented
window is 1.25 seconds after the initial held jump; held jump can last up to 0.2
seconds. Schema path and SHA are bound in `audit.json`.

The probe changes only:

- Initial Jumping: timer zero, matching simulator phase semantics.
- Airborne, has jumped, no double jump/dodge spent, positive remaining timeout
  at most 1.25 seconds: elapsed = 1.25 minus the native remaining timeout.
- The corresponding observation `dodge_available` is recomputed consistently.

Spent, expired, unobserved and unsupported timer cases are left unchanged. This
does not claim a complete native timer reconstruction or change any quality mask.
Falling off a surface is not treated as an elapsed initial jump.

## Full-recording results

| Check | Blue | Orange |
|---|---:|---:|
| Recorded decisions | 10,793 | 11,251 |
| Observations changed by bounded timer probe | 1,032 | 1,039 |
| Original recorded controller replay | exact | exact |
| Immediate action differences, same original hidden state | 3 | 3 |
| Action differences with sequential probe hidden state | 4 | 5 |
| Rival's initial Jumping observations with timer discrepancy | 76 | 78 |
| Rival's positive remaining-time observations with discrepancy | 137 | 132 |
| Rival's dodge-availability bits changed | 6 | 15 |
| Opponent dodge-availability bits changed | 8 | 0 |

The maximum elapsed-timer discrepancy is about **0.20000014 seconds**, consistent
with counting the held initial jump too early. For Rival's valid positive native
timer samples, the mean overestimate is 0.10529 seconds on Blue and 0.15581 seconds
on Orange. Complete opponent counts, errors and per-channel action differences
are in `audit.json`.

The mismatch is real, but this test finds low immediate sensitivity of the
selected actor to these supported corrections: six of 22,044 paired decisions
change. This does not prove the effect on a real game is small: even one changed
action can alter a closed-loop trajectory. It also does not explain the entire
native 0-24 / 1-30 loss, justify training more, or demonstrate a recovered policy.

Both counterfactual arms retain the recorded physical trajectory and original
previous-action fields. Sequential hidden-state replay is an input-sensitivity
probe, not a simulated alternative match and not a deployment-ready transform.

## Evidence and tests

`reference600_blue.npz` and `reference600_orange.npz` retain per-decision frame,
reset, native phase/flags/remaining timeout, supported-case mask, recorded/probed
timer and availability values, and original/paired/sequential controller outputs.
No raw capture is modified. Source checkpoint, export, source code and packet
hashes are bound and verified before/after the CPU-only run.

Reproduction (refuses to overwrite completed output):

`G:/dev/RLBot-Rival/.venv/Scripts/python.exe -B benchmarks/audit_entity_native_jump_timer.py`

Focused tests exercise held-jump zero, independent positive-timeout reconstruction,
unsupported/spent/expired cases, every stored probe's permitted scope, exact
source/data hashes and immutable parent identity. Final results are in `tests.xml`.

## Next action

Preserve this diagnosis alongside the confirmed Nexto kickoff/scheduling
differences. Do not apply ad-hoc persistent masks or claim a full transfer fix.
The largest unexplained gap still needs a bounded physical/opponent-fidelity
comparison before another training campaign; no native game is running now.
