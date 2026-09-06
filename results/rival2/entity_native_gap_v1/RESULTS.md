# Confirmed kickoff input-domain sensitivity, not recovered native competence

2026-09-06. This follows the preserved partial native reference600 Blue loss
(0-37) at `df2d184d0247fe9811d14a36e43d76a3b9fd0001`. No training or production
policy, reward, physics, opponent, observation-builder or runtime change occurred.

## Finding

RivalSim presents `on_ground=1` but zero individual wheel-contact fields at the
first policy observation after a kickoff reset. Its contact cache has just been
cleared; the next physical tick calculates wheel contacts. This is explicit in
`rivalsim/kernels/rival2.py`, `rival2_interval_reset`, and in initial construction
through `VehicleState`. The native packet runtime cannot observe individual
wheels and uses a qualified `AirState.OnGround` broadcast, yielding four ones
for each grounded car at that first observation. The masks already classify
these eight fields unavailable, not exact.

The recorded 38 native kickoffs were matched to the five standard simulator
layouts using both cars' canonical positions. The largest normalized combined
position distance was 0.000025943; this is very close geometry, not an exact
native solver-state match. Eight wheel fields differed by 1. Native settled
pitch and initial small velocities also differed, but were much smaller in
normalized magnitude. The complete 182-field error table is in
`kickoff_input_audit.json`.

With the exact unchanged reference600 actor and zero initial recurrent state:

- Actual native observations request immediate boost in **8/38** kickoffs.
- Changing only the eight wheel inputs to the simulator's reset zeros changes
  the first action in **30/38**, and requests boost in **38/38**.
- In the first diagonal example, actual native input gives straight throttle
  without boost. The zero-wheel ablation gives throttle, left steer/yaw and
  boost. Applying the opposite all-one ablation to the simulator observation
  changes its first command to the native no-boost command.

These are offline input ablations, **not installed masks**, measured native wheel
states, or evidence that the altered controller would win a real game.

## Controlled closed-loop consequence inside RivalSim

Two fixed 20-second arms used reference600, all five standard layouts on both
teams (ten worlds), unchanged physics/Nexto/recurrent-reset rules and raw argmax.
The single changed factor was the first Rival observation at each kickoff: the
probe set the eight wheel fields to 1. Every other field and all later decisions
were unmodified. No policy weights or simulator contact state were changed.
The unmodified arm reproduced all 6,000 controller decisions from the previously
committed simulator sequence capture exactly.

| 20-second arm, totals across ten worlds | Goals for | Goals against | Touches | Worlds with no Rival touch |
| --- | ---: | ---: | ---: | ---: |
| Original simulator reset input | 14 | 3 | 34 | 0 |
| Native-style first wheel input only | 4 | 14 | 8 | 8 |

These are **partial simulated games**, not ten match wins/losses. The experiment
demonstrates that the reset-input mismatch alone substantially changes this
policy's subsequent gameplay. It does not establish that this accounts for all
of the native 0-37 result, or that native compatibility is repaired.

## Short native ground-motion comparison

Using physical states and issued controls from all 38 native kickoffs, a
separate open-loop probe advanced the unchanged simulator for 40 physics ticks
(one-third second). This interval is before car-ball/car-car encounters. Only
Rival's motion is compared; opponent endpoint inputs do not reconstruct Nexto's
complete native control sequence.

| Fixed assumed command-onset delay | Mean final position error | Max position error | Mean final velocity error |
| --- | ---: | ---: | ---: |
| 0 ticks | 3.060 uu | 3.214 uu | 5.787 uu/s |
| 1 tick | 7.231 uu | 8.900 uu | 16.535 uu/s |
| 2 ticks | 11.344 uu | 14.851 uu | 27.570 uu/s |

All three hypotheses were declared in the script before execution; none is
installed as a timing correction. Native pose, velocities, boost, aggregate
ground state and prior input were copied. Unavailable solver caches and wheel
history use normal simulator initialization and are **not claimed exact**.
This does not validate aerial or collision physics, but gives no indication of
a gross early ground-acceleration mismatch under the same inputs.

## What remains unresolved

The reset-input mismatch is a demonstrated contributor, not an exclusive root
cause. Midplay wheel proxies, other lifecycle/timer reconstruction, longer
physics/contact fidelity and the installed native versus simulator Nexto
integration remain unproven. No claim of Nexto equivalence was added: the
native compiled executable identity is known, but its internal model and exact
scheduling have not been compared to the pinned simulator policy here.

The next bounded step is described in `NEXT_RESET_ALIGNMENT.md`. Do not resume
PPO merely because this diagnosis found a plausible explanation. Do not silently
present unobservable wheel values as physical native measurements. Preserve both
reference600 and finishing650 unchanged.

## Reproduction and evidence

`diagnose_entity_native_kickoff_gap.py` (native RLBot Python) exports the recorded
physical NPZ and produces first-input probes and full field differences.
`diagnose_entity_native_ground_motion.py` (RivalSim Python) produces the three
short motion comparisons. `diagnose_entity_kickoff_wheel_rollout.py` (RivalSim
Python) produces the one-factor closed-loop arms and full telemetry. Both GPU
probes use the existing exclusive lease and owned stream. All diagnostic output
paths refuse overwrite. Hashes bind native source packets, source checkpoint,
export, script and derived arrays in the JSON artifacts.

`tests/test_entity_native_gap.py` covers nonmutating field-isolated ablation and
the native-rotator to simulator-quaternion mapping, including 100 independent
orientations. `focused_tests.xml` also includes the seven existing packet
reducer/control-delivery tests: **10 tests passed**. No optimizer was constructed
or stepped by these diagnostic scripts.
