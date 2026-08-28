# Rival 2.0 Targeted Mechanics Calibration Correction V1

## Result

The targeted Musty, Breezi, and Redirect correction is `PASS_GREEN`.

- Accepted calibration baseline: `f49768368377dcb5aa0cc67f3a08f79bd68538a3`
- Exact implementation source used by the final evidence run:
  `0124cd7f29278702158c9cbba9c741c11a29f111`
- Targeted calibration seed: `2026082707`
- Prospectively frozen held-out scenario variant: `509`
- Gameplay V1 +239 checkpoint:
  `77BF257131FB71DDEAEAE49D668C5E25AB1D06EE26149AB0D0AE303573CA5F21`
- Arena geometry:
  `2239556BDC74D205CAA6E46A0F6E91FA2C6E4257E84D4F608BA775958B0A5538`

No training ran. Mechanics reward contribution was exactly `0.0`. No reward,
policy, PPO, observation, action, simulator-physics, 30/120 Hz cadence, or
episode-lifecycle contract changed.

## Scope preservation

This was not a nine-family recalibration. The accepted threshold records and
all 72 accepted case rows for `speedflip`, `half_flip`, `pinch`, and `pogo`
are exact structured-data matches to the baseline. The same is true for
`possession` and `ground_carry`; those two families remain unchanged and
`NOT_READY_FOR_REWARD`.

The machine-readable summary records a canonical SHA-256 independently for
each protected threshold record and each protected case-row set, with both
exact-match flags true for all six frozen families.

## Corrected detector semantics

### Musty — `CALIBRATED`

A Musty now requires one legitimate contact onset during an actual backward
dodge (`flip_rel_torque.y < -0.25`), then measures the signed rotating contact
surface against the geometric car-to-ball direction. The GPU observer retains
nine ticks of car quaternion/position and ball position, reconstructs the
contact-point sweep from the authoritative contact point, and requires both
signed closure and non-trivial sweep-path length. Absolute normal speed is no
longer treated as a scoop, and the old local-X torque-axis test is not used by
the corrected detector.

Final physical conjunction:

| Feature | Direction | Threshold | Separation margin |
|---|---:|---:|---:|
| actual backward dodge | min | 0.5 | 1.0 |
| contact age (120 Hz ticks) | min | 4.5 | 7.0 |
| signed rotational closing speed (uu/s) | min | 154.479401 | 11.651337 |
| positive rotational share of closure | min | 0.264016 | 0.017990 |
| rotational sweep closure (uu) | min | 13.487574 | 0.075039 |
| rotational sweep path length (uu) | min | 23.017265 | 0.430721 |
| actual ball delta-v (uu/s) | min | 354.311600 | 103.652008 |

The 24 hard near misses include four each of: developed-rotation,
translation-dominated rear bonk; rotated front clear; lateral loose-ball
backflip contact; pre-scoop contact; lateral/descending roof slap; and
high-delta-v head-on backflip hit. Important negative contacts therefore carry
meaningful rotation and real impulse instead of reducing the problem to
"rotation versus zero rotation."

### Breezi — `CALIBRATED`

A Breezi is one continuous ball-present trace. Control-distance state begins
and remains resident on the GPU, the tornado-like roll/yaw path and ordered
nose-up -> inverted -> nose-down stages accumulate only during that continuous
relation, and the completed setup is frozen at the terminal backward-dodge
onset. The event can complete only on the corrected Musty terminal scoop.
Losing control clears the setup; terminal-only reacquisition cannot reconstruct
it.

Final physical conjunction:

| Feature | Direction | Threshold | Separation margin |
|---|---:|---:|---:|
| corrected terminal Musty | min | 0.5 | 1.0 |
| ordered orientation topology | min | 0.5 | 1.0 |
| nose-up peak | min | 0.265685 | 0.531370 |
| inverted depth | min | 0.770279 | 0.048977 |
| nose-down depth | min | 0.066282 | 0.132564 |
| integrated roll path (rad) | min | 3.815429 | 0.486946 |
| integrated yaw path (rad) | min | 1.277929 | 0.291349 |
| simultaneous roll/yaw ticks | min | 130.0 | 2.0 |
| simultaneous roll/yaw fraction | min | 0.981506 | 0.021836 |
| continuous control distance (uu) | max | 394.807556 | 342.654083 |

The control-distance envelope is derived specifically against the control-loss
and terminal-only-reacquisition negatives. Ordinary Musties are intentionally
allowed inside that envelope and are rejected by the missing Breezi path, so
the distance limit is not fitted into a proxy for the entire mechanic.

The 24 hard near misses include four each of: ordinary Musty without Breezi
setup; roll-only setup; yaw-only setup; wrong orientation ordering ending in a
Musty-class contact; control lost during an otherwise correct setup; and ball
reacquired only at the terminal interaction.

### Redirect — `CALIBRATED`

A Redirect requires legitimate contact with an already moving ball, material
incoming-to-outgoing direction change, retained outgoing speed, and transverse
contact context. The corrected GPU rule compares the pre-contact car approach
and authoritative contact normal with the incoming ball direction. It does not
use goal accuracy, reward, or a stricter angle alone. One deliberately weak but
genuine transverse redirect is part of derivation so the speed/retention edge
does not become a quality grade.

Final physical conjunction:

| Feature | Direction | Threshold | Separation margin |
|---|---:|---:|---:|
| legitimate contact | min | 0.5 | 1.0 |
| incoming speed (uu/s) | min | 514.813248 | 125.103729 |
| outgoing speed (uu/s) | min | 249.117188 | 12.197571 |
| direction change (rad) | min | 0.939067 | 1.200024 |
| transverse approach fraction | min | 0.372830 | 0.113740 |
| transverse contact-normal fraction | min | 0.186995 | 0.268654 |
| outgoing/incoming speed retention | min | 0.339024 | 0.051282 |

The 24 hard near misses include three real-contact examples each of: high-speed
head-on clear; high-speed trajectory continuation; small incidental
deflection; dead catch; normal aerial head-on touch; dribble/bounce touch;
strong hit on a slow ball; and high-angle non-transverse clear.

## Prospective calibration and held-out results

Each corrected family contains 24 intended positives, 24 hard near misses,
and 24 ordinary controls. The first 16 of each class are derivation; the eight
held-out cases per class use the prospectively frozen variant `509`. Thresholds
were frozen before that held-out evaluation and were not retuned afterward.

| Family | Split | TP | TN | FP | FN | Intended-positive physics failures |
|---|---|---:|---:|---:|---:|---:|
| Musty | derivation | 16 | 32 | 0 | 0 | 0 |
| Musty | held-out | 8 | 16 | 0 | 0 | 0 |
| Breezi | derivation | 16 | 32 | 0 | 0 | 0 |
| Breezi | held-out | 8 | 16 | 0 | 0 | 0 |
| Redirect | derivation | 16 | 32 | 0 | 0 | 0 |
| Redirect | held-out | 8 | 16 | 0 | 0 | 0 |

## Focused zero-reward shadow

The final shadow ran 256 short-lifecycle episodes: 128 against deterministic
Nexto and 128 against deterministic Wisp, with stochastic Rival and balanced
Rival Blue/Orange assignment. It covered 68.498333 simulated minutes.

| Family | Events | Events/min | Rearms | Duplicate suppressions |
|---|---:|---:|---:|---:|
| Musty | 0 | 0.000000 | 0 | 0 |
| Breezi | 0 | 0.000000 | 0 | 0 |
| Redirect | 84 | 1.226307 | 84 | 1 |

The single Redirect duplicate was suppressed while contact state was still
locked; all 84 accepted Redirects later re-armed through the existing physical
release rule. Impossible-state classifications were `0`. Both unchanged
`NOT_READY_FOR_REWARD` families emitted `0`. Mechanics reward contribution was
exactly `0.0`.

## Previously retained real-game events

The same deterministic shadow seeds/world assignments were replayed.

| Family | Baseline total | Corrected total | Baseline bounded samples | Exact sampled identities retained |
|---|---:|---:|---:|---:|
| Musty | 15 | 0 | 14 | 0 |
| Breezi | 0 | 0 | 0 | 0 |
| Redirect | 83 | 84 | 16 | 6 |

This is an important negative result for Musty: none of the 14 previously
retained bounded events survives the corrected signed backward-dodge/sweep
topology. The old events depended on the permissive absolute-normal/local-X
test and are not supported as real Musties by the corrected replay. No new
Breezi appeared. Redirect frequency is essentially unchanged in aggregate,
but only six bounded event identities are exact matches because the new rule
rejects ordinary high-angle hits and accepts transverse, speed-retaining
contacts according to the corrected contact context. Every old bounded event,
its old four-feature payload, retention decision, and any matching corrected
payload is preserved in `targeted_legacy_event_replay.json`.

## Validation and artifacts

- Targeted CUDA test suite: `10 passed`.
- Real-threshold GPU smoke: exact-zero reward, zero impossible states, no host
  copy in the per-tick hot path.
- Final shadow gate: `PASS_GREEN`.
- `targeted_correction_summary.json` SHA-256:
  `1936863986F4ACF310DA4F3F53EDC88F490106E5E5CD46B2220EF8FF001B29D6`
- `targeted_legacy_event_replay.json` SHA-256:
  `5CDB0E33F9A080C4BE0654851E51550C25EFF8BEAC424B32FDBC798065BC72B3`

The complete scenario parameters, trace features, extrema, thresholds, and
per-case classifications are in `results/rival2/mechanics_calibration_v1`.
