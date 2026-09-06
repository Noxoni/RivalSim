# Native reset V2: completed two-side comparison

Exact reference600, unchanged policy weights/export, against installed native
Nexto. No training or optimizer steps. The separate protocol was published at
`9c9dffe8b59c6b2bdd909bf672fdc630ea7fd4f8` before either game. This report currently
covers both completed games. Orange has closed and passed the same published
reducer. Both reached actual native Ended packets within their unchanged caps.

## Blue completed result

- Full regulation game: **Rival0 – Nexto24**, actual native Ended packet.
- Wall duration544.64seconds including loading, goal sequences and countdowns;
  within the unchanged900second prospective cap.
- Observed distinct touch timestamps: Rival61, Nexto140. These are lower bounds
  from30Hz decision packets, not possession/control classifications.
- Mean Rival speed1195.97uu/s; p952023.96; below100uu/s1.59% of decisions.
- Grounded78.61%; car above300uu2.86%; supersonic2.49%.
- Forward throttle87.07%, reverse7.76%, boost37.57%, jump2.38%, handbrake1.20%.
- Mean car–ball distance2097.26uu; ball in opponent half15.56%.

Rival is active and makes contacts, but this is poor competitive gameplay, not
native recovery or SSL. The earlier V1 result was **partial0–37**, with23observed
Rival touches and about80regulation seconds left. These different, uncontrolled
game trajectories are not a matched experiment. The V2 outcome cannot establish
the magnitude of causal improvement; it establishes that first-kickoff input
compatibility alone did **not** make this checkpoint competitive.

## Execution and input integrity

All10,793recorded decisions replay exactly from the exact export with the
recorded recurrent resets. All64,159delivered packet records are checked for
correct held controls. There were25reset projections, each only on the first
decision after actual Countdown→Kickoff evidence. All other observation values
are bit-identical to the original packet reconstruction and the eight wheel
quality masks remain unavailable. This is explicit reconstruction of the
simulator reset cache, not measured native wheel contacts.

At10,768eligible uninterrupted decision endpoints, native `last_input` equals
the previously issued command in all eight channels. Twenty-five reset
boundaries are excluded. Missed/duplicate/out-of-order packet and skipped
decision counters are0. This verifies endpoints, not every intervening native
physics tick. Source/export SHA-256 identities remain unchanged; no nonfinite
runtime failure occurred.

Raw captures, ready/closed state, actual match result, hashes, replay/motion
audit, consumed-control audit and projection audit are in `reference600_blue/`.
The reduced records are committed byte-exactly rather than newline-normalized.

## Orange completed result

- Full regulation game: **Rival 1 - Nexto 30**, actual native Ended packet.
- Wall duration 610.45 seconds, within the unchanged 900-second cap.
- Observed distinct touch timestamps: Rival 57, Nexto 176 (30 Hz lower bounds).
- Mean speed 1150.31 uu/s; p95 1972.96; below 100 uu/s 1.61% of decisions.
- Grounded 77.74%; car above 300 uu 1.03%; supersonic 2.27%.
- Forward throttle 84.03%, reverse 9.40%, boost 40.31%, jump 2.43%.
- Mean car-ball distance 2192.59 uu; ball in opponent half 21.31%.

All 11,251 Orange decisions replay exactly. All 71,853 delivered packet records
pass held-control checks. All 32 projections occur only at their intended first
kickoff decisions, with unchanged other inputs and unavailable wheel masks.
All 11,219 eligible consumed-control endpoints match all eight issued channels;
32 reset boundaries are excluded. All missed/duplicate/out-of-order/skipped
decision counters are zero. Source/export hashes remain unchanged.

Rival is active but is not competitive with native Nexto on either side.
This is not native recovery, finishing mastery or SSL. The earlier partial V1
game is not a completed matched baseline; causal improvement is not established.

## Completed integrity checks and next action

The runner has exited with external status `complete_review`. Final external
state and launcher logs are preserved in `completion/`. Both raw cases and
their audits are complete; do not relaunch or reduce them again. No learning
or production runtime changes occurred during the comparison.

A parallel read-only audit now rules out different Nexto model weights, but
finds simulator kickoff-table and action-scheduling discrepancies plus a weak
old table reference. See `results/rival2/native_nexto_integration_audit_v1/RESULTS.md`.
These are specific fidelity findings, not an explanation of the entire native
loss. No opponent or policy implementation was changed during this comparison.

The full Blue-game observation-format comparison in
`results/rival2/entity_native_observation_math_v1/RESULTS.md` found a maximum
normalized difference of 2.384185791015625e-7 across 10,793 decisions, with zero
sequential action changes. It uses the actual production simulator observation
methods. Its 97 shared reconstructed fields are explicitly not independent
measurements of native timers, wheels, pads or lifecycle. This isolates formatting
but does not establish complete state or physics equivalence.

Continue focused transfer diagnosis, not PPO. The jump-timer discrepancy is now
measured in `results/rival2/entity_native_jump_timer_v1/RESULTS.md`: about 0.2
seconds maximum error, but only six paired immediate action changes over both
games under the supported-case probe. No closed-loop causal result is claimed.
Shooting already occupies 20% of the direct-skills scenario bank; the completed
finishing-goal block uses actual goals/concedes/timeouts. Another shooting
scenario is not the missing implementation here.
