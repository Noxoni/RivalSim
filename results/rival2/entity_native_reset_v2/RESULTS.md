# Native reset V2: first completed game, comparison still in progress

Exact reference600, unchanged policy weights/export, against installed native
Nexto. No training or optimizer steps. The separate protocol was published at
`9c9dffe8b59c6b2bdd909bf672fdc630ea7fd4f8` before either game. This report currently
covers **Blue only**; do not call the two-case comparison complete until Orange
has closed and been reduced.

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

## Current next action

The same runner automatically started the Orange case. Inspect its actual
process plus `G:/dev/RivalSim-runs/entity-native-reset-v2/campaign_state.json`;
do not relaunch, restart, extend deadlines or change the live runtime.
After it closes, run the same published reducer and add the second result.

A parallel read-only audit now rules out different Nexto model weights, but
finds simulator kickoff-table and action-scheduling discrepancies plus a weak
old table reference. See `results/rival2/native_nexto_integration_audit_v1/RESULTS.md`.
These are specific fidelity findings, not an explanation of the entire native
loss. No opponent or policy implementation was changed during this comparison.
