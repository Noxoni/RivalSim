# Deployment wheel semantics: measured policy sensitivity, not a live test

Read-only follow-up while the unchanged learner continues toward +500. No
simulator/GPU rollout, optimizer, checkpoint, observation rewrite or live runtime
change occurred in this diagnostic. Native contact-reset invalidation remains
the published correction; this note does not silently redefine that contract.

## Source evidence

The existing external `G:/dev/RLBot-Rival/bot/rival2_live/runtime.py` converts
`AirState.OnGround` into `on_ground`, then fills all four wheel fields with that
single aggregate value (`_state`, around lines 324-350). The observation builder
concatenates those fields. `rival2_unified_v5/unified_runtime.py` feeds that
adapter's observation into its exported model and declares the broadcast in
its summary. Hashes of both inspected files are recorded in
`deployment_wheel_semantics_000451.json`.

This is an older installed bridge. The new Direct Skills entity/joint90 policy
is NOT claimed installed, exported or tested through it. Aggregate ground
contact is not an exact measurement of each individual wheel, especially on
walls, landing or partial contacts. No claim is made here about a current SDK's
other possible telemetry interfaces.

## Bounded +451 input intervention

Use the ten initial native kickoff observations in the immutable +300 trace:
same five layouts and both sides. Both cars have on_ground=1, but initial wheel
caches are invalidated zeros. Load preserved +451 (SHA256
`B9D1BE5F75D22A9751832982A44B4A50F5BEE521DD4A1EE04C7640101EA8F324`)
with strict state loading on CPU and zero recurrent memory. Compare deterministic
joint90 outputs on the copied native observations versus copies with only the
eight wheel fields set to 1, matching the inspected bridge's aggregate rule
when both packet players are OnGround. All other 174 input values are unchanged.

Eight of ten first actions change: boost in eight, steer/yaw in two, roll in
six. Throttle, pitch, jump and handbrake do not change in these cases. All native
initial actions request boost; only two of the ten aggregate-contact variants
do so. Maximum logit displacement is 31.034584045410156. Complete controller
vectors and case identities are retained in the JSON. Model and checkpoint
remain byte-/tensor-identical; no optimizer exists in this calculation.

This isolates sensitivity to one known input-semantic difference. It is NOT a
complete reconstructed RLBot observation, a measured live controller trace, or
evidence that an entire kickoff/match would fail. Other live timer, cadence,
state-history and availability differences are outside this calculation.

## Consequence for the actual goal

Simulator kickoff scoring cannot certify live Rocket League kickoff transfer.
Before promoting this exact policy for live play, audit the complete deployment
observation/cadence path and test it. Do not copy the old V5 runtime blindly or
turn all wheel features off merely to preserve a simulator score. Any eventual
domain handling must identify aggregate/unavailable versus native individual
wheel state honestly and be validated on the resulting policy trajectory.

The active PPO run remains unchanged and is not interrupted for this legacy
bridge finding. +500 still uses the corrected same-method +451 baseline. The
SSL objective remains unmet; do not confuse this diagnostic with achieved
capability or with a reason to weaken reward/finite protection.
