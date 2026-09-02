# Rejected V3 run: duplicate orange sign

The first V3 process was stopped after training block 25 and is not an eligible
checkpoint lineage.  The physical outcome tracker multiplied the orange
agent's already-canonical observation velocity by the raw team sign a second
time.  Consequently, the same positive goalward momentum transfer was rewarded
for blue and treated as negative for orange.

The failure was visible in block-25 easy-finish telemetry: blue reported 12
goalward-velocity contacts (1.17%) and a transfer sum of -93,694.59 uu/s,
while orange reported 400 contacts (39.06%) and +95,319.01 uu/s.  One shared
policy and symmetric scenario generation cannot physically justify that split.

The process was interrupted immediately.  No checkpoint from this run was
promoted, selected, or copied into the repository.  The external run directory
was preserved as `ground-to-air-goal-v3-rejected-double-sign`, and the complete
curve is retained in `training_curve_rejected_double_sign.jsonl`.

The corrected tracker relies on Rival2TensorBridge's established convention:
positive observation-space Y is goalward for both perspectives.  A focused
two-side unit test now asserts identical causal outcome semantics.  Training
must restart from the byte-exact passing V2 parent under the corrected,
prospectively committed authority.
