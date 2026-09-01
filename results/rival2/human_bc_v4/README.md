# Human BC V4 prospective authority

This directory binds the complete Human BC V4 simulator-retention authority
before any optimizer step. The established training/validation corpus now
includes rollout-aligned opponent-family and train-mask hashes. The separate
stress-validation and untouched-test corpora use distinct deterministic seed
namespaces and reserve exactly 8,192 whole worlds each.

Orientation-sensitive membership is derived only from the frozen Human BC V1
teacher and authoritative simulator training state. Dynamic hard-tail mining
may inspect only the frozen training candidate pool. Neither human nor simulator
test data may affect training, selection, stopping, thresholds, or acceptance
until the single final post-selection evaluation.

The authority build performs no optimizer step, PPO update, reward change,
human-data mutation, or test-candidate evaluation.
