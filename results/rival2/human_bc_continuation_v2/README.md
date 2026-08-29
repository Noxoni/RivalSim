# Rival human BC continuation V2

V2 corrects one V1 selection-semantics defect without changing training data,
sampling, objectives, optimizer source, validation metrics, test discipline, or
any retention/distribution safety threshold.

Among eligible guard-safe candidates, any strict reduction in the frozen combined
validation score updates the best checkpoint. The inherited 0.0025 material-score
delta controls only whether plateau patience resets. Final acceptance additionally
requires at least 1% held-out RMSE improvement in both gameplay and mechanic
families relative to the selected human-BC V1 parent.

The V1 accepted cumulative-step-192 candidate never accessed either test split.
V2 is rerun deterministically from the immutable step-160 parent, selects solely
from validation, then opens the continuation candidate test data once.

## Final outcome

The continuation selected cumulative step 192 after 32 additional accepted
optimizer steps. Validation complete-action RMSE improved from 0.560036 to
0.549960 for gameplay and from 0.535502 to 0.524622 for mechanics. The next
step-224 boundary was rejected by the frozen critic maximum-absolute-drift guard
at every prospectively frozen LR retry, with exact transactional rollback.

The selected candidate then opened the untouched test splits exactly once. Human
test RMSE improved in both families, and simulator-test actor mean KL was
0.005008. However, simulator-test critic maximum absolute drift was 0.518298,
above the frozen 0.5 guard. The final verdict is therefore `BLOCKED`; the saved
checkpoint is an inspection artifact and is not accepted for deployment.
