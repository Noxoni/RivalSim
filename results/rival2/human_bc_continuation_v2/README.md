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
