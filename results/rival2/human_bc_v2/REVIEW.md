# Human Behavior Cloning V2 — Final Review

Verdict: **BLOCKED** by the unchanged simulator actor-retention test guard.

## Selected validation checkpoint

- Parent: `checkpoints/rival2/human_bc_v1/rival2_human_bc_v1.pt`
- Parent SHA-256: `560C2414C17039DC920126EA148BF73FE6CC4677EE440F043599A7E1C76D2874`
- Selected V2 actor-only steps: 2,304
- Total accepted V2 steps executed before plateau: 2,752
- Stop: held-out validation improvement plateaued
- Gameplay validation complete-action RMSE: `0.5600362420 -> 0.5171583295`
- Mechanic validation complete-action RMSE: `0.5355024934 -> 0.4797895849`
- Mechanic labels improved: 12 / 13
- Mechanic labels within the prospectively frozen 2% nonregression tolerance: 13 / 13
- Validation simulator actor mean KL: `0.0038030504`
- Validation simulator maximum sample KL: `0.5221092105`

The only mechanic label that did not strictly improve was `walldash`, at a
`1.013047` RMSE ratio to the parent. It remained inside the frozen 2%
nonregression tolerance. No label crossed that tolerance.

## Structural freeze proof

The optimizer was a fresh AdamW instance containing only `actor.weight` and
`actor.bias`; the Human BC V1 optimizer, all PPO optimizers, and all historical
Adam state were excluded. The shared trunk and critic tensor hash remained:

`44DD21E20FAF9D07C58009D791564B7CD7A7D5FB85EF02A17E4FC99F58C250AA`

before training, at every accepted validation boundary, in the selected state,
and in the final checkpoint. Critic RMSE and maximum absolute drift were both
exactly zero on validation and test.

## Once-only test outcome

The test split was not loaded before validation checkpoint selection and was
evaluated once after selection. Mean actor KL remained small (`0.0037696635`),
all per-channel mean KL values remained below `0.001`, and outputs were finite.
However, the maximum per-sample actor KL was `2.4985938072`, exceeding the
unchanged hard limit of `2.0`. This single failed hard condition makes the
campaign blocked. The test result was not used to choose a different checkpoint,
alter training, weaken a guard, or resume optimization.

## Scope audit

No PPO update, reward change, mechanic definition/detector change, demonstration
addition, dataset/split change, observation-adapter change, raw-recording
mutation, or observation/action-contract change occurred. The parent and frozen
adapter hashes remain exact. The detailed curve, per-label table, corpus hashes,
once-only test metrics, checkpoint metadata, and integrity checks are retained
in the adjacent machine-readable artifacts.

## Recommendation

Do not promote this checkpoint as accepted Human BC V2. Investigate the held-out
maximum-KL tail without tuning this campaign against the now-opened test split.
Any further BC should begin under a new prospective authority with an independently
frozen retention validation design; the hard guard should not be weakened
retrospectively.
