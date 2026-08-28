# Rival 120 Hz missing-feature invariance distillation V1

## Authority and scope

This lane starts from accepted parent `9fbef28ba08e9df6598e2d14bf339e306a055934` and the
immutable iteration-479-derived bootstrap. It trains only a new student to preserve the frozen
teacher actor under the committed human-demo observation degradation. It performs no human-action
behavior cloning, PPO update, reward change, mechanic-detector change, recording mutation, or split
change.

The complete pre-step contract is
`results/rival2/missing_feature_distillation_v1/frozen_config.json`, SHA-256
`5F20CE9FDE854A99405D53864FB1FB72F9B28FA4EC882F8D4C675DF627A16955`. That file is committed
and pushed before the first supervised optimizer step. Training code rejects any other byte hash.

## Frozen corpus and split

The authoritative corpus geometry is exactly 32,768 RivalSim worlds by 128 one-tick decisions at
120 Hz, with both car perspectives retained: 8,388,608 complete observations. The bootstrap and
mixed-opponent state generate the fixed rollout once. The observation tensor, RNG provenance,
environment configuration, opponent-family composition, and model/optimizer before/after hashes
are recorded. The large regenerated tensor is not committed; its full canonical content hash and
all inputs needed to regenerate it are committed.

World indices are permuted once with NumPy PCG64 seed `2026082822`, then assigned as 26,214 train,
3,277 validation, and 3,277 test worlds. All 256 observations from one world trajectory remain in
one split. The exact world-index lists and their hashes are written to `corpus_manifest.json`.

## Profiles and objective

The gameplay view uses the committed 16 direct / 58 derived / 34 approximate / 74 unavailable
mask. The Freeplay view uses 8 / 36 / 22 / 116 and neutralizes opponent and relative-opponent
context instead of fabricating a car. Both retain quality classifications and the same zero-plus-
unavailable-mask semantics as `RIVAL2_HUMAN_DEMO_BC_OBSERVATION_BRIDGE_V1`.

Each supervised batch uses the immutable teacher on complete observations and an independently
initialized, architecture-identical student. The loss weights frozen before training are:

- gameplay-degraded hybrid actor KL: `0.5`;
- Freeplay-degraded hybrid actor KL: `0.5`;
- full-observation actor-retention KL: `2.0`; and
- full-observation teacher-value MSE: `1.0`.

The hybrid KL is the committed analytic five-Normal plus three-Bernoulli implementation. The
student uses a fresh Adam optimizer at `1e-4`; it never reads or mutates the historical PPO
optimizer. Any resulting checkpoint marks that PPO state stale for the modified student.

## Frozen safety and stopping

Every 64 proposed supervised steps are transactional. The interval is rejected and rolled back if
the fixed 512-observation full-input retention corpus exceeds any of: actor mean KL `0.02`, maximum
sample KL `2.0`, maximum action-channel mean KL `0.01`, value RMSE `0.075`, value maximum absolute
drift `0.5`, or any nonfinite state. A rejected interval halves LR down to `1.25e-5`; guards are not
weakened.

Validation uses whole held-out worlds. A relative combined degraded-KL improvement below `0.5%` is
not material. Three such validations halve LR; six stop training. The frozen cap is eight epochs
or 1,600 accepted steps. Only guard-accepted material validation checkpoints replace the resumable
best checkpoint.

Final acceptance requires at least 70% gameplay-degraded test-KL reduction and gameplay KL at most
`2.9`, at least 40% Freeplay-degraded reduction, full-input actor KL at most `0.01`, full-input value
RMSE at most `0.05`, finite outputs, and the predeclared saturation/collapse bounds. Freeplay is not
required to approach zero because opponent information is genuinely absent.

## Commands

Pre-step validation:

```powershell
.\.venv\Scripts\ruff.exe check rivalsim\human_demo\bc_observation_bridge.py `
  rivalsim\human_demo\missing_feature_distillation.py `
  benchmarks\run_rival2_missing_feature_distillation.py `
  tests\test_human_demo_bc_observation_bridge.py `
  tests\test_missing_feature_distillation.py
.\.venv\Scripts\pytest.exe -q tests\test_human_demo_bc_observation_bridge.py `
  tests\test_missing_feature_distillation.py
```

Authorized distillation and final verification:

```powershell
.\.venv\Scripts\python.exe benchmarks\run_rival2_missing_feature_distillation.py
.\.venv\Scripts\python.exe benchmarks\run_rival2_missing_feature_distillation.py --verify-only
```

The final test metrics, human pre-BC inference baseline, training curve, hashes, and acceptance
verdict are stored under `results/rival2/missing_feature_distillation_v1/`.

## Frozen-run outcome

The authorized V1 run is blocked by its full-observation retention guard. The exact 32,768×128
corpus completed with 8,388,608 finite observations and canonical observation SHA-256
`F47EE006A1BEC75DD6D34F858748A207F2E2DF1161C0B72AA7F3C7B1AED57349`. At the first 64-step
boundary, all four frozen LR attempts (`1e-4`, `5e-5`, `2.5e-5`, `1.25e-5`) violated the actor and
value guard. Even the minimum-LR attempt reached full-input actor mean KL `0.264397263`, maximum
channel KL `0.135721892`, value RMSE `0.138564673`, and maximum value drift `0.878026009`, versus
limits `0.02`, `0.01`, `0.075`, and `0.5` respectively.

Every interval was rolled back to the byte-identical student initialization. No optimizer step was
accepted, no distilled checkpoint was emitted, the bootstrap and historical PPO optimizer remain
unchanged, and the human files/splits remain unchanged. Human post-distillation inference was not
run because no guard-accepted distilled student exists. V1 thresholds were not weakened. Any retry
with finer-grained transactional guarding or stronger retention enforcement requires a new
prospectively frozen authority version.

A separate read-only deterministic regeneration recorded the held-out test baselines without
constructing an optimizer: gameplay-degraded actor KL `6.721889837`, Freeplay-degraded actor KL
`26.916040359`, full-observation actor KL `0`, and full-observation value RMSE `0`. Post-distillation
values are unavailable rather than imputed because the guard rejected every candidate.
