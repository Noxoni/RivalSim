# Rival 2.0 Human BC V3

> Final status: **BLOCKED**. Training and complete validation were safe, but
> the once-only prospective simulator test measured one all-perspective sample
> at actor KL `2.1052461326179`, above the unchanged `2.0` hard limit. The
> checkpoint is diagnostic evidence and is not an accepted BC parent. See
> `BLOCKED_AUDIT.md` and `evidence.json`.

Human BC V3 is an actor-only continuation from the accepted Human BC V1
checkpoint. It exists to preserve V2's human-imitation gains while addressing
the rare all-perspective simulator-retention tail that made V2 ineligible.
This lane does not run PPO, change rewards, alter demonstrations, change
mechanic adjudication, modify the observation adapter, or change any frozen
human split or control contract.

## Prospective authority

The files in this directory were generated and bound before the first V3
optimizer step:

- `frozen_config.json` is the complete training, selection, guard, and test
  authority. Its SHA-256 is
  `E82AFD2E8C34CB43792FD4DE01066692720C38E96A03BA6DB71953FAD7504450`.
- `retention_strata_manifest.json` binds the teacher-only training strata.
  Low variance means the bottom 10 percent by the frozen BC-V1 teacher's
  minimum clamped analog log standard deviation. No student output was used
  to define membership.
- `new_simulator_test_authority.json` binds the new deterministic test corpus
  at corpus seed `2026090107` and split seed `2026090108`. The corpus-seed
  namespace is disjoint from the previous train, validation, and now-opened
  diagnostic test worlds. Only its identity was generated before training;
  no V3 student was evaluated on it.
- `pre_step_preflight.json` records zero optimizer steps, zero PPO updates,
  zero human-test access, and zero V3 student evaluations of the new simulator
  test before the authority freeze.

The frozen BC-V1 parent is
`checkpoints/rival2/human_bc_v1/rival2_human_bc_v1.pt`, SHA-256
`560C2414C17039DC920126EA148BF73FE6CC4677EE440F043599A7E1C76D2874`.
The blocked V2 checkpoint is diagnostic evidence only and is prohibited as a
training parent.

## Structural training boundary

Only `actor.weight` and `actor.bias` are trainable. The shared trunk and critic
are frozen and must remain byte-identical to BC V1. A fresh AdamW optimizer is
created for the actor; no PPO optimizer or prior BC optimizer state is loaded.
The original V1 human objective, adapter, data mixture, and train/validation/test
splits remain authoritative.

Every retention step samples a fixed with-replacement mixture of ordinary
natural states, current-policy-applicable perspectives, historical-opponent
perspectives, and teacher-defined low-variance states. The loss combines mean
teacher-to-student actor KL with a smooth squared-softplus barrier activated at
sample KL `0.5`. The prior hard all-perspective maximum sample KL limit remains
`2.0`; existing mean, per-channel, critic, finite-output, and saturation guards
are not weakened.

Selection uses the entire simulator validation split of 3,277 worlds and tracks
all, current-policy-applicable, counterfactual-opponent, historical-opponent,
and low-teacher-variance perspectives. The opened old simulator test is not
used. After one validation-only checkpoint is selected, the human test split
and newly bound simulator test are each evaluated once, and selection is not
reopened.

## Commands

Focused validation before training:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_human_bc_v3.py tests/test_human_bc_v2.py tests/test_human_bc_continuation.py
.\.venv\Scripts\python.exe benchmarks/run_rival2_human_bc_v3.py --preflight-only
```

Authoritative training after the prospective Git authority is remotely
persisted:

```powershell
.\.venv\Scripts\python.exe benchmarks/run_rival2_human_bc_v3.py
```

Final evidence is written to `evidence.json`, `training_curve.json`,
`human_test_metrics.json`, `new_simulator_test_results.json`,
`artifact_manifest.json`, and `BLOCKED_AUDIT.md`. The selected checkpoint is
`checkpoints/rival2/human_bc_v3/rival2_human_bc_v3.pt`; it is blocked diagnostic
evidence and explicitly not PPO-resumable.
