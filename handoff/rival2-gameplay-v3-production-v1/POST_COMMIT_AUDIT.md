# Gameplay V3 Post-Commit Package Audit

Status: **mandatory supplement to the Gameplay V3 production handoff**

This file was added after re-opening the first committed package from Git and auditing it again against the source tree. Its requirements supplement and, if wording conflicts, override the earlier package wording.

No Gameplay V3 implementation existed when this audit was performed.

## Finding 1 — Gameplay V3 must not lose retained Gameplay event accumulation

Current `rival2_accumulate_tick()` only enters its Gameplay event-accumulation block for:

- `REWARD_MODE_GAMEPLAY`
- `REWARD_MODE_GAMEPLAY_V2`

That block accumulates state needed for retained V3 terms including:

- physical boost-use interval event;
- small/full boost-pad pickup events;
- save events based on pre/post ball threat.

If Codex simply adds `REWARD_MODE_GAMEPLAY_V3 = 5` and adds a final V3 formula without changing the event-accumulation dispatch, V3 would silently lose these retained Gameplay terms.

### Required implementation rule

Gameplay V3 must execute the same authoritative Gameplay event accumulation semantics for boost use, pad pickup, and saves.

Do this without changing V1/V2 behavior or arithmetic order.

Valid approaches include:

- explicitly include V3 in the Gameplay event-accounting condition while keeping historical reward composition branches intact; or
- extract a semantically identical event-accounting helper/kernel and prove V1/V2 exact compatibility.

### Required tests

Construct deterministic V1/V2/V3 cases proving that, for identical physics/event inputs:

- V3 physical boost-use detection matches Gameplay V1 semantics;
- V3 small/full pad event counts match Gameplay V1 semantics;
- V3 save detection matches Gameplay V1 semantics;
- V3 final touch component is still exactly `0.0`;
- V1/V2 emitted reward and component behavior remain unchanged.

## Finding 2 — current reward kernel has an unknown-mode fallthrough hazard

Current `rival2_accumulate_tick()` uses explicit branches for base/acquisition/goal-only and then a broad final `else` for Gameplay V1/V2 composition.

A new numerical reward mode can therefore accidentally enter the historical Gameplay branch unless V3 dispatch is made explicit.

### Required implementation rule

`REWARD_MODE_GAMEPLAY_V3` must have an explicit, auditable composition path.

Do not rely on a generic final `else` to distinguish V1/V2/V3.

The final branch structure must fail closed on an unsupported reward mode rather than silently treating an unknown mode as Gameplay.

A preferred safe shape is conceptually:

```text
if BASE:
    historical base
elif ACQUISITION:
    historical acquisition
elif GOAL_ONLY:
    historical goal only
elif GAMEPLAY or GAMEPLAY_V2:
    exact historical Gameplay path
elif GAMEPLAY_V3:
    explicit V3 composition
else:
    fail/flag unsupported mode
```

Warp limitations may require a device-safe impossible/error flag rather than a Python exception inside the kernel; if so, expose and assert that flag in construction/smoke tests.

### Required tests

- V3 cannot receive historical `+0.05` touch reward through fallthrough.
- V3 cannot receive the Gameplay V2 standalone double-dash reward through fallthrough.
- every supported reward mode maps to the intended branch.
- an unsupported test reward mode fails closed / raises before ordinary environment stepping, rather than behaving like Gameplay.

## Finding 3 — update every V3 reward-dispatch site, not only the kernel

Current V3 support must be wired consistently through all reward identity/dispatch paths.

At minimum audit and update/test:

- `rivalsim/rival2_contracts.py`
  - V3 version constant;
  - V3 immutable contract;
  - canonical V3 hash;
  - `contract_hashes_for_reward()` V3 dispatch.
- `rivalsim/rival2_env.py`
  - `Rival2Env.__init__` reward-version -> reward-mode dispatch;
  - `Rival2Env.set_reward_version()` dispatch, even though production campaign transition must use fresh-env checkpoint migration;
  - `Rival2WorldSim` V3-state allocation;
  - bridge binding conditional on V3 state.
- `rivalsim/kernels/rival2.py`
  - non-renumbering V3 mode constant;
  - explicit V3 event/reward path;
  - historical modes unchanged.
- checkpoint/curriculum transition validation
  - destination V3 contract lookup must succeed;
  - strict ordinary V2->V3 load must still reject.

If any benchmark/evaluator has a closed reward-version allowlist and is used by V3 validation, update that allowlist explicitly and test it. Do not expand unrelated tools merely for completeness.

## Finding 4 — full-shape rollout smoke is disposable and must never become campaign state

A real `collect_rollout()` call can legitimately advance runtime state such as:

- simulator world state;
- RNG/generator state;
- trainer `total_agent_samples` accounting.

That does not constitute PPO learning if no update is called, but it must not be persisted as the V3 campaign continuation state.

### Required validation rule

The 131,072-world horizon-32 rollout smoke and the 256-episode shadow gate must run in **disposable validation trainer/process state**.

- Never save that post-smoke trainer as a campaign checkpoint.
- Never overwrite or mutate the source checkpoint file.
- The authoritative future campaign start remains a newly constructed V3 trainer transitioned afresh from the exact accepted V2 source checkpoint after human review.
- Validation output must record source checkpoint iteration/policy/sample counters separately from disposable post-rollout accounting so there is no ambiguity.
- Model tensors and optimizer state must remain unchanged by validation.
- Policy version and iteration must remain unchanged because no PPO update ran.

The package phrase “sample counter not represented as training progress” means exactly this; the disposable object's sample counter may move during rollout collection, but it is not a resumable campaign state.

## Finding 5 — test the environment at every relevant reward identity

Because V3 state is optional and the bridge is constructed after the world, add a small construction/step matrix:

- Gameplay V1 environment constructs and steps;
- Gameplay V2 environment constructs and steps;
- Gameplay V3 environment constructs and steps;
- V1/V2 do not allocate/bind V3-only production state;
- V3 does allocate/bind the expected state;
- all observations remain exactly `(N, 2, 182)`;
- V1/V2 contract hashes and existing tests remain exact.

This catches accidental unconditional V3-state references in `Rival2TensorBridge` or `Rival2WorldSim` before the 131,072-world smoke.

## Finding 6 — V3 logical memory reporting must be derived from actual production arrays

The ~292 MiB shadow-observer estimate in the package is only a pre-implementation logical estimate.

For the final V3 production state:

- compute `logical_bytes` from the actual array inventory/shapes/dtypes, not a stale hand-maintained magic formula;
- include V3 state in `Rival2WorldSim.logical_state_bytes` only when allocated;
- compare the calculated incremental logical bytes against the actual CUDA allocated/reserved delta during the full-scale smoke;
- explain allocator overhead/caching differences rather than expecting equality;
- keep diagnostic evidence arrays out of production state.

## Finding 7 — current `world.step()` uses direct launches in the audited training environment

The audited `RivalSim.step()` calls `_launch_tick()` directly for each tick. CUDA graph capture exists as a separate optional API, and no repository callsite was found using it for the current Rival2 training path.

Therefore Gameplay V3 does not need a new graph-capture design for this task.

However, do not break the inherited capture API accidentally. If changed V3 code touches graph invalidation/capture-related state, add a focused smoke or document why it is unaffected. Do not broaden scope otherwise.

## Added release gate

Before `GAMEPLAY_V3_READY_FOR_REVIEW`, Codex must explicitly report these seven post-commit findings as `PASS` or `BLOCKED`, with evidence paths.
