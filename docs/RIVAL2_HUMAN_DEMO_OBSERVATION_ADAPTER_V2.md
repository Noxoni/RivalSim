# Rival human-demo observation adapter V2

## Scope and immutable authorities

`RIVAL2_HUMAN_DEMO_OBSERVATION_ADAPTER_V2` is an external, read-only domain bridge.
It repairs masked 182-field observations before the unchanged frozen iteration-479 Rival policy
consumes them. It does not behavior-clone human actions, run PPO, change rewards, change mechanic
detectors, or mutate the bootstrap model. The V1 whole-policy distillation failure and every V1
threshold/evidence artifact remain unchanged.

The prospective contract is
`results/rival2/human_demo_observation_adapter_v2/frozen_config.json`, SHA-256
`227AFE90C5678E299851C30D14F9CA914C1B05D679BA2D67440248DED30F08A1`. The runner rejects any
other bytes. It binds the adapter to bootstrap checkpoint SHA-256
`ADAF8D015C340CAFAE857B7253FBBDE3A6C842C4EA0BB091B31F8B1C210ED350`, bootstrap model tensor
SHA-256 `1AA50DC45E9E0FDD0B24510A26781787742BBE8C8ED5FF6B77FD72BEC3EFA8C3`, bridge quality
contract `49A6D3C09A3DD5C88263850CF816804FBF3B2BAAED28154F1C46EADED6B1D9BC`, and the unchanged V1
simulator corpus/split identities.

## Architecture and routing

The adapter is a two-hidden-layer, 256-unit SiLU MLP. Its input is the degraded float32[182]
observation, the committed quality code for every field, and a gameplay/Freeplay profile bit. Its
output is float32[182] in the exact `RIVAL2_OBS_V2_120HZ` field order. It is a separate artifact;
the mask is not appended to Rival and Rival's architecture is not modified.

Routing is structural:

- `full_authoritative` returns the original observation tensor directly, without evaluating an
  adapter parameter. Consequently, full-input actor KL and critic drift are exactly zero and stay
  zero even if every adapter parameter changes.
- `gameplay` uses the committed 16 exact-direct / 58 exactly-derived / 34 approximate / 74
  unavailable mask.
- `freeplay` uses the committed 8 / 36 / 22 / 116 mask. Opponent and relative-opponent fields are
  hard neutral nuisance context, excluded from reconstruction, and never presented as a fabricated
  opponent.

Exact/direct and exactly-derived fields are hard-copied after the MLP. Approximate fields permit
only a bounded residual and remain classified approximate. Unavailable bounded timers, flags, and
pad values use sigmoid imputation; other unavailable values use tanh imputation. They remain
classified unavailable. Per-sample quality masks are inputs only and are never modified or
promoted.

## Paired simulator objective

The fixed 32,768-world by 128-tick 120 Hz V1 corpus is regenerated from the clean bootstrap with
both car perspectives and the unchanged whole-world 26,214 / 3,277 / 3,277 split. Each training
sample retains the true complete observation and a degraded copy made with the same committed mask
as human recordings.

Only the adapter is optimized, using a fresh AdamW optimizer. The frozen policy has
`requires_grad=False`; its historical PPO optimizer is neither loaded into the adapter optimizer
nor mutated. The objective combines:

- frozen-teacher actor KL on repaired gameplay observations;
- reconstruction MSE on non-exact gameplay fields;
- reconstruction MSE on meaningful Freeplay self/ball/resource fields, with no Freeplay actor KL;
- a penalty on approximate-field residual magnitude.

Each adapter step checks the loss, gradients, and post-step parameters for nonfinite values and
rolls back before failing. Human actions are never inputs or targets.

## Native boost-pad observability boundary

The `nexto_1v1` event stream has authoritative pickup callbacks. The JSON event itself carries a
transient pointer but no position or canonical index; however, the recorder's event-backed frame
state retains that same stable pickup identity with its native position, respawn delay, and derived
cooldown after the pad's first observed pickup. The V2 pipeline maps those XY positions uniquely to
RivalSim's canonical 34-pad Soccar geometry. It then overlays the supported active/cooldown pair
after learned imputation, applying the orange-perspective pad remap when required.

The overlay is deliberately conservative. It is classified separately as approximate event-timed
support because the cooldown is derived from pickup time and respawn delay. The committed 182-field
input-quality mask remains unchanged and is not promoted. A pad before its first observed pickup,
and any pad never observed, stays unavailable and retains the adapter's imputation. Pointer sorting,
nearby-car guesses, and fabricated pre-session history are forbidden. The complete measured mapping,
coverage, conflicts, and remaining unknown indices are stored in `native_boost_pad_audit.json`.

## Frozen validation gates

Held-out gameplay starts from authoritative raw-degradation KL `6.721889836925684`. The adapter
must cut that by at least 50% and reach KL no greater than `3.360944918462842`. Gameplay,
meaningful-Freeplay, and pad reconstruction RMSE must each improve by at least 20%. Full-input actor
KL and critic drift must both be exactly zero. Every simulator and all 114,311 human outputs must be
finite; human actions, exact/derived fields, source hashes, and quality classifications must remain
unchanged.

## Commands

```powershell
.\.venv\Scripts\ruff.exe check rivalsim\human_demo\observation_adapter_v2.py `
  benchmarks\run_rival2_human_demo_observation_adapter_v2.py `
  tests\test_human_demo_observation_adapter_v2.py
.\.venv\Scripts\pytest.exe -q tests\test_human_demo_observation_adapter_v2.py `
  tests\test_human_demo_bc_observation_bridge.py `
  tests\test_missing_feature_distillation.py
.\.venv\Scripts\python.exe benchmarks\run_rival2_human_demo_observation_adapter_v2.py
.\.venv\Scripts\python.exe benchmarks\run_rival2_human_demo_observation_adapter_v2.py `
  --finalize-existing
```

The detailed run evidence, simulator reconstruction/KL metrics, pad audit, human inference audit,
training curve, and artifact manifest are under
`results/rival2/human_demo_observation_adapter_v2/`. The separate adapter checkpoint is
`checkpoints/rival2/observation_adapter_v2/rival2_human_demo_observation_adapter_v2.pt`.
