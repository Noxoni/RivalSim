# Human BC V1 to 120 Hz PPO final audit

Verdict: `PASS_GREEN`

## Authority and transition

- Required parent at launch: `90faba5918abefc032089c331c074a66b2391b9d`.
- Accepted BC parent: `checkpoints/rival2/human_bc_v1/rival2_human_bc_v1.pt`.
- BC parent SHA-256: `560C2414C17039DC920126EA148BF73FE6CC4677EE440F043599A7E1C76D2874`.
- Exact rejected anchor: policy LR `1e-4`, minibatch KL `0.414318710565567`, retention KL `0.12192033976316452`.
- Highest passing transition LR: `1.25e-5`.
- Transition sweep and warmup authority were committed and pushed before any accepted step.
- Fresh PPO Adam was used; neither historical PPO Adam nor supervised BC optimizer state was restored.

The adaptive transition warmup remained active for all 4,067 updates. It did
not satisfy two consecutive clean updates at `1e-4`, so normal production
`1e-4` reset operation was never entered. Final update start/end LR was
`6.25e-6`; the prospectively authorized next start would be `1.25e-5`.

Starting-LR update counts were: `1e-4`: 36, `5e-5`: 59, `2.5e-5`: 351,
`1.25e-5`: 1,424, `6.25e-6`: 1,435, `3.125e-6`: 580, and `1.5625e-6`: 182.

## Campaign and checkpoints

- Wall-clock authority: 36,000 seconds.
- Measured elapsed time: 36,008.227 seconds.
- Accepted PPO offset: `+4067`.
- Final iteration / policy version: `4546` / `4547`.
- Final checkpoint: `checkpoints/rival2/human_bc_ppo_v1/rival2_human_bc_ppo_10h.pt`.
- Final checkpoint SHA-256: `C75F08F91E2FF29D78C15A67B5DE85BDF0B0A548E0268FFDB0CF29C272750067`.
- +30 checkpoint SHA-256: `E96774DE139156AE6C055E65033383CD50071F2BA73506C6D87F70DCAAD92B0D`.
- Agent decisions at 120 Hz: 30,733,257,575.
- Physical physics ticks: 17,062,428,672.

The final checkpoint is finite and resumable, has ten populated fresh-Adam
parameter states, contains CPU/CUDA/policy/opponent RNG states, retains a
bounded 16-policy historical pool, and records the exact BC parent lineage.

## Contracts and production-path audit

- `RIVAL2_ACTION_V2_120HZ`: `5E3747CCF9F59BA18D81D07014D60637F7D886907A0F44B0CA681C74F20EF91A`.
- `RIVAL2_OBS_V2_120HZ`: `BF9E141E5A1E5D2F15581C8BBB10F31F11FC5AA6736B327E61C03DD8D2388237`.
- `RIVAL2_REWARD_GAMEPLAY_120_V1`: `0D4C9A78803BBAF851AB4FDD7D9AC4196AB08E42B51DC0A173A1EAEC066AFAED`.
- `RIVAL2_EPISODE_V1`: `E22B6014C6D975D700D1129B9F554D6F35E4CA5003F1C1BD09C7D394D4F9347E`.

Every update recorded zero Nexto and Wisp training samples, no named-mechanics
hot path or arrays, zero ordinary-touch reward, and zero strict-dash component.
The mechanics-removal preflight remained unchanged with SHA-256
`583EE6CEB4F0E13B8CC366E47DDDF638D992390126DAA1327E4EB31CD139E8D3`.

## PPO safety

- Hard safety status: no hard guard fired.
- Maximum accepted post-step minibatch KL: `0.019998306408524513`.
- Maximum retention mean KL: `0.01999814063310623`.
- Maximum completed-update mean KL: `0.021767310798168182`.
- Updates with allowed soft retention early-stop: 84.
- Updates requiring at least one backoff: 1,617.
- Total LR backoffs: 2,447.
- Total transactional retries: 2,531.

The completed-update maximum is above the `0.02` soft target but below the
unchanged `0.05` hard guard. All accepted minibatch and retention measurements
remained within their `0.02` soft limits; rejected proposals were rolled back.

## Focused gameplay trend

First-100 to last-100 update-window means:

| Metric | First 100 | Last 100 | Delta |
|---|---:|---:|---:|
| Touches/min | 16.511825 | 19.198042 | +2.686217 |
| Goals/update | 1307.98 | 1424.92 | +116.94 |
| No-touch truncations/update | 6.38 | 4.14 | -2.24 |
| Movement speed (uu/s) | 1077.6763 | 1143.4074 | +65.7311 |
| Analog action saturation | 0.024030 | 0.021811 | -0.002219 |
| Unnecessary flip contacts/min | 7.557684 | 8.775441 | +1.217757 |
| Unnecessary / flip-active fraction | 0.617188 | 0.542080 | -0.075108 |

Touch acquisition, scoring, movement, no-touch behavior, and saturation do not
show catastrophic collapse. The unnecessary-contact fraction improved, but
the absolute rate increased alongside higher touch activity. This is evidence
for visual/opponent review, not proof that the undesirable habit is resolved.

## Validation

- Focused tests: 18 passed, 0 failed; 15 deprecation warnings.
- Ruff: passed.
- `git diff --check`: passed.
- Final checkpoint load/hash/finite/lineage/RNG/pool sanity: passed.
- No large opponent evaluation was run.

Recommended next step: launch a visual current-Rival versus historical-Rival or
Wisp/Nexto evaluation task without training, paying particular attention to
whether the lower unnecessary-contact fraction reflects legitimate power/50
conversion or merely greater overall contact volume.
