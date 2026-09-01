# Rival 2.0 — Fresh Human Seed PPO Exploration Ramp Fix

This is a **minimal correction** to the Fresh Human Seed v1 PPO transition. Do not create a new BC lineage and do not resume the stopped update-60 PPO descendant.

## Goal

Restart PPO from the exact selected Stage-1 human-imitation checkpoint and make the rollout policy begin operationally deterministic, then smoothly increase **one unified exploration schedule across all eight controller channels** after PPO has initialized.

The correction is specifically about action-sampling noise. Do not redesign rewards, BC, observations, opponents, PPO architecture, or safety machinery.

## Authoritative source

Use only:

`checkpoints/rival2/fresh_human_seed_v1/rival2_fresh_human_seed_v1.pt`

Required checkpoint SHA-256:

`CA2DB62A709BBD7DBA9D2997D701E2E6010584F8119BDA6F5D1686AD7425F9D2`

Required Stage-1 selected step: `2800`

Required validation complete-action RMSE: `0.09304266346161075`

Required untouched-test RMSE: `0.09109495009433785`

The stopped PPO update-60 run is diagnostic history only. It must not supply model weights, optimizer state, critic state, RNG state, counters, or opponent state.

## Preserve the existing PPO setup

Keep unchanged:

- native 120 Hz physics and policy cadence
- `RIVAL2_OBS_V2_120HZ`
- `RIVAL2_ACTION_V2_120HZ`
- `RIVAL2_REWARD_GAMEPLAY_120_V2`
- current-policy self-play only
- no Nexto/Wisp/historical opponents during training
- fresh critic initialization
- fresh PPO optimizer, RNG, counters, and empty historical pool
- full actor/trunk/critic PPO learning apart from any implementation detail required to make the scheduled exploration distribution coherent
- existing PPO gamma, GAE lambda, clip, value coefficient, minibatches, epochs, gradient clipping, and KL transactional guards
- existing reward values and all existing reward semantics
- 32,768 worlds when hardware permits the already-validated production configuration

Do not import any earlier Rival/BC/PPO policy.

## Unified exploration schedule

There is one normalized exploration progress value `a(update)` used for both analog and button sampling.

For accepted PPO update `u`:

```text
if u <= 60:
    a = 0
elif u >= 300:
    a = 1
else:
    x = (u - 60) / 240
    a = x*x*(3 - 2*x)   # smoothstep
```

The schedule value for an update must remain fixed for that rollout and the PPO optimization transaction that consumes it.

### Analog controls

The five analog channels are throttle, steer, pitch, yaw, and roll.

Use an effective Gaussian standard deviation driven by the unified schedule:

```text
sigma_start = 0.01
sigma_end   = 0.08
log_sigma(u) = lerp(log(sigma_start), log(sigma_end), a(update))
sigma(u) = exp(log_sigma(u))
```

This makes the first 60 updates effectively follow the learned BC action means, then gradually allows modest analog exploration through update 300.

Do not use the Stage-1 `log_std=-1` values as the rollout exploration scale. They caused approximately `sigma=0.368` and are the identified blocker.

The probability distribution used for sampling, stored old log-probability, recomputed PPO log-probability, ratios, and KL must all use the same effective scheduled analog distribution. Do not multiply sampled actions by a noise scalar while leaving PPO likelihood math on a different distribution.

If the cleanest implementation is to make the campaign use scheduled effective log-std instead of the actor's raw log-std rows, do so. Do not let raw log-std outputs bypass the schedule and raise exploration above the authorized effective value during this campaign.

### Button controls

Jump, boost, and handbrake follow **the same normalized schedule**. Do not leave buttons on the old unscheduled Bernoulli sampling while analog is near deterministic.

Preserve the learned BC button logits, but sharpen/relax the effective Bernoulli distribution using scheduled positive temperature:

```text
button_temperature_start = 0.02
button_temperature_end   = 0.50
T(u) = lerp(button_temperature_start, button_temperature_end, a(update))
effective_button_logits = learned_button_logits / T(u)
```

Because temperature is always positive, the deterministic button decision boundary/sign is unchanged. At the start, learned button decisions are nearly deterministic; as the same exploration curve rises, uncertain decisions gain increasing probability of alternative actions while strongly learned decisions remain relatively stable.

Sampling, stored old log-probability, PPO recomputation, ratios, entropy diagnostics, and KL must all use these same effective button logits. Do not sample with one temperature and optimize with another.

### Why the two numeric mappings differ

The **curve is shared**; the distribution parameterization differs because analog controls are continuous Gaussians and buttons are Bernoulli variables. The intent is one coherent controller-wide progression from near-deterministic imitation toward modest exploration, not independent noise policies.

## Restart semantics

Start a completely fresh Stage-2 PPO run from the Stage-1 checkpoint:

1. Load the exact Stage-1 model and policy config.
2. Preserve the learned trunk, analog means, and button logits.
3. Reinitialize the critic exactly as the Fresh Human Seed PPO transition already specifies.
4. Create a fresh PPO optimizer only after the corrected exploration transition is established.
5. Reset PPO iteration/policy counters and RNG state to a fresh campaign.
6. Start with no historical policies and pure current-policy self-play.
7. Do not resume the stopped update-60 checkpoint.

## Training campaign

Actually run the campaign after implementing the correction.

- target: `600` accepted PPO updates
- no early stop before `500` accepted updates unless an existing hard corruption/KL guard rejects an update and the existing authorized retry/backoff cannot recover
- preserve snapshots every 30 accepted updates through 480
- preserve explicit snapshots at 500, 510, 540, 570, and 600
- record the effective `a`, analog sigma, analog log-sigma, and button temperature with each update/snapshot

Keep the existing learning-rate behavior unless the already-authorized PPO transactional backoff requires one of the existing lower rates. Do not invent a new LR schedule as part of this fix.

## Minimal verification

Before the long run, verify only what is necessary to ensure the identified blocker is fixed:

- Stage-1 checkpoint SHA and selected step match exactly.
- Deterministic actor/trunk/button outputs are unchanged by the transition before PPO learning begins.
- At update 0, effective analog sigma is `0.01` and button temperature is `0.02`.
- At update 60 they remain at the start values.
- At update 300 they reach sigma `0.08` and button temperature `0.50`.
- Sampling and PPO log-probability/KL code paths use the same scheduled distributions.
- Fresh critic/optimizer/counters/pool are actually fresh.

Do not create new simulator validation corpora, retention systems, regression families, or unrelated tests.

## Telemetry to retain

Keep the existing PPO safety telemetry. Also retain enough normal gameplay telemetry to tell whether the identified symptom corrected itself, especially:

- touches / legitimate touches
- no-touch truncations/terminations
- goals
- possession/control reward occupancy
- supersonic occupancy
- effective exploration schedule values

These are diagnostics. Do not change the objective mid-run based on them unless an existing hard safety guard requires stopping.

## After training

After the 600-update campaign, evaluate the preserved snapshots/final candidate using the existing intended post-training process, including the Nexto comparison. Nexto must not enter training.

## Scope discipline

This task fixes one known blocker: excessive controller-wide stochasticity at the BC-to-PPO boundary.

Do not:

- alter Stage-1 BC or retrain it
- add retention toward an older Rival policy
- add mechanic labels/rewards/detectors
- redesign `RIVAL2_REWARD_GAMEPLAY_120_V2`
- import old policies
- make the stopped 60-update descendant authoritative
- add extra validation walls because the previous PPO attempt was stopped

## Commit and final response

Commit implementation, authority/evidence needed for this corrected run, snapshots/checkpoints, and the completed campaign results to `main` using the repository's established large-artifact handling.

Keep the final response short:

```text
STATUS: PASS / BLOCKED
COMMIT: <sha>
SOURCE: Stage-1 step 2800 / CA2DB62...
EXPLORATION: sigma 0.01→0.08; button T 0.02→0.50; smoothstep 60→300
TRAINING: <accepted updates>
CHECKPOINT: <selected/final path + sha>
RESULT: <one-line gameplay/Nexto outcome or blocker>
```
