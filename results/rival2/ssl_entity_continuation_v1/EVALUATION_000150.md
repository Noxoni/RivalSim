# +150: better acquisition/finishing, Nexto still unresolved

The scheduled deterministic development evaluation completed at
2026-09-05 21:44:34 UTC. Same scenario hashes and 64 worlds per case as +100.
These are short scenario episodes, **not** full-match win rates. Self-play
opposition changes with the current model; the pinned Nexto case is the fixed
opponent anchor. No untouched test data or checkpoint selection was involved.

| Metric | +100 | +150 |
|---|---:|---:|
| Acquisition worlds with a Rival touch | 51/64 | 52/64 |
| Acquisition contacts | 59 | 69 |
| Acquisition goals for / against | 10 / 2 | 27 / 1 |
| Acquisition no-touch truncations | 50 | 34 |
| Acquisition conditional median first touch, s | 0.966667 | 0.966667 |
| Finishing worlds with a Rival touch | 62/64 | 62/64 |
| Finishing contacts | 82 | 86 |
| Finishing goals for / against | 35 / 0 | 42 / 0 |
| Finishing no-touch truncations | 28 | 22 |
| Finishing conditional median first touch, s | 0.525000 | 0.533333 |
| Nexto kickoff worlds with a Rival touch | 13/64 | 13/64 |
| Nexto kickoff Rival contacts | 26 | 13 |
| Nexto kickoff goals for / against | 0 / 64 | 0 / 64 |
| Nexto kickoff conditional median first touch, s | 2.575000 | 2.691667 |

No-touch truncations can occur after earlier contact, so they are not the
complement of touch coverage. Conditional first-touch speed does not measure
failure-to-acquire cases. The Nexto case's positive resulting-ball-velocity
fraction also falls from1 to0 and exposure increases 538.101s to631.668s;
neither the longer survival nor the acquisition-case gains establishes better
Nexto play. The unchanged 0/64 scoring and fewer repeated contacts remain
concerning. Do not claim a Nexto breakthrough.

## Prospective opponent transition actually executed

The original acquisition criterion passed for a **second consecutive** scheduled
boundary: +100 and +150 both exceed .60 touch coverage, conditional median
first touch <=5s and at least one finishing goal. +50 did not pass. The runner
therefore changed the already frozen schedule to20% Nexto /80% current-selfplay
**episode assignments at subsequent physical resets**, without new thresholds,
reward changes or another parent. It did not make20% of all ongoing worlds or
all training samples switch instantly.

The +150 checkpoint includes the enabled probability and streak2 but contains
no learning against Nexto yet. Actual first mixed rollouts are preserved in
`nexto_activation_evidence.json`:

- +151 includes60,677 current-agent samples against Nexto;
- +152 includes161,458 current-agent samples against Nexto.

For each, the count excluded from the potential5,898,240 two-current-agent
samples exactly equals the Nexto-opponent slots. Nexto's actions and returns
remain excluded from PPO, while current-player samples train normally. The
earlier native tests verify actual tick controls, reset assignment, family-local
normalization and masked-update invariance. No scripted action prefix acts for
Rival. No architecture, reward, physics or observation changes occurred.

This is the next relevant learning exposure: acquisition and simple finishing
are improving, but the model has not learned reliable possession or scoring
under Nexto's pressure. Continue the frozen mixed phase and inspect +200, not
a new restart or unmeasured reward change. Future full matches remain necessary
to assess possession and actual scoring; the most recent completed full-match
comparison is still +100, not +150.

## Training and checkpoint integrity

50 continuation updates, +101 through+150, are preserved as an immutable
UTF8/LF prefix in `training_curve_through_000150.jsonl`. Do not substitute the
still-growing live log. Count-weighted ten-update blocks are retained in
`evaluation_000150_comparison.json`.

Last10 pilot updates versus last10 continuation updates:

- contacts/player-minute3.0001 ->3.6156;
- goals/world-minute1.2915 ->1.8104;
- ended-player-episode touch fraction51.03% ->56.42%;
- no-touch share of world resets68.99% ->53.85%;
- mean movement speed738.62 ->864.50uu/s;
- action entropy3.3769 ->3.1918, not a collapsed distribution.

These are stochastic training statistics, not fixed-opponent outcomes. All
fifty rows used pure self-play; mixing began only after the evaluation.
The first continuation block starts fresh episodes and must not be interpreted
as an immediate causal improvement over the pilot's steady-state rollouts.

The independent CPU-only checkpoint audit passed:

- +150 model/Adam finite; parent remains byte-identical;
- actor/action-table/observation identities preserved;
- Adam counters27,300 =18,200 parent +9,100 additional steps;
-294,912,000 additional trainable current-agent decisions;
-884,736,000 cumulative entity-phase decisions;
-1,769,472,000 cumulative entity-phase physical **world** ticks;
- largest completed-update mean KL0.00581897;
- largest sample KL16.808094, recorded as telemetry, zero KL rejections;
- no optimizer step was executed by the audit.

These exposure totals refer to the entity phase, not all earlier fresh-lineage
training. Integrity PASS is not an SSL or deployment verdict.

Checkpoint: `checkpoints/rival2/ssl_entity_continuation_v1/plus_000150.pt`

SHA256: `D88B7C9FD16A42C57970AC488D9A56C0F6174B77D591DDD0D6BFAABBD8AB4449`

Full numerical evidence: `evaluation_000150.json`,
`evaluation_000150_comparison.json`, `accepted_000150_integrity.json`,
`nexto_activation_evidence.json`, and the frozen training-curve prefix.
