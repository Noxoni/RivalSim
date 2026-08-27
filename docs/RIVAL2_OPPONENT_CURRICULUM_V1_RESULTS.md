# Rival 2.0 Opponent Curriculum V1 results

Status: `STOPPED_KL_GUARD_REJECTION`.

Source checkpoint: `77BF257131FB71DDEAEAE49D668C5E25AB1D06EE26149AB0D0AE303573CA5F21` at iteration `359`.

Gameplay V2 reward hash: `4073E29C1013458D5784435061FE47C639525BE37E8CD519783889C69BA87D41`.

Opponent mix per newly reset episode: 35% Nexto, 35% Wisp, 20% current Rival, and 10% historical Rival.

## Held-out opponent curve

| checkpoint | opponent | mode | episodes | Rival W-L-NG | goal diff | no-touch | first-touch | Rival/opp touches |
|---|---|---|---:|---:|---:|---:|---:|---:|
| source | Nexto | deterministic | 10 | 1-9-0 | -8 | 0.000000 | 1.000000 | 23/63 |
| source | Nexto | stochastic | 256 | 20-235-1 | -215 | 0.000000 | 1.000000 | 687/2097 |
| source | Wisp | deterministic | 10 | 2-3-5 | -1 | 0.000000 | 1.000000 | 47/124 |
| source | Wisp | stochastic | 256 | 90-118-48 | -28 | 0.000000 | 0.902344 | 777/2790 |

## Held-out side splits

| checkpoint | opponent | mode | Rival side | episodes | Rival W-L-NG | goal diff | no-touch | first-touch | Rival/opp touches |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| source | Nexto | deterministic | Blue | 5 | 0-5-0 | -5 | 0.000000 | 1.000000 | 11/39 |
| source | Nexto | deterministic | Orange | 5 | 1-4-0 | -3 | 0.000000 | 1.000000 | 12/24 |
| source | Nexto | stochastic | Blue | 128 | 11-116-1 | -105 | 0.000000 | 1.000000 | 342/1020 |
| source | Nexto | stochastic | Orange | 128 | 9-119-0 | -110 | 0.000000 | 1.000000 | 345/1077 |
| source | Wisp | deterministic | Blue | 5 | 1-2-2 | -1 | 0.000000 | 1.000000 | 27/55 |
| source | Wisp | deterministic | Orange | 5 | 1-1-3 | 0 | 0.000000 | 1.000000 | 20/69 |
| source | Wisp | stochastic | Blue | 128 | 39-65-24 | -26 | 0.000000 | 0.890625 | 400/1446 |
| source | Wisp | stochastic | Orange | 128 | 51-53-24 | -2 | 0.000000 | 0.914062 | 377/1344 |

## Checkpoints

| label | iteration | cumulative samples | SHA-256 | audit |
|---|---:|---:|---|---|
| pre_rejected_update_00360 | 359 | 3016538976 | `430F0113AEF437CA778552A40654AE09263CAC4487A455CC013467FAFD271481` | `PASS_GREEN` |

## Mandatory KL-guard stop

Rejected update: `360`.

Reason: `minibatch_kl_limit_exceeded`; post-step minibatch KL `0.106567591` exceeded the hard `0.1` limit.

No PPO update completed. Model, optimizer, gradients, and relevant RNG state were restored, and no later training or evaluation ran.

No v0.6 work, five-minute training, opponent training, imitation, or continuation was run.

Machine-readable PPO safety evidence, per-family sample ledgers, side-separated opponent evaluations, touch-height distributions, and dash-event evidence are stored under `results/rival2/opponent_curriculum_v1/`.
