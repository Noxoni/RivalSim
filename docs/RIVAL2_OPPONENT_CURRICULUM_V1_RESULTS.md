# Rival 2.0 Opponent Curriculum V1 results

Status: `COMPLETE_120_UPDATE_BOUNDARY`.

Source checkpoint: `77BF257131FB71DDEAEAE49D668C5E25AB1D06EE26149AB0D0AE303573CA5F21` at iteration `359`.

Gameplay V2 reward hash: `4073E29C1013458D5784435061FE47C639525BE37E8CD519783889C69BA87D41`.

Opponent mix per newly reset episode: 35% Nexto, 35% Wisp, 20% current Rival, and 10% historical Rival.

## Training and safety summary

The bounded continuation completed all 120 updates, iterations `360` through
`479`, and added `644343766` trainable Rival decision samples. All 120 training
ledger rows are `PASS_GREEN`; no hard KL guard fired.

| measure | result |
|---|---:|
| accepted / proposed optimizer steps | 18548 / 18591 |
| transactional retries | 43 |
| policy-LR backoffs | 34 |
| normal retention/soft-KL early stops | 9 |
| policy LR at the start of every update | 1e-4 (120/120) |
| policy LR at update end | 1e-4: 99; 5e-5: 8; 2.5e-5: 13 |
| critic LR | 3e-4 (120/120) |
| maximum accepted minibatch KL | 0.019681312 |
| maximum completed-update mean KL | 0.021269297 |
| maximum retention mean KL | 0.019727562 |
| maximum value-loss to trunk gradient | 0.0 |

Realized episode-family assignments were current `20.0544%`, historical
`9.9138%`, Nexto `35.0187%`, and Wisp `35.0131%`. Effective PPO sample shares
were current `43.7739%`, historical `7.9178%`, Nexto `23.9322%`, and Wisp
`24.3761%`; current-vs-current worlds contribute both Rival-controlled cars.

The monitored updates `360-369` all passed. Update 360 accepted 19 of 22
proposals, transactionally retried three times, backed off
`1e-4 -> 5e-5 -> 2.5e-5`, then ended normally at the retention budget with
completed KL `0.005559046`, maximum minibatch KL `0.017498828`, and retention
KL `0.019727562`. Updates 361-369 accepted every proposed step and each began
again at `1e-4`; their completed KL range was `0.005790-0.008660`.

## Stochastic gameplay/mechanics curve

| checkpoint | opponent | Rival W-L-NG | mean speed | supersonic | boost active | flips/min | airborne touch | touch height mean/p90 | strict double dash |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| source | Nexto | 20-235-1 | 1126.5 | 5.43% | 13.95% | 29.23 | 93.16% | 119.9 / 176.4 | 1 |
| +30 | Nexto | 41-214-1 | 1179.9 | 5.88% | 13.22% | 27.79 | 82.82% | 124.1 / 187.6 | 1 |
| +60 | Nexto | 24-230-2 | 1161.8 | 5.46% | 13.69% | 28.43 | 79.42% | 123.1 / 176.9 | 4 |
| +90 | Nexto | 23-230-3 | 1150.8 | 5.65% | 13.53% | 26.91 | 91.39% | 121.6 / 172.5 | 2 |
| +120 | Nexto | 23-233-0 | 1169.5 | 5.95% | 14.02% | 28.39 | 90.37% | 121.9 / 172.5 | 2 |
| source | Wisp | 90-118-48 | 1162.4 | 4.22% | 11.72% | 27.26 | 89.32% | 115.0 / 166.1 | 8 |
| +30 | Wisp | 145-81-30 | 1216.7 | 4.76% | 12.73% | 26.64 | 84.55% | 111.9 / 158.5 | 5 |
| +60 | Wisp | 169-52-35 | 1228.3 | 5.38% | 13.79% | 27.24 | 86.85% | 109.9 / 153.0 | 4 |
| +90 | Wisp | 192-43-21 | 1278.7 | 6.38% | 16.64% | 26.67 | 94.38% | 105.0 / 126.5 | 1 |
| +120 | Wisp | 191-46-19 | 1259.7 | 6.66% | 17.50% | 26.24 | 86.71% | 107.7 / 138.6 | 1 |

The behavioral result is asymmetric: Rival became decisively strong against
Wisp while stochastic Nexto performance peaked at +30 and then regressed close
to the source level. Acquisition did not collapse: no-touch remained zero in
every scheduled suite, and final stochastic first-touch share was 100% against
both opponents. The evidence does not show increased high-ball capability;
touch-height percentiles stayed broadly flat or decreased. Strict double-dash
events remained rare and did not explain the Wisp gains.

## Held-out opponent curve

| checkpoint | opponent | mode | episodes | Rival W-L-NG | goal diff | no-touch | first-touch | Rival/opp touches |
|---|---|---|---:|---:|---:|---:|---:|---:|
| source | Nexto | deterministic | 10 | 1-9-0 | -8 | 0.000000 | 1.000000 | 23/63 |
| source | Nexto | stochastic | 256 | 20-235-1 | -215 | 0.000000 | 1.000000 | 687/2097 |
| source | Wisp | deterministic | 10 | 2-3-5 | -1 | 0.000000 | 1.000000 | 47/124 |
| source | Wisp | stochastic | 256 | 90-118-48 | -28 | 0.000000 | 0.902344 | 777/2790 |
| plus_030 | Nexto | deterministic | 10 | 0-10-0 | -10 | 0.000000 | 1.000000 | 22/82 |
| plus_030 | Nexto | stochastic | 256 | 41-214-1 | -173 | 0.000000 | 0.992188 | 786/1984 |
| plus_030 | Wisp | deterministic | 10 | 6-3-1 | 3 | 0.000000 | 1.000000 | 29/30 |
| plus_030 | Wisp | stochastic | 256 | 145-81-30 | 64 | 0.000000 | 1.000000 | 686/1855 |
| plus_060 | Nexto | deterministic | 10 | 0-10-0 | -10 | 0.000000 | 1.000000 | 22/85 |
| plus_060 | Nexto | stochastic | 256 | 24-230-2 | -206 | 0.000000 | 0.988281 | 904/2275 |
| plus_060 | Wisp | deterministic | 10 | 9-1-0 | 8 | 0.000000 | 1.000000 | 25/44 |
| plus_060 | Wisp | stochastic | 256 | 169-52-35 | 117 | 0.000000 | 1.000000 | 730/1846 |
| plus_090 | Nexto | deterministic | 10 | 3-7-0 | -4 | 0.000000 | 1.000000 | 27/45 |
| plus_090 | Nexto | stochastic | 256 | 23-230-3 | -207 | 0.000000 | 0.996094 | 825/2390 |
| plus_090 | Wisp | deterministic | 10 | 10-0-0 | 10 | 0.000000 | 1.000000 | 16/5 |
| plus_090 | Wisp | stochastic | 256 | 192-43-21 | 149 | 0.000000 | 0.996094 | 587/1262 |
| plus_120 | Nexto | deterministic | 10 | 0-10-0 | -10 | 0.000000 | 1.000000 | 77/147 |
| plus_120 | Nexto | stochastic | 256 | 23-233-0 | -210 | 0.000000 | 1.000000 | 841/2207 |
| plus_120 | Wisp | deterministic | 10 | 10-0-0 | 10 | 0.000000 | 1.000000 | 20/4 |
| plus_120 | Wisp | stochastic | 256 | 191-46-19 | 145 | 0.000000 | 1.000000 | 602/1044 |

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
| plus_030 | Nexto | deterministic | Blue | 5 | 0-5-0 | -5 | 0.000000 | 1.000000 | 12/55 |
| plus_030 | Nexto | deterministic | Orange | 5 | 0-5-0 | -5 | 0.000000 | 1.000000 | 10/27 |
| plus_030 | Nexto | stochastic | Blue | 128 | 23-105-0 | -82 | 0.000000 | 0.992188 | 359/954 |
| plus_030 | Nexto | stochastic | Orange | 128 | 18-109-1 | -91 | 0.000000 | 0.992188 | 427/1030 |
| plus_030 | Wisp | deterministic | Blue | 5 | 3-1-1 | 2 | 0.000000 | 1.000000 | 21/18 |
| plus_030 | Wisp | deterministic | Orange | 5 | 3-2-0 | 1 | 0.000000 | 1.000000 | 8/12 |
| plus_030 | Wisp | stochastic | Blue | 128 | 76-39-13 | 37 | 0.000000 | 1.000000 | 354/801 |
| plus_030 | Wisp | stochastic | Orange | 128 | 69-42-17 | 27 | 0.000000 | 1.000000 | 332/1054 |
| plus_060 | Nexto | deterministic | Blue | 5 | 0-5-0 | -5 | 0.000000 | 1.000000 | 11/47 |
| plus_060 | Nexto | deterministic | Orange | 5 | 0-5-0 | -5 | 0.000000 | 1.000000 | 11/38 |
| plus_060 | Nexto | stochastic | Blue | 128 | 8-120-0 | -112 | 0.000000 | 0.976562 | 406/1220 |
| plus_060 | Nexto | stochastic | Orange | 128 | 16-110-2 | -94 | 0.000000 | 1.000000 | 498/1055 |
| plus_060 | Wisp | deterministic | Blue | 5 | 5-0-0 | 5 | 0.000000 | 1.000000 | 8/7 |
| plus_060 | Wisp | deterministic | Orange | 5 | 4-1-0 | 3 | 0.000000 | 1.000000 | 17/37 |
| plus_060 | Wisp | stochastic | Blue | 128 | 83-25-20 | 58 | 0.000000 | 1.000000 | 401/954 |
| plus_060 | Wisp | stochastic | Orange | 128 | 86-27-15 | 59 | 0.000000 | 1.000000 | 329/892 |
| plus_090 | Nexto | deterministic | Blue | 5 | 1-4-0 | -3 | 0.000000 | 1.000000 | 14/24 |
| plus_090 | Nexto | deterministic | Orange | 5 | 2-3-0 | -1 | 0.000000 | 1.000000 | 13/21 |
| plus_090 | Nexto | stochastic | Blue | 128 | 10-115-3 | -105 | 0.000000 | 1.000000 | 421/1189 |
| plus_090 | Nexto | stochastic | Orange | 128 | 13-115-0 | -102 | 0.000000 | 0.992188 | 404/1201 |
| plus_090 | Wisp | deterministic | Blue | 5 | 5-0-0 | 5 | 0.000000 | 1.000000 | 8/2 |
| plus_090 | Wisp | deterministic | Orange | 5 | 5-0-0 | 5 | 0.000000 | 1.000000 | 8/3 |
| plus_090 | Wisp | stochastic | Blue | 128 | 93-27-8 | 66 | 0.000000 | 1.000000 | 285/662 |
| plus_090 | Wisp | stochastic | Orange | 128 | 99-16-13 | 83 | 0.000000 | 0.992188 | 302/600 |
| plus_120 | Nexto | deterministic | Blue | 5 | 0-5-0 | -5 | 0.000000 | 1.000000 | 58/75 |
| plus_120 | Nexto | deterministic | Orange | 5 | 0-5-0 | -5 | 0.000000 | 1.000000 | 19/72 |
| plus_120 | Nexto | stochastic | Blue | 128 | 14-114-0 | -100 | 0.000000 | 1.000000 | 372/1062 |
| plus_120 | Nexto | stochastic | Orange | 128 | 9-119-0 | -110 | 0.000000 | 1.000000 | 469/1145 |
| plus_120 | Wisp | deterministic | Blue | 5 | 5-0-0 | 5 | 0.000000 | 1.000000 | 12/2 |
| plus_120 | Wisp | deterministic | Orange | 5 | 5-0-0 | 5 | 0.000000 | 1.000000 | 8/2 |
| plus_120 | Wisp | stochastic | Blue | 128 | 96-25-7 | 71 | 0.000000 | 1.000000 | 285/562 |
| plus_120 | Wisp | stochastic | Orange | 128 | 95-21-12 | 74 | 0.000000 | 1.000000 | 317/482 |

## Checkpoints

| label | iteration | cumulative samples | SHA-256 | audit |
|---|---:|---:|---|---|
| plus_030 | 389 | 3167194343 | `B4C06AD6650FAD697A28CBFACB3A0502DC3B5E44156E7DC4D8786C43C931893A` | `PASS_GREEN` |
| plus_060 | 419 | 3327281168 | `32BD32D34CB53E9E2FB86C6A35EB687B90441B10D822D4CBDDFE5BC0B55DF34B` | `PASS_GREEN` |
| plus_090 | 449 | 3490451155 | `E08F2FE8DEDE4506D310EEF8A785619BA89C0C88B51AAEEE8640592BFDE4BA00` | `PASS_GREEN` |
| plus_120 | 479 | 3655854038 | `3B994E118A9498713DC6115D38F061958A900EA8F4D00CE568F916942E851D9A` | `PASS_GREEN` |

## Boundary

Training stopped at the authorized +120 boundary.

No v0.6 work, five-minute training, opponent training, imitation, or continuation was run.

Machine-readable PPO safety evidence, per-family sample ledgers, side-separated opponent evaluations, touch-height distributions, and dash-event evidence are stored under `results/rival2/opponent_curriculum_v1/`.
