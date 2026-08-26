# Rival 2.0 vs pinned public Nexto

Verdict: **PASS_GREEN**.

The result is intentionally reported by Rival's assigned team. Blue and Orange are never collapsed into a headline aggregate because team assignment materially changes the observed scoring distribution.

## Canonical deterministic deployment suite (primary)

- Rival as **Blue**: 5-0 (5 regulation, 0 OT wins), win rate 100.0000%, goals 87-33, GD mean 10.800, median 11.000
- Rival as **Orange**: 5-0 (5 regulation, 0 OT wins), win rate 100.0000%, goals 112-27, GD mean 17.000, median 19.000

| Layout | Rival side | Blue | Orange | Rival-Nexto | Winner | OT |
|---|---:|---:|---:|---:|---|---:|
| 0 (Blue diagonal-left) | Blue | 20 | 6 | 20-6 | Rival | no |
| 0 (Blue diagonal-left) | Orange | 4 | 28 | 28-4 | Rival | no |
| 1 (Blue diagonal-right) | Blue | 16 | 10 | 16-10 | Rival | no |
| 1 (Blue diagonal-right) | Orange | 6 | 19 | 19-6 | Rival | no |
| 2 (Blue off-center-left) | Blue | 22 | 6 | 22-6 | Rival | no |
| 2 (Blue off-center-left) | Orange | 4 | 24 | 24-4 | Rival | no |
| 3 (Blue off-center-right) | Blue | 12 | 5 | 12-5 | Rival | no |
| 3 (Blue off-center-right) | Orange | 4 | 23 | 23-4 | Rival | no |
| 4 (center) | Blue | 17 | 6 | 17-6 | Rival | no |
| 4 (center) | Orange | 9 | 18 | 18-9 | Rival | no |

These ten trajectories are the complete deterministic 5-layout by 2-side matrix; duplicated copies would not be independent evidence.

Physical-team totals in the canonical matrix: Blue 114 goals / 5 wins; Orange 145 goals / 5 wins.

## Stochastic Rival robustness suite (secondary)

- Rival as **Blue** (2,048 matches): 2041-7 (2034 regulation, 7 OT wins), win rate 99.6582%, goals 36777-12812, GD mean 11.702, median 12.000
- Rival as **Orange** (2,048 matches): 2043-5 (2041 regulation, 2 OT wins), win rate 99.7559%, goals 41313-9793, GD mean 15.391, median 15.000
- Physical-team totals: Blue 46,570 goals / 2,046 wins; Orange 54,125 goals / 2,050 wins.

This suite samples Rival's ordinary hybrid policy distribution with a fixed seed. Nexto remains deterministic. It is a robustness measurement, not the headline deployment matchup.

## Observed Blue/Orange asymmetry

In the deterministic matrix, Rival-as-Orange scored 5.000 more goals per match and conceded 1.200 fewer than Rival-as-Blue; mean goal differential was 6.200 higher as Orange.
In the stochastic suite, Rival-as-Orange scored 2.215 more goals per match and conceded 1.474 fewer; mean goal differential was 3.689 higher as Orange.
This is a descriptive benchmark finding, not a causal diagnosis. No simulator, observation, policy, reward, or controller behavior was changed in response to it.

## Side-separated behavior

| Suite / Rival side | Touches | Touch share | Kickoff first touches | Kickoff goals | Same next touch | Opponent handoff | Demos |
|---|---:|---:|---:|---:|---:|---:|---:|
| Canonical / Blue | 641 | 0.452364 | 100 | 75 | 0.712980 | 0.287020 | 1 |
| Canonical / Orange | 709 | 0.464309 | 112 | 86 | 0.745000 | 0.255000 | 1 |
| Stochastic / Blue | 347,611 | 0.516789 | 38,630 | 30,081 | 0.785730 | 0.214270 | 396 |
| Stochastic / Orange | 359,871 | 0.532528 | 39,865 | 31,089 | 0.786427 | 0.213573 | 353 |

The machine-readable suite files additionally retain side-separated forward/neutral/backward touch direction, net displacement, wall/backboard continuation, complete goal-mouth X/Z histograms, and layout-by-side results. Backward, lateral, wall, and backboard touches are descriptive categories, not value judgments.

## Fidelity and runtime gates

- Observation parity: q `0.0`, kv `6.106226635438361e-16`, mask `0.0` max absolute error.
- Deterministic action agreement: `2048/2048` (100%).
- Action table: `90` actions, SHA-256 `86BAA15C48C42C497F3EA0FE62EFEB49E4A8241CB3191957822E453CD2D0B655`.
- Stock kickoff: `168` controls at 120 Hz, SHA-256 `55BBDEB064EE173D9D2F48CBD3109509F82FDE1AA5EBD176B86C9AF2768A7FEF`.
- Fidelity hot-path H2D/D2H events: `0`.
- Nexto observation throughput at fidelity batch 2048: `766045.54` worlds/s.
- Nexto model throughput at fidelity batch 2048: `1662851.68` inferences/s.
- Canonical match throughput: `3894.58` world-ticks/s; peak CUDA `0.036` GiB.
- Stochastic match throughput: `994344.07` world-ticks/s; peak CUDA `0.422` GiB.
- Timed match-loop transfer profiler events: canonical `0`, stochastic `0`.

Compact match-status exports occur only after regulation and at coarse overtime boundaries; they are outside the timed per-tick loop. The match clock intentionally omits Rocket League's zero-second airborne continuation rule, as authorized. Training episode timeouts and no-touch truncation do not control this runtime.

## Identity

- Rival checkpoint SHA-256: `4DC158DC2A9D16B79FB5FE7D868E3B50928AB113B55DFCC753F3734F8D87372E`
- Rival policy version / samples: `5403` / `45,323,649,024`
- Nexto upstream commit: `2e6ed7d6ed2b352e8ff529d4a12a0c9c70c28cca`
- Nexto model SHA-256: `BF5343B5EEACAC6BF7CDB75DAC4A5C14BA0F94D820EAE75F00A211B6119D69FA`
- Pinned upstream license: CC BY-NC-SA 4.0; exact source/model/license blobs are isolated under `third_party/nexto/`.

## Evidence files

- `results/rival2/nexto/fidelity.json`
- `results/rival2/nexto/canonical_deterministic.json`
- `results/rival2/nexto/canonical_match_ledger.json`
- `results/rival2/nexto/stochastic_robustness.json`
- `results/rival2/nexto/summary.json`
