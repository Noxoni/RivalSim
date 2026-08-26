# Rival 2.0 Campaign 02 Results

Campaign 02 completed the controlled entropy-off rerun. It stopped at update
`12` with `100,663,296` agent decision
samples, the first completed update crossing 100M. No later update and no v0.6 work ran.

## Independent verdicts

- execution status: `COMPLETE`;
- behavioral result: `IMPROVED`;
- initialization control: `PASS_GREEN`;
- final checkpoint continuation: `PASS_GREEN`;
- frozen v0.5 trainer: unchanged (`PASS_GREEN`).

The behavioral classification is the prospectively fixed Campaign 02 rule; it is independent of
execution correctness and was not adjusted after results were observed.

## Controlled-variable proof

- authorized Campaign 02 commit: `52713ef13309d8c5c219456ca6e66bdc10a5586a`;
- Campaign 01 closeout parent: `1ce5932cadd66b14032e61750836763499567bc9`;
- campaign seed: `20260826`;
- evaluation seed: `920260826`;
- worlds / horizon: `131,072` /
  `32`;
- Campaign 01 entropy coefficient: `0.01`;
- Campaign 02 entropy coefficient: `0.0`;
- all other PPO fields: exact match;
- model/contract/self-play/seed/evaluation fields: exact match;
- initialization model SHA-256: `890F224879DB6E458472985B226A664D8AE49B8303C21CFB0FD83A485CF42848`;
- Campaign 01 initialization SHA-256 match: exact;
- initialization evaluation semantic metrics: exact;
- evaluation protocol SHA-256: `964D7281C9BB8EF12C4A831B984015259A777D82285A02EAB329FBB6CC098CE7`.

Only non-semantic initialization-evaluation timestamps and wall-clock durations differ. The
diagnostic entropy metric remained logged, but its optimization contribution was exactly zero in
every Campaign 02 update.

## Checkpoint custody

- `000m`: update 0, 0 samples, `AF40E2C1FA2E3B7C4B34FB18802B3A99D84ECB444CF36AD6D8E729C9BAF1232F`, 6,074,626 bytes
- `010m`: update 2, 16,777,216 samples, `693D14FDBCF38E39E938F437A2F2759933B53718814E031B9743454B3B9D1CB9`, 13,601,402 bytes
- `025m`: update 3, 25,165,824 samples, `4D1A93F02F6855E139DD3BC762ADA547FA2A23A4B6522BBD023B9A7B19D5BA75`, 16,109,752 bytes
- `050m`: update 6, 50,331,648 samples, `B42546207535AF36C09CA8714C629C214B584C3EF28A6F70940803A0223A21FB`, 18,618,038 bytes
- `100m`: update 12, 100,663,296 samples, `4A9B366CD3A04222D639252EB2E3EBAD194AF2154D9DBFF213B1AF89A3909FA0`, 21,126,324 bytes

The final full resumable v0.5-format artifact is committed at
`checkpoints/rival2/campaign02/rival2_campaign02_100m_resume.pt`. Its exact size is
`21,126,324` bytes and SHA-256 is
`4A9B366CD3A04222D639252EB2E3EBAD194AF2154D9DBFF213B1AF89A3909FA0`.

## Direct Campaign 01 versus Campaign 02 evaluation

All values use the same 4,096-world, five-layout, first-episode protocol. `std` is the mean of the
five ordinary-self-play analog policy standard deviations.

| checkpoint | C01 touches/min | C02 touches/min | C01 stochastic goal diff | C02 stochastic goal diff | C01 stochastic touch diff | C02 stochastic touch diff | C01 std | C02 std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 000m | 0.272091 | 0.272091 | -2 | -2 | 15 | 15 | 1.000406 | 1.000406 |
| 010m | 0.222450 | 0.201576 | -10 | -2 | -35 | -36 | 1.010788 | 0.998347 |
| 025m | 0.203673 | 0.246547 | -16 | -5 | -37 | -8 | 1.095506 | 0.994279 |
| 050m | 0.209253 | 0.244693 | -15 | -1 | -30 | -30 | 2.540995 | 0.978897 |
| 100m | 0.175624 | 0.291182 | -16 | -3 | -46 | 35 | 2.646943 | 1.008370 |

Final Campaign 02 ordinary self-play had `0.291182`
touches/minute, `0.040362` goals/minute,
`0.989746` no-touch truncation fraction, and
`15.242952` seconds mean episode duration.

Final deterministic play against initialization produced goal differential
`0`, touch differential `-819`, and
outcomes `0` current wins / `0`
initialization wins / `4096` draws. Final stochastic play produced goal
differential `-3`, touch differential
`35`, and outcomes `19` /
`22` / `4055`.

## Prospective behavioral classification

- `ordinary_self_play_touches_per_simulated_minute`: init `0.272091`, C01 final `0.175624`, C02 final `0.291182`; C02-init `+0.019092`, C02-C01 `+0.115558`
- `stochastic_vs_initialization_goal_differential`: init `-2.000000`, C01 final `-16.000000`, C02 final `-3.000000`; C02-init `-1.000000`, C02-C01 `+13.000000`
- `stochastic_vs_initialization_touch_differential`: init `15.000000`, C01 final `-46.000000`, C02 final `35.000000`; C02-init `+20.000000`, C02-C01 `+81.000000`

The rule counted `2` improvements and
`1` regressions relative to initialization.
Campaign 02 was not worse than Campaign 01 on all three primary metrics:
`true`. The resulting
classification is **`IMPROVED`**.

## Optimizer diagnosis

| update | C01 entropy | C02 diagnostic entropy | C01 KL | C02 KL | C01 clip | C02 clip | C02 frozen-observation std |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 9.216151 | 9.166355 | 0.006836 | 0.006712 | 0.073601 | 0.074111 | 1.002328 |
| 2 | 9.216660 | 9.143536 | 0.009246 | 0.008194 | 0.066860 | 0.063496 | 0.994664 |
| 3 | 9.600565 | 9.112284 | 0.049464 | 0.006493 | 0.334012 | 0.069165 | 0.991839 |
| 4 | 11.785848 | 9.050990 | 1.085214 | 0.005091 | 0.616750 | 0.046522 | 0.989712 |
| 5 | 12.100061 | 8.992380 | 0.011614 | 0.005467 | 0.104144 | 0.054885 | 0.997118 |
| 6 | 13.950670 | 8.998739 | 0.763421 | 0.007089 | 0.692366 | 0.087534 | 0.989849 |
| 7 | 14.051939 | 8.984861 | 0.071255 | 0.006972 | 0.239422 | 0.084549 | 0.996445 |
| 8 | 14.132200 | 8.988533 | 0.007483 | 0.006793 | 0.072765 | 0.081452 | 1.003111 |
| 9 | 14.143708 | 9.010206 | 0.008340 | 0.006980 | 0.078458 | 0.083129 | 1.004006 |
| 10 | 14.154118 | 9.035633 | 0.007627 | 0.005687 | 0.087377 | 0.060878 | 1.012195 |
| 11 | 14.159352 | 9.123774 | 0.005476 | 0.006360 | 0.050125 | 0.073648 | 1.018883 |
| 12 | 14.162256 | 9.151028 | 0.004812 | 0.005679 | 0.042955 | 0.058526 | 1.015194 |

- maximum Campaign 02 approximate KL: `0.008194` at
  update `2`;
- maximum Campaign 02 clip fraction: `0.087534` at
  update `6`;
- threshold-flagged updates: none;
- Campaign 01 update-4 style instability recurred:
  `false`;
- final representative analog standard deviation:
  `1.015194` / `exp(1)` ceiling
  `2.718282`;
- standard deviation trended toward the ceiling:
  `false`.

The repository artifacts contain every policy/value loss, diagnostic entropy, total loss,
approximate KL, clip fraction, pre/post-clip gradient norm, fixed-observation standard deviation,
integrity check, transfer count, and policy/sample age for all 12 updates.

## Integrity, immutability, and boundary

All 12 updates passed finite rollout/loss/gradient/parameter/optimizer checks, action bounds,
binary buttons, done/reset accounting, frozen historical-opponent custody, version/sample age, and
zero hot-path H2D/D2H traffic. The final checkpoint reproduced deterministic outputs and the next
stochastic action/pre-tanh/log-probability exactly under the entropy-zero config identity.

Tracked v0.1-v0.5 results, Campaign 01 artifacts, and the frozen v0.5 training implementation
matched their prospectively frozen byte manifests at closeout. No inherited expensive simulator
authority rerun was required because no shared simulator/trainer implementation file changed.

Campaign 02 is closed. This result does not authorize a reward change, another hyperparameter
trial, curriculum work, or v0.6 RocketSim/RLBot transfer.
