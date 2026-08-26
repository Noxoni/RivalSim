# Rival 2.0 Overnight Curriculum Results

The authorized uninterrupted curriculum completed from the exact Campaign 04 checkpoint.
Reward V2 acquisition training continued until two consecutive 4,096-world held-out
evaluations met the no-touch threshold, the reward alone was explicitly migrated to the
preserved base `RIVAL2_REWARD_V1`, exactly 239 additional PPO updates were completed, and the
same V1 policy then trained until the first completed update at or after six real elapsed
hours. The viewer was not built and v0.6 was not begun.

## Phase A — Reward V2 acquisition completion

| Phase A update | Update | Cumulative samples | Touches/min | Goals/min | Goal fraction | No-touch fraction | Mean duration, s |
|---|---:|---:|---:|---:|---:|---:|---:|
| 150 | 150 | 1,258,291,200 | 17.207281 | 0.263087 | 0.128906 | 0.528564 | 29.398551 |
| 180 | 180 | 1,509,949,440 | 27.106260 | 0.331807 | 0.197998 | 0.230957 | 35.803646 |
| 210 | 210 | 1,761,607,680 | 31.045811 | 0.522961 | 0.306641 | 0.135498 | 35.181283 |
| 240 | 240 | 2,013,265,920 | 34.175757 | 0.686807 | 0.395264 | 0.104248 | 34.530542 |
| 270 | 270 | 2,264,924,160 | 37.947045 | 1.028846 | 0.559814 | 0.046143 | 32.647135 |
| 300 | 300 | 2,516,582,400 | 47.024301 | 1.106932 | 0.592529 | 0.027832 | 32.117391 |
| 330 | 330 | 2,768,240,640 | 53.209083 | 1.511728 | 0.729248 | 0.011719 | 28.943620 |
| 360 | 360 | 3,019,898,880 | 53.702787 | 1.363763 | 0.695557 | 0.010010 | 30.601644 |
| 390 | 390 | 3,271,557,120 | 57.441686 | 1.781317 | 0.808350 | 0.007324 | 27.227596 |
| 420 | 420 | 3,523,215,360 | 56.322408 | 1.685836 | 0.786377 | 0.006104 | 27.987671 |

The first qualifying consecutive pair was updates
`390` and `420`, with no-touch fractions
`0.007324` and
`0.006104`. The confirming checkpoint is
`checkpoints/rival2/overnight/rival2_overnight_acquisition_complete_resume.pt` with SHA-256
`F25BF8DBEA9FA316CAB299F7ECA8D243EEB9F44423081D2964FF1352689619E7`.

## Explicit Reward V2 -> Reward V1 transition

The transition occurred at update `420` /
`3,523,215,360` cumulative samples. The
source checkpoint SHA-256 was `F25BF8DBEA9FA316CAB299F7ECA8D243EEB9F44423081D2964FF1352689619E7`. Every
model, optimizer, RNG, counter, assignment, historical-policy, and live runtime identity check
was exact; only the reward version/contracts changed, and the transition record was embedded in
the post-transition and all descendant checkpoints.

## Phase B — 2B additional base-reward samples

| Phase B offset | Update | Cumulative samples | Touches/min | Goals/min | Goal fraction | No-touch fraction | Mean duration, s |
|---|---:|---:|---:|---:|---:|---:|---:|
| +60 | 480 | 4,026,531,840 | 60.466974 | 1.877070 | 0.839355 | 0.007080 | 26.829761 |
| +120 | 540 | 4,529,848,320 | 68.894077 | 2.365756 | 0.917236 | 0.003662 | 23.262826 |
| +180 | 600 | 5,033,164,800 | 57.061773 | 2.062022 | 0.764648 | 0.152588 | 22.249479 |
| +239 | 659 | 5,528,092,672 | 70.371717 | 2.013538 | 0.775146 | 0.139893 | 23.098047 |

Phase B completed exactly `239` updates /
`2,004,877,312` additional samples. The durable 2B
base-reward checkpoint is `checkpoints/rival2/overnight/rival2_overnight_2b_base_reward_resume.pt` with SHA-256
`5AFA138BE8513968BE126FB2548A3C803AEDF5E19ED491D2F4059DCC93C7AA59` and passed exact reload.

## Phase C — six real elapsed hours

| Elapsed point | Update | Cumulative samples | Touches/min | Goals/min | Goal fraction | No-touch fraction | Mean duration, s |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1h | 1460 | 12,247,367,680 | 66.022572 | 2.598408 | 0.932129 | 0.003418 | 21.523844 |
| 2h | 2248 | 18,857,590,784 | 75.795357 | 2.574506 | 0.928955 | 0.005371 | 21.649707 |
| 3h | 3031 | 25,425,870,848 | 79.884180 | 2.296324 | 0.899902 | 0.006104 | 23.513289 |
| 4h | 3820 | 32,044,482,560 | 75.911922 | 2.517297 | 0.933594 | 0.002197 | 22.252287 |
| 5h | 4609 | 38,663,094,272 | 83.108105 | 2.586748 | 0.931641 | 0.004150 | 21.609538 |
| 6h | 5403 | 45,323,649,024 | 85.483708 | 2.496403 | 0.926270 | 0.003418 | 22.262500 |

The hourly trigger details were:

- 1h: update 1460, 12,247,367,680 samples, 3600.958 elapsed seconds at update completion, 3625.739 seconds after evaluation.
- 2h: update 2248, 18,857,590,784 samples, 7204.405 elapsed seconds at update completion, 7228.889 seconds after evaluation.
- 3h: update 3031, 25,425,870,848 samples, 10801.228 elapsed seconds at update completion, 10825.795 seconds after evaluation.
- 4h: update 3820, 32,044,482,560 samples, 14400.690 elapsed seconds at update completion, 14425.328 seconds after evaluation.
- 5h: update 4609, 38,663,094,272 samples, 18001.968 elapsed seconds at update completion, 18026.228 seconds after evaluation.
- 6h: update 5403, 45,323,649,024 samples, 21601.926 elapsed seconds at update completion, 21626.613 seconds after evaluation.

The final training update completed at
`21601.926` elapsed seconds. Its held-out
evaluation recorded `85.483708` touches/minute,
`2.496403` goals/minute, and
`0.003418` no-touch truncation.

## Training integrity and final checkpoint

All `5283` continuation updates passed the trainer's finite,
optimizer, policy/sample counter, action-bound, active-reward-contract, entropy-zero, and
zero-hot-transfer checks. Mean throughput across the full continuation was
`1,917,781.57` agent decisions/second. Maximum
approximate KL was `0.255075` at update
`4831`; maximum clip fraction was
`0.351971` at update
`1251`.

The final full resumable checkpoint is `checkpoints/rival2/overnight/rival2_overnight_final_6h_resume.pt`. It is
`48,720,678` bytes with SHA-256
`4DC158DC2A9D16B79FB5FE7D868E3B50928AB113B55DFCC753F3734F8D87372E`, binds to `RIVAL2_REWARD_V1`, retains the
authorized curriculum-transition record, and passed exact reload.

No preflight, smoke, parity/regression, pytest, Ruff, compileall, viewer, or v0.6 work was run.
The overnight curriculum is closed at this final six-hour boundary.
