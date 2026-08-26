# Rival 2.0 Campaign 04 Results

Campaign 04 completed the exact long-run continuation of the Campaign 03 Reward V2 policy.
It loaded checkpoint SHA-256 `A0F2E554448B31A373BD73254125AC0ADFDB541EE3B695AD9D040B2CCFA68991` with optimizer, Torch and
CUDA RNGs, policy/opponent generator states, counters, opponent assignments, and historical
policies intact. Training resumed at update 12 / 100,663,296 cumulative samples and stopped at
update 120 / 1,006,632,960 samples. Update 121 did not run.

No preflight, reward smoke, baseline rerun, world-count sweep, inherited parity/regression suite,
post-run test/lint/compile ceremony, viewer work, or v0.6 work was performed.

## Training integrity

All 108 continuation updates passed finite-state, optimizer,
policy-version, sample-count, action-bound, historical-policy, Reward V2 identity, and zero
tracked hot-path transfer checks. The final checkpoint passed exact reload and continuation.

- maximum approximate KL: `0.010950` at update
  `13`;
- maximum clip fraction: `0.130710` at update
  `13`;
- maximum gradient norm: `0.777115`;
- mean continuation throughput: `1,940,848.52` agent
  decisions/second;
- total continuation wall time including the four evaluations:
  `587.040` seconds.

## Authorized behavioral curve

Each new point is exactly one 4,096-world ordinary stochastic self-play evaluation at seed
`920260826`. The 100M Campaign 03 point is reused from its published evidence and was not rerun.

| Point | Update | Cumulative samples | Touches/min | Goals/min | Goal fraction | No-touch fraction | Mean duration, s |
|---|---:|---:|---:|---:|---:|---:|---:|
| 100m | 12 | 100,663,296 | 1.308672 | 0.243800 | 0.063721 | 0.936279 | 15.681901 |
| 250m | 30 | 251,658,240 | 3.202896 | 0.477765 | 0.131104 | 0.867676 | 16.464591 |
| 500m | 60 | 503,316,480 | 6.453265 | 0.324010 | 0.105713 | 0.869873 | 19.575846 |
| 750m | 90 | 754,974,720 | 8.712013 | 0.426426 | 0.159424 | 0.770752 | 22.431649 |
| 1b | 120 | 1,006,632,960 | 16.661451 | 0.311649 | 0.152100 | 0.550293 | 29.282894 |

The prospectively frozen two-axis classification is
**`CONTINUING`**: from 750M to 1B, touches/minute increased from
`8.712013` to
`16.661451`, while no-touch truncation fell from
`0.770752` to
`0.550293`. Relative to the 100M baseline, the final policy
increased touches/minute by
`+15.352779` and
reduced no-touch truncation by
`0.385986` absolute.

Secondary goal metrics were not monotonic: the 1B goal rate was
`0.311649`, down from the 750M value
`0.426426` even while the two frozen approach-learning axes
continued improving. No setting was changed in response.

## Final checkpoint

The exact final resumable checkpoint is
`checkpoints/rival2/campaign04/rival2_campaign04_1b_resume.pt`. It is
`31,159,541` bytes with SHA-256 `DB5AA09B2CAD40D4C1F5DB1014FDE245C58994A6948458212751724F782BE6B0` and contains
policy/update version 120, 1,006,632,960 cumulative samples, optimizer/RNG/assignment state, and
historical policy versions `[0, 2, 3, 6, 12, 30, 60, 90, 120]`.

Campaign 04 is closed at the 1B boundary. The result does not itself authorize a viewer,
additional training, reward/PPO changes, or v0.6 RocketSim/RLBot transfer work.
