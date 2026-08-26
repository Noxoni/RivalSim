# Rival 2.0 Campaign 01 Results

Campaign 01 completed its exact bounded execution. The fresh policy finished update
`12` with `100,663,296` agent decision
samples, which is the first completed PPO update at or above 100,000,000 samples. No additional
training update and no v0.6 work was performed.

## Verdict

- execution status: `COMPLETE`;
- behavioral result: `DEGRADED`;
- behavioral rationale: Final self-play touch rate fell from 0.272091 to 0.175624 per simulated minute; stochastic play against initialization lost 7-23 with a -46 touch differential; deterministic play lost 0-819; and analog policy standard deviations rose from about 1.0 to about 2.65 near the frozen ceiling.
- frozen v0.5 trainer verdict: unchanged (`PASS_GREEN`);
- final checkpoint reload and exact next stochastic sample:
  `PASS_GREEN`.

The behavioral verdict is descriptive and independent of execution acceptance. Campaign 01 did
not change a reward, observation, action, episode, architecture, curriculum, or PPO setting in
response to the measured outcome.

## Frozen authority

- authorized campaign commit: `4235963a0648d7148b93f073311bb3343dd68ac4`;
- campaign seed: `20260826`;
- evaluation seed: `920260826`;
- evaluation protocol SHA-256: `964D7281C9BB8EF12C4A831B984015259A777D82285A02EAB329FBB6CC098CE7`;
- selected worlds: `131,072`;
- horizon: `32`;
- entropy coefficient: `0.01`;
- policy configuration SHA-256: `58C7409F34EA24CB7FAE7505A7F5FE2CC1B65021EE48B5200ED12BB8990C6136`.

All four frozen contract identities are recorded in `config.json` and match v0.5.

## Capacity preflight

The candidates were attempted once in the authorized order and the first fully passing capacity
was selected:

- 131,072 worlds: `PASS`, peak 29,675,270,144 bytes, margin 4,515,647,488 bytes

This was a real horizon-32 rollout/GAE/PPO update with finite-state checks, checkpoint/inference
allocation, zero simulator hot-path H2D/D2H traffic, and a prospectively frozen 4 GiB safety
margin.

## Checkpoint custody

Initialization and every first threshold-crossing checkpoint remain in the ignored local campaign
artifact directory with exact SHA-256 identities:

- `000m`: 0 samples, update 0, `5629BD2D55E90B12910A58254B85F8C8181E7B1AF15A64FAE20375F357A80615`, 6,074,626 bytes
- `010m`: 16,777,216 samples, update 2, `AD7C328E833D7D8C64EC23131E54CFB9091A1C5FF7A94180A8485D16356076AA`, 13,601,402 bytes
- `025m`: 25,165,824 samples, update 3, `ED1283E4B9EE2B41D6A93FB180121DFA0B51343E41FFA14A0B1630D881B1E3CE`, 16,109,752 bytes
- `050m`: 50,331,648 samples, update 6, `099ED7CABE7FD38D270901161F1AE5D55DB8CF8A08B1754DC6B1CC05ACFFD138`, 18,618,038 bytes
- `100m`: 100,663,296 samples, update 12, `704F2B887BF50E767C86B7080C1E881644480D41A3302D245E833BDE65752B4A`, 21,126,324 bytes

The final resume checkpoint is also published at
`checkpoints/rival2/campaign01/rival2_campaign01_100m_resume.pt`; its committed artifact is
`21,126,324` bytes with SHA-256
`704F2B887BF50E767C86B7080C1E881644480D41A3302D245E833BDE65752B4A`. It is the exact v0.5-format full resume artifact, not
an inference-only substitute.

## Fixed evaluation

Each checkpoint used the same 4,096 held-out worlds, all five kickoff layouts, seeds, balanced
sides, and first-episode limit. Rates below are from stochastic ordinary self-play. The final four
columns are current-checkpoint minus frozen-initialization results.

| checkpoint | samples | goal fraction | no-touch fraction | hard fraction | touches/min | goals/min | det goal diff | det touch diff | stochastic goal diff | stochastic touch diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 000m | 0 | 0.006348 | 0.993652 | 0.000000 | 0.272091 | 0.024998 | 0 | 0 | -2 | 15 |
| 010m | 16,777,216 | 0.002686 | 0.997314 | 0.000000 | 0.222450 | 0.010593 | -819 | -819 | -10 | -35 |
| 025m | 25,165,824 | 0.005127 | 0.994873 | 0.000000 | 0.203673 | 0.020271 | 0 | -819 | -16 | -37 |
| 050m | 50,331,648 | 0.003418 | 0.996582 | 0.000000 | 0.209253 | 0.013438 | 0 | -819 | -15 | -30 |
| 100m | 100,663,296 | 0.003418 | 0.996582 | 0.000000 | 0.175624 | 0.013510 | -819 | -819 | -16 | -46 |

At the final checkpoint, ordinary stochastic self-play recorded
`182` accepted touch entries,
`14` goal terminations, and
`0` demolition events. Its mean first-episode duration was
`15.180216` seconds. The complete analog magnitudes,
button activation rates, policy standard deviations, Bernoulli probabilities/entropies,
termination mix, and outcome counts are in the five `evaluation_*.json` artifacts.

Final deterministic play against initialization produced goal differential
`-819`, touch differential `-819`, and
outcomes `0` current wins / `819`
initialization wins / `3277` draws. Final stochastic play produced goal
differential `-16`, touch differential
`-46`, and outcomes `7` /
`23` / `4066`.

## Integrity and regression

Every one of the `12` completed campaign updates passed finite checks
for observations, actions, rewards, values, log probabilities, advantages, returns, losses,
gradients, parameters, and optimizer state. Analog actions remained bounded, buttons remained
binary, selective done/reset accounting remained valid, historical snapshots remained frozen and
gradient-free, sample/version accounting remained exact, and ordinary simulator hot-path transfer
counters remained zero.

The final v0.5 checkpoint loader reproduced model/value outputs, deterministic inference, and the
next stochastic action/pre-tanh/log-probability exactly. Tracked `results/v0.1/` through
`results/v0.5/` matched the prospectively recorded byte manifest at closeout. Campaign tooling did
not modify simulator or trainer implementation behavior, so no expensive prior authority corpus
was rerun.

## Boundary

Campaign 01 is closed at the first completed update crossing 100M samples plus its fixed
evaluation and evidence closeout. This report does not authorize or begin v0.6.
