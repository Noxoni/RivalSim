# Rival 2.0 Campaign 03 Results

Campaign 03 is complete. It implemented `RIVAL2_REWARD_V2`, ran only the authorized
targeted GPU reward smoke, immediately trained from scratch at 131,072 worlds / horizon 32,
and stopped at update 12 / 100,663,296
agent decision samples, the first completed update crossing 100M. It did not run capacity
preflight, initialization evaluation, inherited parity/regression gates, a world-count sweep,
or intermediate held-out evaluations.

## Reward V2 custody

`RIVAL2_REWARD_V1` remains unchanged. Reward V2 adds exactly one per-agent term:

`(car_ball_distance_before - car_ball_distance_after) / 4096.0`

The true 3D distances are reconstructed on CUDA from frozen observation relative-position
fields at decision start and the final pre-reset transition state. Reward V2's deterministic
content SHA-256 is `54CD5AC582133D9BA77CF7DF7976C549B3E659920BA407C9ACCE8A9FD5F50B32`.

The one targeted smoke was `PASS_GREEN`. Its closing, opening, and unchanged cases
produced `0.244140625`, `-0.244140625`,
and `0.000000000`. The forced reset case's integrated approach
matched the pre-reset value `0.000008821487`; the
post-reset-contaminated alternative was `0.000000000000`.
All smoke tensors were finite and device-resident.

## Bounded training execution

The Campaign 02 PPO/model/observation/action/episode/self-play baseline was otherwise unchanged,
including entropy coefficient `0.0`. Checkpoints were saved at the first updates crossing 25M,
50M, and 100M. All 12 updates passed numerical and
device-transfer integrity, and the final checkpoint passed exact reload/continuation checks.

- maximum approximate KL: `0.018967` at update
  `1`;
- maximum clip fraction: `0.258459` at update
  `1`;
- maximum gradient norm: `0.493067`;
- mean training throughput: `2,085,827.10`
  agent decisions/s;
- training wall time: `50.591` seconds.

The final resumable checkpoint is
`checkpoints/rival2/campaign03/rival2_campaign03_100m_resume.pt`, with SHA-256
`A0F2E554448B31A373BD73254125AC0ADFDB541EE3B695AD9D040B2CCFA68991` and size `21,126,388` bytes.

## Single final evaluation

Exactly one 4,096-world ordinary stochastic self-play evaluation was run after the final
checkpoint, using seed `920260826`.

| Metric | Campaign 02 final | Campaign 03 final | C03 - C02 |
|---|---:|---:|---:|
| Touches / simulated minute | 0.291182 | 1.308672 | +1.017490 |
| Goals / simulated minute | 0.040362 | 0.243800 | +0.203438 |
| Goal-terminated fraction | 0.010254 | 0.063721 | +0.053467 |
| No-touch truncation fraction | 0.989746 | 0.936279 | -0.053467 |
| Mean episode duration, seconds | 15.242952 | 15.681901 | +0.438949 |

The dense approach term coincided with a large increase in touch and goal frequency and a
5.3467 percentage-point reduction in no-touch truncation on this single frozen final protocol.
This is the requested direct Campaign 02 comparison, not a claim of external Rocket League
competence or v0.6 transfer readiness.

Campaign 03 is closed. No curriculum, extra reward term, action mask, hyperparameter tuning,
v0.6 RocketSim/RLBot work, or post-boundary training was begun.
