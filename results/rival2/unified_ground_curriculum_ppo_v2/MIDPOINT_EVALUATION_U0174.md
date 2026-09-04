# Unified Ground Curriculum PPO V2 Midpoint Evaluation

Training was paused after accepted update 174. The rolling checkpoint was copied
byte-for-byte to
`G:/dev/RivalSim-runs/unified-ground-curriculum-ppo-v2/snapshots/ground_curriculum_midpoint_u0174.pt`
with SHA-256
`95DA0ACA2832ECE460F40872FD2474F908B0D373472D3CD351174E1C7D5C0B15`.
The rolling and preserved files still had that hash after evaluation. No optimizer
step or policy mutation occurred during evaluation.

## Verdict

`BLOCKED_NATURAL_NEXTO_REGRESSION`

The checkpoint retained the controlled aerial and offensive-demo behaviors, but
its deterministic natural-play result against the same pinned Nexto evaluation
collapsed. Training remains paused and must not resume from update 174 without a
prospective correction to the opponent/evaluation strategy.

## Frozen comparison

Both rows use 256 deterministic episodes, 120 Hz Rival inference, the same five
kickoff layouts, the same evaluation seed, the same pinned Nexto adapter, and the
same episode termination rules.

| Metric | Unified V5 parent | PPO update 174 |
|---|---:|---:|
| Rival goals | 77 | 0 |
| Nexto goals | 154 | 256 |
| hard-timeout ties | 25 | 0 |
| episodes with a Rival touch | 256 | 256 |
| Rival touches | 1,289 | 692 |
| Nexto touches | 2,041 | 2,810 |
| Rival forward contacts | 1,179 | 640 |
| Rival mean speed, uu/s | 1,176.351 | 1,225.517 |
| no-touch truncations | 0 | 0 |

The five initial deterministic kickoff actions remain close to the parent. The
failure therefore begins after the opening recurrent trajectory diverges; it is
not explained by a gross initial kickoff-action change or an evaluator contract
change. The model is faster but obtains 46.3% fewer touches, produces 45.7% fewer
forward contacts, scores no goals, and concedes in every episode.

## Controlled capability retention

The controlled aerial gate still passes. Selected parent to update-174 fractions
are:

| Metric | Unified V5 parent | PPO update 174 |
|---|---:|---:|
| elevated follow touch | 0.482910 | 0.492188 |
| high follow touch | 0.260742 | 0.255859 |
| second airborne touch | 0.042480 | 0.048340 |
| productive continuation | 0.153320 | 0.130859 |
| sustained control | 0.125488 | 0.097656 |
| goal within contact budget | 0.021484 | 0.023926 |

Offensive-demo behavior also remains: both checkpoints recorded 421 actual demos.
The parent recorded 394 post-demo touches and 305 post-demo goals; update 174
recorded 393 and 310. Recovery telemetry regressed from 90 to 65 productive floor
landings, 201 to 155 productive landing chains, and 136 to 120 productive wall
landings.

## Interpretation

Pure symmetric current-policy self-play supplied no fixed behavioral anchor.
Rollout telemetry showed more goalward contacts and many aggregate self-play goals,
but that co-adapted signal did not transfer to a fixed opponent. The midpoint
evaluation therefore rejects the apparent self-play improvement as sufficient
evidence of stronger play. The complete machine-readable record is
`midpoint_evaluation_u0174.json`.
