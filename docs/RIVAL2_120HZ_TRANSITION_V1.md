# Rival 2.0 120 Hz control-contract transition V1

## Outcome and boundary

The no-learning cadence transition is complete and validated. The new line is:

```text
120 Hz physics / 120 Hz Rival policy / one-tick action hold
```

No PPO optimizer update, reward training, behavior cloning, or model fitting was performed. The
source and bootstrap checkpoint hashes were unchanged by the rollout-only validation. Gameplay V1,
V2, and V3 and their evidence remain immutable historical contracts.

## Source and bootstrap

The exact parent was verified before migration:

| Item | Value |
| --- | --- |
| Source | `G:\dev\RivalSim-runs\opponent-curriculum-v1-safe-20260827-b2af03d\checkpoints\rival2_opponent_curriculum_plus_120_resume.pt` |
| SHA-256 | `3B994E118A9498713DC6115D38F061958A900EA8F4D00CE568F916942E851D9A` |
| Iteration / policy | `479 / 479` |
| Historical 30 Hz agent decisions | `3,655,854,038` |
| Bootstrap | `checkpoints/rival2/120hz_bootstrap/rival2_120hz_from_iteration_479.pt` |
| Bootstrap SHA-256 | `ADAF8D015C340CAFAE857B7253FBBDE3A6C842C4EA0BB091B31F8B1C210ED350` |

The source and migrated model tensor digest is identically
`1AA50DC45E9E0FDD0B24510A26781787742BBE8C8ED5FF6B77FD72BEC3EFA8C3`.
The source and migrated optimizer-state digest is identically
`46C5D80514908BF9498B455A93E195977234F84A8F0EE39811C921B463AF5FA0`.

All parameter-owned Adam moments and the shared step counter (`103,536`) were preserved. The split
policy/shared-trunk and critic parameter groups, update-local `1e-4` policy LR reset, `3e-4` critic
LR, adaptive KL/backoff/rollback configuration, RNG state, and historical policy tensors remain in
the checkpoint.

Intentionally reinitialized at the fresh simulator boundary:

- the 32,768-world physical simulator and episode state;
- current per-world family, side, and historical-snapshot assignments;
- Nexto temporal/kickoff state;
- Wisp action-delay/ETA state (the pinned observation-generator state is preserved into the
  transition and then consumed normally by fresh assignments);
- legacy-Rival action-hold caches;
- 120 Hz decision/physical-exposure counters;
- the retention corpus, because the V1 corpus has incompatible temporal semantics.

The source 30 Hz count is provenance only. The bootstrap begins with zero 120 Hz decisions and
separate physical-tick/world-second/agent-second counters. Cross-cadence decision counts are marked
non-comparable.

## Frozen V2 contracts

| Contract | SHA-256 |
| --- | --- |
| `RIVAL2_ACTION_V2_120HZ` | `5E3747CCF9F59BA18D81D07014D60637F7D886907A0F44B0CA681C74F20EF91A` |
| `RIVAL2_OBS_V2_120HZ` | `BF9E141E5A1E5D2F15581C8BBB10F31F11FC5AA6736B327E61C03DD8D2388237` |
| `RIVAL2_REWARD_GAMEPLAY_120_V1` | `0D4C9A78803BBAF851AB4FDD7D9AC4196AB08E42B51DC0A173A1EAEC066AFAED` |
| `RIVAL2_PPO_120HZ_V1` identity | `F5DF4C30EE80BC39E52CFCB4E2813E03D8D26C795A9FC18BC62DD2140D9C9F8A` |
| serialized PPO config | `02A4FBC09D79AD3BA677C3D2F942FE4719DA621EE87E4417D28318BFECA87F93` |

The 182 observation fields, order, team canonicalization, and normalization are unchanged.
`previous_action.*` now means the immediately preceding 120 Hz action. Lifecycle touch, demo, and
kickoff/reset flags are one-tick observations rather than four-tick aggregates.

The actor remains five tanh-Normal analog channels plus three Bernoulli channels. The active V2
environment evaluates Rival every tick and advances exactly one physics tick. The only four-tick
cache reachable from the 120 Hz trainer is explicitly scoped to historical 30 Hz Rival opponents.

## Clean gameplay reward

Event rewards retain their physical-event magnitude: goal `10`, demolition `0.1`, small pad
`0.01`, large pad `0.03`, save `0.25`, and unnecessary flip-through contact `-0.01`. Ordinary
touch remains `0`. Progress retains coefficient `0.5` and is computed from actual one-tick ball
displacement.

Dense time-occupancy coefficients are divided by four:

| Component | Historical 30 Hz | New 120 Hz |
| --- | ---: | ---: |
| Speed | `0.00010` | `0.000025` |
| Supersonic | `0.00020` | `0.00005` |
| Physical boost-use occupancy | `0.00005` | `0.0000125` |

Named mechanics, recognized-mechanic exemptions, controlled-flick classification, and all positive
mechanics rewards are absent from the active runtime. The only bad-flip exemptions are, in order:

1. physically contested 50/challenge;
2. dodge-powered physical power contact.

There is no generic jump or flip penalty. Historical Gameplay V3 detector code remains available
only for audit of prior evidence and is not allocated or launched by the V2 environment.

## PPO physical-time identity

The initial configuration is 32,768 worlds and a 128-decision horizon:

- gamma `0.9987476493904754`, with `gamma^4 = 0.995` within tight float tolerance;
- GAE lambda `0.9872585449014338`, with `lambda^4 = 0.95`;
- `(gamma * lambda)^4 = 0.995 * 0.95`;
- physical span `128 / 120 = 1.066666...` seconds per trajectory;
- clip `0.2`, value coefficient `0.5`, entropy coefficient `0`, max gradient norm `0.5`, two
  epochs, and minibatch size `65,536`.

Mixed-PPO family-local advantage normalization, critic isolation, split optimizer groups,
transactional retry, adaptive LR backoff, retention KL, soft minibatch KL `0.02`, hard minibatch KL
`0.10`, and hard completed-update mean KL `0.05` are retained. They were not exercised by an
optimizer update in this task.

## Retention corpus

`RIVAL2_RETENTION_120HZ_FROM_ITERATION_479_V1` contains 512 fixed float32 V2 observations selected
from 4,194,304 trainable observations generated by the unmodified iteration-479 actor at 120 Hz.
The final collection was 1,024 worlds by 2,048 one-tick decisions, all five kickoff layouts, with no
optimizer step. It covers near-ball interaction, possession approach, recovery, airborne, ordinary
ground play, all three X/Y field regions, and all eight heading octants.

- tensor-content SHA-256: `BED4DD26268A667251B944844544306734E0E44E54AB31689B4DE782FC0965FA`
- artifact SHA-256: `C4957B06847E7F61B5DC313ABAC58CD2FE8AB696561C7A4898C6ACFF219DACDC`

The initial two-second candidate correctly failed closed on longitudinal/orientation diversity and
was not accepted. The longer bounded collection replaced it before target validation.

## Deterministic and CUDA validation

The focused suite proves:

- equal held actions produce matching car, ball, wheel, boost, and lifecycle state after four
  physical ticks (`atol <= 2e-6`);
- one 30 Hz progress/boost-occupancy interval matches the sum of four 120 Hz ticks
  (`atol <= 2e-7`);
- goal reward/termination/reset occurs on its single physical tick;
- V2 previous action advances every tick;
- a legacy Rival snapshot is evaluated once and held for four ticks;
- Nexto and Wisp emit matching four-tick action sequences and physical state under the old outer
  step and four new one-tick steps.

The required single target rollout was 32,768 worlds by 128 decisions:

| Measurement | Result |
| --- | ---: |
| Logical rollout buffer | `6,970,933,248` bytes |
| Environment logical state | `420,741,120` bytes |
| Peak allocated VRAM | `8,462,390,784` bytes |
| Peak reserved VRAM | `8,956,936,192` bytes |
| Allocated headroom at peak | `25,728,068,096` bytes |
| Reserved headroom at peak | `25,233,522,688` bytes |
| Rollout wall time | `6.3213299001 s` |
| Rival/historical inference | `1.0556156216 s` |
| Physics/reward | `0.1877666566 s` |
| Other adapter/sampling/observation | `5.0779476219 s` |
| World physics ticks/s | `663,516.07` |
| Trainable agent decisions/s | `796,510.87` |

All observations, actions, and rewards were finite. Consecutive-tick action changes occurred in
`0.9620985985` of adjacent world/agent pairs. Simulator and Nexto timed H2D/D2H counters were zero.
Pinned Wisp retains its accepted source-specific Windows CPU scalar ETA cache; that adapter was not
changed or falsely relabeled as a GPU-only implementation.

No OOM occurred. With the complete rollout resident there was ample memory for future PPO work,
but no claim is made that an optimizer update was validated here.

## Opponents, consumers, and demonstrations

All 13 preserved historical Rival snapshots are tagged `30 Hz / RIVAL2_ACTION_V1`; the active
mixed trainer evaluates each on phase zero and holds its action for four physical ticks. Future V2
snapshots carry 120 Hz metadata and evaluate every tick. Nexto and Wisp remain driven once per
physics tick through their unchanged adapters.

RivalVis reads checkpoint action/observation/cadence metadata. The smoke loaded the bootstrap and
advanced two policy decisions in exactly two physics ticks with finite controls and an unchanged
checkpoint hash. Legacy evaluation modules (`full_match.py`, `open_play.py`, short evaluators,
Campaign scripts, and RocketSim crosschecks) retain explicit 30 Hz constants and therefore remain
historical V1 consumers; they are not silently reinterpreted as V2.

Human demonstration tooling now exposes `action-alignment`: native Rocket League frame N maps to
one V2 eight-channel target at decision N with no averaging, subsampling, or four-frame combination.
Observation quality classifications remain separate and unavailable telemetry is not invented.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_rival2_120hz.py
.\.venv\Scripts\python.exe benchmarks\run_rival2_120hz_transition.py
.\.venv\Scripts\python.exe -m rivalsim.human_demo action-alignment
```

Machine-readable evidence is under `results/rival2/120hz_transition_v1/`. The bootstrap is ready
for human demonstration collection and later adapter development. PPO training and behavior
cloning remain explicitly unauthorized at this boundary.
