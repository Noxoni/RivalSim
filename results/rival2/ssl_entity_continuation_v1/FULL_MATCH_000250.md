# Fixed +250 natural Nexto follow-up: gameplay regression

This is a development comparison, not an untouched test or checkpoint selection.
The fixed +250 checkpoint and ten-match method were published before execution at
`cdc72905374d48338ae820ef22b1091f192dc5f5`. The already-completed +200 baseline was
not rerun. The policy, reward, opponents, sampling and PPO settings were not tuned.

| Ten regulation games, five layouts / both sides | +200 | +250 |
|---|---:|---:|
| Wins / losses | 0 / 10 | 0 / 10 |
| Rival goals | 0 | 0 |
| Nexto goals | 243 | 317 |
| Rival contacts | 238 | 302 |
| Rival contacts per minute | 4.76 | 6.04 |
| Matches without a Rival contact | 0 | 0 |
| Rival first kickoff contacts | 0 | 0 |
| Same-player follow-up / resolved contacts | 22 / 237 | 6 / 301 |
| Forward net ball-displacement intervals | 88 | 65 |
| Backward net ball-displacement intervals | 22 | 53 |
| Concessions with at most one contact since kickoff | 13 | 63 |
| Native Rival demolition events | 7 | 0 |

Concessions increased in **all ten paired matches**: `[25,24,24,25,24,31,24,21,23,22]`
became `[30,31,35,34,33,35,32,29,28,30]`. Total concessions rose 30.45% while
contacts rose 26.89%. The same-player next-contact fraction fell from 9.28% to
1.99%. This is **not overall gameplay improvement**. It demonstrates more contact
activity alongside worse scoring resistance and contact follow-through in this
fixed development corpus. Both policies still failed to score or win a kickoff.
It does not establish a universal ranking or a causal explanation for the change.

## Counter interpretation

The full-match velocity-change counter is positive on all 302 Rival contacts at
+250. That is **change in canonical velocity**, not necessarily a ball moving
toward the opponent goal: slowing a ball moving toward our goal can also produce
a positive change. It is not evidence of 302 offensive shots or useful possession.
The short evaluation's positive-ball-velocity fraction measures a different thing
(velocity at the decision boundary) and must not be conflated with this counter.

Follow-up is the next distinct contact identity, not possession time. Net ball
displacement extends until the next contact or goal and uses the existing +/-100
uu bins. One-contact kickoff concessions are classified by contact count, not
elapsed time. Native demos do not imply intentional demo seeking. There are no
no-touch resets in this full-match method; all games had Rival contact, but this
does not prove continuous activity or absence of long idle periods.

## Integrity and resumption

`full_match_000250_integrity.json` verifies complete games, native goal entries,
scoreboard/scorer conservation, goal-linked hidden resets, no event overflow,
zero evaluation optimizer steps, immutable model/checkpoint, and unchanged saved
training checkpoint. This PASS means **recording integrity only**, not capability.
Raw evidence: `full_match_000250.json`. Exact evaluated checkpoint:

`checkpoints/rival2/ssl_entity_continuation_v1/plus_000250.pt`

SHA256 `22C3D4762784F7AC9D48DD429DFDFCA7E88333DFC2DABE2B11A9C01ED4C53878`.

The learner was briefly stopped at the next accepted boundary, **+270**, not
rolled back to the evaluated +250 snapshot. The preserved resume source has
48,182 cumulative Adam steps, path
`G:/dev/RivalSim-runs/ssl-entity-continuation-v1/paused_000270.pt`, SHA256
`0EB666C56EFCAEBA0C1686D8A6C332499678979E7A4B144F8200FB1DA6DD5DBB`.
Only the exact agent-owned evaluation STOP was removed. Training was relaunched
at 2026-09-06T00:36:20Z into `resume2.stdout.log` / `resume2.stderr.log`.
`resume_after_match_000250_integrity.json` separately records zero-step equality
of restored model, Adam, counters, contracts and policy/shuffle/CPU/CUDA RNG.
Fresh physical episodes/zero hidden and new assignments from the restored
opponent generator remain the original declared resume semantics.

Eight focused CPU report/resume-audit tests passed. These tests and report tools
do not participate in inference, reward calculation, PPO or checkpoint selection.

## Next evidence decision

Keep the existing user-authorized learning campaign active without silently
changing its authority. At the next scheduled **+300** boundary, examine the
same short scenarios **and a fixed +300 ten-game natural Nexto follow-up** using
the existing match method, compared with the already-completed +250 and +200.
Publish/bind the exact +300 artifact before evaluation; preserve the latest
accepted training checkpoint if an evaluation pause occurs after +300.

This additional natural check is chosen now, before seeing +300, because the
short scenario improvements did not predict improved full-match control. Do not
select a different checkpoint based on its scores, rerun favorable seeds, add
mechanic rewards or reinterpret KL telemetry as failure. If the loss of control
persists, investigate the training-to-natural-game transfer before authorizing
another large unchanged block; a transient touch-count gain is insufficient.
SSL gameplay remains unachieved and no deployment is promoted by this report.
