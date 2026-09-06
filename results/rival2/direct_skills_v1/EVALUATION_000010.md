# Direct Skills +10: mixed early result, no clear gameplay transfer

## Identity and integrity

- Exact checkpoint: checkpoints/rival2/direct_skills_v1/plus_000010.pt
- SHA256: B67880AD02D2F7FE5C34587C6316969D992C922797F81B4A455A392ED519FAFB
- Ten accepted updates, 44,236,800 new learner decisions, 117,964,800 physical
  world ticks, 1,418 new Adam steps (53,568 cumulative).
- Completed-update mean KL range: 0.00281009781938787 to 0.0049207041530241295.
  No KL-based rejection; KL is telemetry only. Model and Adam are finite.
- Parent, observation/action/PPO contracts, exact one-third Nexto learner
  exposure and counters passed the CPU checkpoint audit. Semantic reward
  authority is unchanged. The real worker resumed training automatically.

Evaluation finished 2026-09-06T02:18:26Z. No optimizer step occurred during
evaluation. All five families have the same 64 cases and source hashes as the
offset0 baseline. Raw outcomes reconcile with summaries. The CPU report rebuild
was deterministic; it did not rerun the policy. Raw reports and immutable
training-curve prefix are committed alongside this interpretation.

## Fixed-case comparison

These are **initial episode outcomes against Nexto, not full-match wins**.
Every family contains 64 cases. Reported control/shot-clear events are frozen
task proxies, not independent proof of intent or universal possession/save logic.

| Family | Goals for, 0 to +10 | Goals against | Cases touched | Task outcome |
|---|---:|---:|---:|---|
| Natural starts | 2 to 2 | 62 to 62 | 23 to 25 | No scoring improvement |
| Challenge | 0 to 0 | 53 to 54 | 51 to 52 | Controlled gains 3 to 3; controlled forward advances 2 to 2 |
| Finishing | 3 to 5 | 55 to 53 | 64 to 64 | Goal-bound contact events 52 to 53 |
| Defense | 2 to 1 | 36 to 41 | 59 to 61 | Shot-clear events 32 to 30 |
| Kickoff | 0 to 0 | 52 to 46 | 25 to 25 | Controlled gains remain 0; timeouts 12 to 18 |

The five-family contact total is not an appropriate broad capability score.
More defense touches did not mean fewer concessions. Likewise, the reduction
in kickoff concessions accompanies six additional timeouts, not an increase
in acquisition or scoring. It does not establish a learned kickoff win/fake.

For the same individual cases, goal outcome (+1 score / 0 timeout / -1 concede)
improved/worsened in 7/8 challenge cases, 7/5 finishing cases, 10/16 defense
cases, 13/7 kickoff cases and 1/1 natural cases. This is descriptive paired
evidence, not a significance claim or a new acceptance threshold.

## Decision

There is a small finishing gain, a defensive regression and no demonstrated
possession gain. The objective of robust ground gameplay remains unfulfilled.
This is not SSL capability or a promotion recommendation.

Continue the already-frozen campaign through the scheduled +25 skill check and
the +50 skill/full-match review. Do not select a different checkpoint, alter
rewards/exploration or add detectors from this first small comparison. If the
first50 review still lacks task and natural-play progress, the existing authority
requires diagnosis before another large unchanged block. Preserve +10 so later
behavior can be compared or rolled back precisely.

The ten-regulation-match baseline remains offset0: 0/10 wins, 1 goal for,
266 against. **No new full-match evaluation was run at +10**; it is scheduled
at +50. Do not present the natural-start subset as that match comparison.

## Read-only reproduction

From the repository root:

```powershell
.venv/Scripts/python.exe benchmarks/audit_rival2_direct_skills.py --update 10
.venv/Scripts/python.exe benchmarks/report_rival2_direct_skills.py --update 10
```

These commands read saved artifacts; they never launch training or evaluation.
