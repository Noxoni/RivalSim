# Unchanged-temperature follow-up: +600 to +650

The [completed600 review](../EVALUATION_000600.md), published in
`359b322ba4952089d1d777ca322e59bb271e488a`, supports one further50-update block:
65/144 goals versus62/163 at550, seven matched cases better, still0/10wins.
Finishing6/64 and ongoing-ground3/29 remain poor. No SSL/deployment claim.

Resume exact `checkpoints/rival2/direct_skills_v1/plus_000600.pt`, SHA256
`8AC192B4154223DB7CE82B41C77DC475D63B6AFB260EF0B9B9AC4455F6C6F6B2`.
No checkpoint selection or fresh optimizer. Source model,Adam groups/moments/
counters andfour RNG streams are retained; fresh episodes/zero hidden are the
existing declared resume behavior. Do not interpret the first fresh601 rollout
as a steady-state gain against600.

This is a lifecycle/authority extension, not a new exploration intervention.
Training logits remain divided by2 in both collection and PPO likelihoods.
Same `.001` entropy coefficient, actor1e-4/critic3e-4,90decisions,2epochs,
32768worlds,30Hzpolicy/120Hzphysics. Rewards, scenario bank, opponents, physics,
action/observation contract, model architecture and inference unchanged.
The source-file comparison is small; the collector,PPO,loader and evaluation
body are tested against the prior implementation. In review an incidental
mechanical replacement of the evaluation overtime chunk was caught and restored
before package freeze or training; exact evaluation-body parity now has a test.

The1024-world no-step preflight passes15checks: exact model/Adam/source identity,
unchanged deterministic argmax/value/hidden, critic isolation, finite gradients,
native controls and one-third Nexto learner decisions. No optimizer step was
taken. Same visited-state entropy T1=.1816344,T2=.5290186 is not capability proof.

Publish the authority/package/runtime/tests/preflight and verify remote bytes
before launch. Old20root and3T2 runtime identities stay immutable. The checkpoint
binds the new extension and also preserves the previous amendment identity.

```powershell
.venv\Scripts\python.exe benchmarks/run_direct_skills_exploration_followup_v1.py verify
.venv\Scripts\python.exe -u benchmarks/run_direct_skills_exploration_followup_v1.py run --resume checkpoints/rival2/direct_skills_v1/plus_000600.pt --resume-sha256 8AC192B4154223DB7CE82B41C77DC475D63B6AFB260EF0B9B9AC4455F6C6F6B2
```

The exact same exclusive GPU lease is held. Every accepted update is atomically
saved. On an operational recovery use this new entry point and **explicit latest**
checkpoint/SHA, never the startup command if training has advanced. It rejects
stale identities, unrelated authorities,corruption,and allSTOP files. Do not
fall back to either older runner. Only the known agent600review marker may be
archived after publication; a user/unknown stop must not be cleared.

At650: same64cases/family and ten regulation Nexto matches, save permanent650,
then intentional review stop. Compare600/550 and preserve all evidence. No KL
rejection or preservation objective; finite/corruption guards unchanged. A small
block remains a development experiment, not a promise to beat Nexto.

Monitoring uses the existing task, not a duplicate. The OpenAI Docs workflow
informed the update and preservation of the user's new-results-only reporting
preference; see [official scheduled-task guidance](https://learn.chatgpt.com/docs/automations).
