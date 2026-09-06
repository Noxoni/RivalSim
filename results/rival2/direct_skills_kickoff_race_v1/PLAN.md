# Bounded kickoff race and follow-through learning arm

## Why this experiment

The completed corrected-controller block did not show sustained match improvement.
The read-only kickoff trace at `2e104a20427bcab08a5129bd01cce9947ac05f41` showed
full throttle/boost, reasonable ball alignment, but about11% slower opening
speed than Nexto, no early native flips, and no first-contact wins. The +25
opening action sequence differed at only12/670 decisions. This is an approach
deficit, not evidence that the optimizer never updated or that Rival is inactive.

## Exactly what changes

Kickoff role4 only: replace each player's personal-first-touch payment with a
single world-first-contact race payment of0.5. Resolve actual physics ticks;
bilateral first contact on the same physics tick is a tie and pays neither.
The opposite player touching later, even within the same four-tick decision,
does not earn another race payment. All contacts remain in raw touch telemetry.
The existing `first_touch` event column denotes this race event only for role4.

Keep control gain1.5, controlled advancement0.75, native goals+10/-10 and all
other existing task outcomes. A good second-touch control outcome remains more
valuable than simply touching first. No reward for jumping, flipping, boosting,
speed, prescribed inputs or a named mechanic. No fake-kickoff detector.

Keep the exact original positions in every kickoff. In half of dedicated
kickoff exercises, initialize focal-car velocity along its existing heading at
250–750uu/s; everything else is unchanged. The other half retain standing starts.
Moving Rival closer would disable Nexto's nearest-car kickoff admission, so no
position handicap is used. Momentum-assisted exercises are not represented as
genuine standing kickoffs. Rival chooses every action; no scripted prefix.

Shooting remains20% of source starts. Natural30%, challenge20%, defense15% and
kickoff15% remain unchanged. The original natural lane also includes standing
kickoffs. All non-kickoff states, shooting-pressure variants, and other reward
roles are byte-/tensor-exact. No task ID, router, physics or architecture change.

## Training and evaluation

Root: preserved parent650
`checkpoints/rival2/direct_skills_finishing_goal_v2/child_000025.pt`, SHA
`939213BF57FFC751C7C173FFCC8AC34DE84D2CAA66CFECB79C59792CBABB0F6D`.
Restore exact model, Adam/groups/counters and four RNG states. New physical
episodes/zero recurrent hidden and separate native-controller RNG are explicit.
This is not a resume of the regressed child25.

32768worlds,120Hz physics/30Hz decisions,four-tick action hold,90-decision
rollout,two PPO passes,actorLR1e-4/criticLR3e-4,T2 collection and likelihood.
Half worlds evolving self-play, half native-v5 Nexto; one third learner samples
face inference-only Nexto. KL telemetry only. Finite model/gradient/Adam and
transactional corruption safety unchanged.

Exactly15 accepted updates maximum,66,355,200 learner decisions. Permanent
entry0/first1/+5/+15, alternating durable rolling files every accepted update.
At+5 and+15 run the original ten full corrected-controller matches from real
standing kickoffs. No momentum assistance in evaluation. Compare parent650's
already completed matching baseline:0/10wins,7–196goals,456contacts,zero
kickoff first contacts. No historical legacy-Nexto scores as learning evidence.
Judge first contacts, subsequent contact/control outcomes and scoring separately.
No automatic extension of this finite runner, and no SSL/promotion claim merely
from task reward or passing tensor checks.

## Evidence and operations

`training_authority.json` and `training_package.json` bind all settings, sources,
parent, diagnosis and tests. `scenario_audit.json` binds the full32768-row bank.
Thirty focused tests passed, including native goal/reset boundaries and exact
non-kickoff physics/observation/reward parity. Initial test setup caught a
Torch-device-to-Warp identifier mismatch; it was fixed before preflight, with
the original failed test artifact retained. The1024-world/90-decision preflight
made zero optimizer steps; model/Adam/parent stayed unchanged and gradients were
finite with critic isolation. The sampled assisted curriculum yielded27 paid
kickoff races against Nexto; this proves reachable signal, not learned transfer.

The new entrypoint explicitly configures the preserved native-Nexto runner only
inside a process-local context and restores its globals on exit. Its historical
source file is unchanged. Checkpoint metadata field names retain the shared
runner's `native_nexto_*` schema but carry this new version and authority hash.

External run: `G:/dev/RivalSim-runs/direct-skills-kickoff-race-v1`.
Use the existing exclusive GPU lease. Respect user STOP and never restart a
healthy process or a completed block. Operational recovery requires preserving
latest accepted state, focused validation and published recovery evidence.

```powershell
.venv\Scripts\python.exe -B benchmarks/run_direct_skills_kickoff_race_v1.py verify
.venv\Scripts\python.exe -u -B benchmarks/run_direct_skills_kickoff_race_v1.py run
```

The SSL-development goal remains active; this is one bounded intervention.
