#650 can finish accessible shots; set-keeper pressure is the main gap here

All192 fixed native episodes completed under the prospective package published
at `226e9aaada02b83e415fbbb44572bb5006eb3f29`. All six package files were remotely
verified before execution. The established-keeper64 per-case results reproduce
the published650 baseline exactly. No optimizer, checkpoint, model, Nexto,
reward, physics or deployment change occurred.

| Initial defender state | Goals | Concedes | Timeouts | Goals before any opponent contact |
|---|---:|---:|---:|---:|
| Established keeper |5|41|18|5|
| Recovering defender |50|9|5|46|
| Initially open net, active Nexto far behind |49|7|8|49|

Rival takes the **first native contact in all64 cases in every mode**. Mean time
to that contact is0.538/0.531/0.547seconds respectively. Failure to reach these
nearby balls is not the dominant issue. In the set-keeper condition Nexto touches
the ball within one second after Rival in49/64 cases, versus0 in the other two.
It eventually touches59/64 versus18/64 and15/64. The median gap from Rival's first
touch to Nexto's first touch is0.708/2.379/4.833seconds among cases with Nexto contact.

At the first contact interval,50 established-keeper balls satisfy the existing
goal-projection proxy, but the policy scores only5 actual goals. Rival boosts on
61/64 first-contact intervals and jumps on23/64. This is not simply refusal to
boost, a missing touch, or an absent goal reward. On all49 initially open-net
goals, the ball scores before Nexto makes any contact. Four of the50 recovering
defender goals occur after an opponent contact, showing some follow-through,
not proving reliable possession or universal recovery skill.

The intervention changes opponent state in the observation, so Rival is allowed
to react differently. The first chosen action differs from the original in16/64
recovering starts and43/64 initially open-net starts. These are not identical
ball trajectories replayed into three defenders. Do not interpret the counts as
keeper save intent or ignore the remaining15/14 failed accessible finishes.

## Evidence

All three JSON episode reports and compressed30Hz pre/action/post182-observation
traces are retained, including alive masks, native contact/subtick/goal timing.
`report.json` verifies trace hashes,24 reduction checks and checkpoint identity.
Native execution additionally checked unchanged Nexto and model, baseline replay,
finiteness and complete endings. Seven prospective CPU tests passed. The reducer
rebuilds exactly. Its velocity median fields use Torch's lower median for an
even64-case sample; goal/timing summaries use the explicit stored reductions.

No simulation failed or needed rerunning. The original checkpoint stays at
SHA256 `4FCD41C2C8305B0EFF448ED2E78ECAD4BDC447D87C740EAE79EEBBD4621EF09F`.

## Concrete next action

Do not spend the next block mainly teaching already-successful open-net contact,
and do not equate projected shot count with scoring. Make one bounded
**curriculum-only** change: keep half of finishing starts exactly as the original
set-keeper cases, and replace the other half with a continuous intermediate
recovering-to-set-keeper position distribution. This changes about10% of all
source starts; the other90%, full-match reference, rewards, opponents, model,
Adam state, PPO and temperature remain unchanged. No easy open-net-only majority.

The intermediate defender remains active, moves coherently toward its own goal,
and starts at a prospectively fixed interpolation range between the verified
recovering location and its original keeper location. Validate geometry before
training. Freeze source identities and run only +650 to +675 before the next
unchanged five-family skill and ten-regulation-match review. Preserve every
accepted rolling checkpoint. Success must be improved actual finishing and
match outcomes, not the easier curriculum's reward counts. This is a test of
graduated pressure, not a guaranteed fix or an SSL capability claim.

```powershell
.venv\Scripts\python.exe benchmarks/report_direct_skills_shooting_000650.py
```
