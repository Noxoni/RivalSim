# Exploration-only PPO experiment from parent650

The completed kickoff arm did not improve full matches. A no-update matched
trajectory diagnostic found effectively deterministic kickoff starts even at
temperature2. This arm adds the calibrated uniform log-barrier exploration
regularizer beta0.01 to the existing clipped PPO objective. It gives suppressed
actions a nonvanishing exploration gradient without choosing an action for Rival.

Exact parent: checkpoints/rival2/direct_skills_finishing_goal_v2/child_000025.pt
SHA939213BF57FFC751C7C173FFCC8AC34DE84D2CAA66CFECB79C59792CBABB0F6D.
Restore its exact model, Adam groups/moments/counters and four RNG states.
Not a fresh model, not the negative kickoff child15. Fresh simulator episodes,
zero recurrent hidden and new fixed-seed native-controller RNG are explicit.

No reward, scenario, architecture, action parser or inference-cadence changes.
Shooting20% of starts still pays actual goals, not mere projected shots.
Kickoff outcome/momentum-assisted curriculum is the same as the completed arm.
32768worlds,120Hz physics/30Hz policy,90decision rollout,2epochs,actorLR1e-4,
criticLR3e-4,T2 collection AND likelihood,half worlds selfplay/half native-v5
Nexto. One third of learner decisions face Nexto; masked opponent samples never
train Rival. Value loss remains isolated in the independent critic.

The only new loss is beta*KL(Uniform90||pi_T2). It is not old-policy retention,
not a KL guard and not a gameplay reward. All90 joint actions have equal
reference weight. Existing entropy0.001 remains. Deterministic evaluation is
the original raw actor argmax, so there is no router or inference wrapper.

Exactly30accepted updates, max132,710,400learnerdecisions, full matched Nexto
evaluations at+5,+15,+30. Save entry0,first1,permanent5/15/30 and alternating
durable latest checkpoint every update. Stop30 for review, not SSL completion.
Do not retune mid-arm or reject KL magnitude. Finite/corruption protection and
user STOP remain authoritative. No automatic continuation of a completed arm.

Preflight:1024worlds/90decisions, full loss backward and critic-isolation check,
zero optimizer steps. Fourteen focused loss/runtime/adjacent tests passed.
Authority/package must be frozen, committed, pushed and remotely read back
before the first accepted PPO step. Runtime checks source/evidence identities.
Checkpoint package includes the exploration formula and coefficient.

External run:G:/dev/RivalSim-runs/direct-skills-log-barrier-v1.
Use the existing exclusive GPU lease. Never overlap GPU workers.

```powershell
.venv\Scripts\python.exe -B benchmarks/run_direct_skills_log_barrier_v1.py verify
.venv\Scripts\python.exe -u -B benchmarks/run_direct_skills_log_barrier_v1.py run
```
