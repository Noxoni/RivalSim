# Direct skill objectives plus natural gameplay — one entity-aware policy

Latest completed review: [offset +675](EVALUATION_000675.md). The bounded shooting
curriculum block completed cleanly. Original finishing stays5/64; ten full Nexto
matches worsen42/139 ->37/168 goals,0/10wins,9/10 cases worse than650. Follow-ups
39.21% ->31.22%. Do not automatically extend a non-transferring curriculum.
The current675 checkpoint/model/Adam/RNG and full evidence are preserved. Next:
[bounded corrected-runtime kickoff continuity tap](kickoff_continuity_000675/AUTHORITY.md),
no learning, with exact original fullmatch replay required. No new reset defect
is claimed from scores alone. Goal active and unmet; current runner has exited.

Previous completed review: [offset +650](EVALUATION_000650.md). Ten full Nexto
matches produced **42/139 goals for/against and0/10 wins**, versus65/144 at600.
Repeat-contact fraction rose29.3% to39.2%, but scoring fell and finishing remains
5/64. These are not SSL-level results. The learner stopped cleanly at650.
The frozen [sampling-versus-greedy diagnostic](sampling_diagnostic_000650/RESULTS.md)
is now complete: three seeds average37/146.667 goals,0wins,649contacts. Every
sampled seed scores less and concedes more than greedy650. Finishing is only
5-8 goals/64 despite52-54 on-target proxies. Sampling does not rescue this model.
No weights, reward, PPO or deployment changed; the650 checkpoint is preserved.
The [shooting-pressure comparison](shooting_diagnostic_000650/RESULTS.md) is also
complete:49/64 initially open-net goals,50/64 recovering-defender goals,5/64 set
keeper goals. Rival gets first contact in all cases. The next bounded
[650to675 curriculum-only continuation](shooting_progress_v1/README.md) changes
half of finishing starts to intermediate pressure while retaining all other
states,rewards,PPO and source650 lineage. Its package was published/verified at
957a039f634a4b9ab67eab993b9d1a47604ff912 before launch. Actual learner20016 started
2026-09-06T10:40:00Z; [entry/first-update proof](shooting_progress_v1/ENTRY.md)
confirms bit-exact source model/Adam/RNG and accepted651. Evaluation stays on
the original cases. Do not repeat completed
diagnostics or restart the expired650 runner. The goal remains active and unmet.

The [bounded no-learning diagnosis](LEARNING_SIGNAL_000550.md) verified goal
reward/GAE/recurrent replay and found narrow exploration. Simple critic
calibration did not improve held-out error; no critic or reward edit was made.
After a clean +554 pause, the same model/Adam lineage resumed under the
[prospective temperature-only amendment](exploration_t2_v1/authority.json),
published in commit `8ce15e88828ac275fe7375d7919000c1218969c8` before training.
The original `benchmarks/run_direct_skills_exploration_v1.py` segment ended at600.
The `benchmarks/run_direct_skills_exploration_followup_v1.py` block ended at650.
Do not restart either bounded runner after this intentional review stop.
Training used temperature2 in both sampling and PPO likelihoods; deterministic
evaluation remains original raw argmax. The user's ongoing goal is not canceled
by the review/diagnostic pause.

See [resume evidence](exploration_t2_v1/RESUME.md). Preserve all snapshots and
the original reward, authority and 20 runtime-source identities. Shooting
remains **20% of episode-source starts**; it was already included and has not
been duplicated. No capability or RLBot deployment promotion is claimed.
Use `benchmarks/report_rival2_direct_skills.py --update N`, including
`--start-groups`, after the corresponding completed evaluation/checkpoint audit.
Also compare each new review with the previous completed review, for example
`--update 550 --baseline 500` (with and without `--start-groups`). These CPU-only
reports use the saved outcomes and preserve the separate baseline-zero reports;
they do not run an extra evaluation or change training/selection semantics.

The user explicitly superseded the potential-only continuation on September 6,
2026 (UTC). The old learner exited cleanly at entity **+293**, 52,150 cumulative
Adam steps, without a safety failure. Its +300 continuation/evaluation plan is
superseded, not completed. Its exact latest checkpoint is preserved as:

`checkpoints/rival2/direct_skills_v1/parent_entity_000293.pt`

SHA-256 `01CD1D075C3319D19FF607C0498DFFF6FE6335BE70EEA9708B2679884601DCBE`.

The previous source has 1,676,472,162 entity-stage learner decisions. This is not
the total lifetime sample count. Existing model weights and Adam state are used
without a restart, adapter, behavior cloning, new architecture or specialist
router. The new reward and scenario authority begins at **direct-skills +0**.
The old campaign's user-transition STOP remains in place; do not resume that
superseded campaign.

## What changes

The episode source bank is 30% natural play, 20% close challenges, 20% finishing,
15% defense and 15% standard kickoff. These are **episode-source proportions**,
not promises of equal-duration sample proportions. Natural starts mix standard
kickoffs and coherent ongoing ground states. Skill episodes last at most twelve
seconds; natural episodes use the existing thirty-second/time-limit and
fifteen-second no-touch handling. Goals always end/reset the physical episode.
Task success alone does not reset: Rival can follow through and score.

All worlds use the SAME learned policy. Only native 182 observations enter the
network; family, focal side, success flags and reward bookkeeping are not model
inputs. In shooting/defense setups the opposing player receives the complementary
defense/shooting role. Challenge and kickoff objectives are symmetric. These
reward roles do not select a controller at inference time.

Half the world slots are current self-play; half face inference-only Nexto.
Both current agents learn in self-play, only the current agent learns versus
Nexto. Consequently exactly **one third of learner samples** face Nexto, rather
than letting shorter Nexto episodes reduce an episode assignment probability to
about 7% sample exposure. This is a declared change, not the old20% schedule.
Nexto retains its pinned15Hz actor and stock kickoff; Rival has no action prefix.

## Rewards

All roles receive native first-goal terminal outcome **+10/-10**. Natural play
retains the exact seven existing potential differences with actual PPO gamma.
Natural play has **no direct skill bonuses**. Drill roles replace those shaping
potentials with the direct events below, so the combined training objective is
deliberately NOT globally policy-invariant anymore. This follows the new request.

| Direct event | Amount | Scope and anti-farming boundary |
|---|---:|---|
| First native touch | +0.50 | Challenge/finishing/kickoff, once per player episode |
| Controlled acquisition | +1.50 | Challenge/kickoff/defense, once after eight consecutive decision states with own contact history and favorable proximity/relative velocity |
| Forward controlled advancement | +0.75 | Challenge/kickoff, once after400uu canonical advancement after acquisition while still controlled |
| Goal-bound contact | +1.00 | Finishing, once for native contact plus the declared bounded goal-line projection |
| Shot neutralization | +1.50 | Defense, once after a contact removes a previously projected threat and it stays safe for eight decisions |
| Loss of acquired control | -0.50 | Challenge/kickoff, once if the opponent subsequently establishes control |
| Failed timed attempt | -1.00 | Only skill timeout without the role success; never an additional goal-terminal penalty |

Challenge/finishing additionally receive at most +0.5 per second for positive
physical velocity toward the ball until their first touch (maximum6 over a
twelve-second episode, normally much less). No reward for a throttle button,
rotation, jump, flip, aerial, dash, demo, boost press or named mechanic is added.

**Kickoff fakes:** waiting or allowing the opponent first touch earns nothing by
itself. Securing controlled possession and advancing it is valuable whether Rival
rushed or let the opponent touch first. This permits a fake to be useful without
paying inactivity; it does NOT establish that the policy has learned fakes. Both
current self-play and rushing Nexto supply opposition. Faking competence needs
observed gameplay, not inference from reward code.

`reward_authority()` specifies all thresholds and reward amounts. They are frozen
before optimization. Possession and shot-clear predicates are explicit **task
proxies**, not universal measurements of intention or mechanically exact save
classification. In particular goal projection is bounded straight-line xy with
ballistic/floor-clamped z; it does not predict wall rebounds or arbitrary aerial
trajectories. Defense starts are ground shots. A native no-car-obstruction test
verifies those initial shots actually score without interception.

## Learning, evidence and evaluation

Unchanged30Hz decisions/120Hz physics, four-tick action hold,32768 worlds,
90-decision recurrent rollout, gamma.995/lambda.9973145188572297, two PPO epochs,
actorLR1e-4/independentcriticLR3e-4. KL remains telemetry only. Finite checks and
whole-update corruption rollback remain. Existing opponent-family-local advantage
normalization is retained; no reward-specific critic or architecture is introduced.

The no-step32768-world preflight checks exact action/log-probability replay,
critic gradient isolation, frozen parent/Adam/Nexto, full-scale memory, reward
finiteness and sample masking. Focused tests check skill budget saturation,
no-contact/simultaneous-contact exclusions, physical goal timing on all four
ticks, one reset per goal, natural reward parity and pre-reset truncation state.
These checks establish implementation integrity, not learned skill.

Before optimization, publish `authority.json`, `package.json`, tests, native
preflight and exact parent. Run deterministic64-case evaluation per role against
Nexto at offsets0,10,25,50 and every50 afterward. Also run the existing fixed
ten-regulation-match Nexto method at0 and every50. No changes to evaluation seeds
or selection of a lucky intermediate checkpoint. Task proxy successes are
reported separately from native goals, contacts and full-match performance.

Preserve every accepted rolling checkpoint and scheduled permanent checkpoints.
The run is authorized until user stop, with no invented update/time ceiling.
If the first50 updates fail to improve drills and match play, investigate before
another large unchanged compute block. No automatic reward tuning or false SSL
claim follows from completed optimization.

Commands, from repository root:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_direct_skills_v1.py tests/test_direct_skills_native.py tests/test_ssl_entity_mixed_training.py -q --junitxml=results/rival2/direct_skills_v1/focused_tests.xml
.venv\Scripts\python.exe benchmarks/run_rival2_direct_skills_v1.py preflight
.venv\Scripts\python.exe benchmarks/run_rival2_direct_skills_v1.py prepare
# Commit/push package, parent and sources; verify remote before launch.
.venv\Scripts\python.exe -u benchmarks/run_rival2_direct_skills_v1.py run
# Resume ONLY the latest verified accepted checkpoint of this new lineage:
.venv\Scripts\python.exe -u benchmarks/run_rival2_direct_skills_v1.py run --resume <path> --resume-sha256 <hash>
```

External run state: `G:/dev/RivalSim-runs/direct-skills-v1`. An external `STOP`
requests a clean accepted-boundary stop. The runner shares the existing GPU
exclusive lease; it must not overlap another learner or evaluation. User GPU
applications are not stopped. Resume restores model,Adam,counters and action/
shuffle/CPU/CUDA RNG, with fresh episodes and zero hidden as declared. Old V5,
BC, specialist, hybrid-policy and prior entity campaign launch paths remain off.
