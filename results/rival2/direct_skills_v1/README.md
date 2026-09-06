# Direct skill objectives plus natural gameplay — one entity-aware policy

Latest completed review: [offset +150](EVALUATION_000150.md). Ten full Nexto
matches produced 24/203 goals for/against, versus 12/209 at +100 and 17/212 at
+50. All ten remained losses. Contacts were almost unchanged at 368 versus 366;
finishing-drill goals increased from 8 to 9 of 64. There is actual scoring
progress, not uniformly better possession: same-player follow-ups fell to
25/339, challenge control remains only 3/64, and ongoing-ground acquisition
regressed. The review explains the follow-up denominator's exclusion of goals;
its fraction alone is not an exact possession-retention measure.
Continue unchanged to the scheduled +200 review, comparing +50, +100 and +150
and explicitly tracking open-play control rather than declaring the drills
solved. No SSL/promotion verdict, reward retuning or deployment is warranted.
Keep all independently preserved snapshots. Shooting is already 20% of the
episode-source mixture; no extra shooting family was added to this frozen run.
Use `benchmarks/report_rival2_direct_skills.py --update N`, including
`--start-groups`, after the corresponding completed evaluation/checkpoint audit.
Also compare each new review with the previous completed review, for example
`--update 150 --baseline 100` (with and without `--start-groups`). These CPU-only
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
