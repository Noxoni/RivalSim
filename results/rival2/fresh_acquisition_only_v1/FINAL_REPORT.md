# Fresh acquisition-only diagnostic: final result

## Verdict

**Completed at exactly 150 accepted updates; not promoted. Training is stopped.**

A fresh randomly initialized policy learned more first contacts in this setup.
It did not become a reliable ball-acquisition policy, and improvement did not
persist uniformly. Removing the other reset families did not eliminate regression.
This experiment does not identify scenario mixing as the cause of the previous
run's long-term problems.

The run consumed 663,552,000 trainable 30 Hz decisions, 1,769,472,000 world
physics ticks, and 20,836 accepted Adam steps. It reached its prospectively frozen
diagnostic ceiling without an operational restart or a finite/corruption failure.
There was no automatic extension, promotion, reward retuning, or broad match
evaluation.

## What was actually isolated

Every reset used the acquisition family, with the same underlying 16,384 starts
represented in both opponent lanes. No other reset family was sampled. Both actor
and independent recurrent critic began randomly initialized, with fresh Adam.
No old BC, V5, or PPO checkpoint initialized this policy.

This was **acquisition-only starting states**, not a new reward paid per touch.
The original goal-first sustained reward, seven potential differences, small
continuous control payment, and inactivity penalty remained unchanged. A touch
did not end an episode. Episodes continued to a goal or 45 seconds without
either player contacting the ball. Half the worlds used evolving self-play;
half used active Nexto. Thus the experiment did not remove opponents or isolate
all other possible causes of learning difficulty.

The entity-aware recurrent actor and separate recurrent critic, 182 observations,
joint90 actions, 120 Hz physics / 30 Hz decisions, three-second rollout, two PPO
passes, actor LR 1e-4, critic LR 3e-4, and exploration settings were frozen.
KL remained telemetry only, with zero KL rejections. The maximum completed
mean KL was 0.009485978785839204. No capability or reward semantics were
changed while the experiment ran.

## Fixed deterministic acquisition probes

Each probe used the same 1,024 starts against active Nexto: 512 easy starts with
a five-second deadline and 512 varied starts with an eight-second deadline.
A success was Rival's own native first contact in the original episode, not
necessarily first contact before Nexto, possession, or a scoring opportunity.
Contact-time medians below are conditional on success. The probe's separate
no-contact fraction uses the full eight-second observation interval, including
for easy starts; it is not interchangeable with five-second easy failure rate.

These repeated development probes are not an untouched test or a promotion gate.
Observed maxima below are descriptive, not a retrospectively selected deployment
model. There was no new probe rerun during finalization.

| Accepted update | Easy success within 5 s | Varied success within 8 s | Easy conditional median, s | Varied conditional median, s |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 31/512 (6.0547%) | 11/512 (2.1484%) | 0.5083 | 1.0083 |
| 10 | 13/512 (2.5391%) | 2/512 (0.3906%) | 3.3083 | 3.2167 |
| 20 | 12/512 (2.3438%) | 3/512 (0.5859%) | 3.3083 | 3.2667 |
| 30 | 14/512 (2.7344%) | 2/512 (0.3906%) | 3.2167 | 3.2167 |
| 40 | 88/512 (17.1875%) | 22/512 (4.2969%) | 0.9833 | 2.1000 |
| 50 | 207/512 (40.4297%) | 78/512 (15.2344%) | 0.8500 | 1.9250 |
| 60 | 207/512 (40.4297%) | 36/512 (7.0313%) | 0.9250 | 2.4167 |
| 70 | 251/512 (49.0234%) | 82/512 (16.0156%) | 0.9583 | 2.4500 |
| 80 | 144/512 (28.1250%) | 47/512 (9.1797%) | 0.8833 | 2.5083 |
| 90 | 201/512 (39.2578%) | 91/512 (17.7734%) | 0.8750 | 2.2750 |
| 100 | 190/512 (37.1094%) | 117/512 (22.8516%) | 0.9083 | 2.6167 |
| 110 | 153/512 (29.8828%) | 98/512 (19.1406%) | 0.8667 | 2.2583 |
| 120 | 170/512 (33.2031%) | 123/512 (24.0234%) | 0.9250 | 2.2000 |
| 130 | 166/512 (32.4219%) | 112/512 (21.8750%) | 0.9500 | 2.1750 |
| 140 | 107/512 (20.8984%) | 86/512 (16.7969%) | 1.0583 | 2.2667 |
| 150 | 160/512 (31.2500%) | 141/512 (27.5391%) | 1.0917 | 2.2750 |

- Untrained: 31 easy and 11 varied successes, 42/1,024 combined.
- Best observed combined result: update 70, 251 easy plus 82 varied = 333/1,024.
- Best observed easy result: update 70, 49.0234375%.
- Best observed varied result: update 150, 27.5390625%.
- Final: 160 easy plus 141 varied = 301/1,024.
- Final conditional contact medians: 1.0916667 s easy and 2.275 s varied.
- Update 140 fell to 107 easy / 86 varied; update 150 recovered to 160 / 141.
  This recovery does not erase the regression from the easy-start peak.
- At the final checkpoint, 352/512 easy starts missed the five-second deadline;
  371/512 varied starts missed the eight-second deadline.

The policy therefore demonstrated learnability, but not consistent acquisition.
663.6 million learner decisions for these contact rates is not a claim of
successful general gameplay.

## Sustained training telemetry, separately from fixed probes

The following are stochastic rollout summaries, not paired deterministic tests.
The first and last windows each contain ten updates and 491,520 world-seconds
per opponent lane. Changed episode occupancy and the evolving self-play
opponent make these different state distributions.

| Telemetry | Updates 1-10 | Updates 141-150 |
| --- | ---: | ---: |
| Mean logged touches/min | 1.066325 | 1.860270 |
| Mean logged movement speed, uu/s | 685.848880 | 344.668975 |
| Mean categorical entropy | 4.459236 | 3.443134 |
| Nexto-lane focal goals / opponent goals | 3 / 76,422 | 1 / 2,320 |
| Nexto-lane focal contact count | 21,060 | 2,357 |
| Nexto-lane inactivity resets | 0 | 9,544 |
| Self-play lane focal goals / opponent goals | 536 / 195 | 1,506 / 1,604 |
| Self-play lane inactivity resets | 0 | 1,419 |

The late inactivity resets and lower movement speed are adverse evidence.
Fewer late Nexto goals conceded cannot alone be interpreted as stronger defense:
far more episodes were reaching inactivity reset, and Rival still scored only
one goal in that lane during the last ten-update window. The rise in aggregate
touches/min masks materially different self-play and Nexto-lane outcomes.
No general-match capability claim follows from these logs.

## Comparison with the preserved mixed-scenario lineage

The same probe/specification hashes were verified in all 13 earlier acquisition
evaluations from updates 27 through 147. Selected points:

| Previous mixed lineage update | Easy /512 | Varied /512 |
| ---: | ---: | ---: |
| 27 | 114 | 49 |
| 37 | 205 | 81 |
| 107 | 329 | 181 |
| 117 | 315 | 230 |
| 137 | 57 | 104 |
| 147 | 39 | 104 |

The earlier lineage's easy maximum was 329/512 (64.2578125%) at 107;
its varied maximum was 230/512 (44.921875%) at 117. Its combined maximum
was 545/1,024 at 117. Those maxima exceed this fresh acquisition-only run.
Its last acquisition probe regressed to 39 easy /104 varied.

Do not omit contradictory historical evidence: the earlier lineage's separate
full-match evaluation at 150 recorded 179 Rival contacts (3.58/min), no
touchless matches, and 89 kickoff first contacts, despite zero wins and a
0-364 scoring total over ten matches. Its update-100 full-match probe had only
four contacts (0.08/min) and seven touchless matches. Thus the old lineage was
not uniformly worsening on every behavioral measurement.

These are not matched causal controls. The previous policy had prior training,
a mixed reset curriculum, and a stationary-kickoff amendment during its run.
The fresh run changed initialization as well as scenario mix. Both show gains
and setbacks; neither proves that a particular setting is the cause.
The earlier campaign remains stopped at 156 with its own checkpoint and Adam
preserved. No old optimizer or weights were loaded here.

## Checkpoint preservation and integrity

Final permanent checkpoint:

- Path: `checkpoints/rival2/fresh_acquisition_only_v1/plus_000150.pt`
- SHA-256: `676575E3B009D0DD0273350363CE46540030C5AC5DE14D32FFB3C671E2C53712`

Best observed combined acquisition checkpoint (inspection only):

- Path: `checkpoints/rival2/fresh_acquisition_only_v1/plus_000070.pt`
- SHA-256: `9EE3F3E5E10AFB5483E47AC81A96C6C3D26082EC90F9578136D7CBF83685445E`

Latest accepted rolling checkpoint:

- Path: `G:/dev/RivalSim-runs/fresh-acquisition-only-v1/rolling_0.pt`
- SHA-256: `656D6B1B78F9BDDAC6EF122A6C1B7DB6261C7BC05112871DAC84485C8E42A06C`

All permanent probe checkpoints, including baseline, are preserved. The final
rolling model and every Adam tensor are exactly equal to the final permanent
checkpoint; their file hashes differ because the rolling file also records the
just-completed probe. Permanent snapshots are saved before evaluation, so their
`last_probe_update` can correctly name the preceding probe.

The CPU-only final audit verified:

- Frozen source/package/authority hashes and unchanged fresh initialization.
- All 16 checkpoint hashes against completed result and started receipts.
- 16,384 raw first-contact tick records; success counts, fractions, and
  conditional medians independently recomputed.
- Identical probe scenario and specification hashes at every evaluation.
- Zero optimizer steps and unchanged model/Nexto/checkpoint during each probe.
- Finite model and optimizer state; accepted offsets, exposure, fresh lineage,
  and Adam counters consistent with the closed update history.
- Exactly 150 contiguous update rows, no non-acquisition family exposure,
  no process resume, no failure record, and empty stderr.
- Each existing per-boundary audit passed all 16 checks.

Pre-training validation passed 20 focused tests and an actual 32,768-world
rollout/backward preflight with zero optimizer steps. Those checks establish
implementation integrity, not behavioral success. Finalization performed only
CPU inspection of existing evidence; it did not train or reevaluate the policy.

Reproduction entrypoints are documented in README.md. The existing command
`report --offset N` reads a completed probe/checkpoint and creates its audit and
closed training prefix without launching a GPU evaluation. FINAL_AUDIT.json
contains all checkpoint/result/receipt identities, raw-record audit outcomes,
the full probe series, the old series, and training-window aggregates.
`through_000150.json` preserves the complete closed training curve.

## Recommendation

Keep training stopped. Preserve update 70 and update 150 for inspection, without
calling either a gameplay-ready model. Investigate why state/reward advantages
and value estimates support the observed inactive behavior before spending
another long run. This is a recommended follow-up, not an already proven cause
or permission to alter the frozen experiment. If the specific hypothesis is
that mixed resets caused the regression, a matched fresh mixed-reset control
and additional seeds would be needed; this run alone does not establish it.
