# Fresh acquisition restart: direct first-touch reward

User instruction: restart from scratch, reward the first touch in acquisition
scenarios, and retain all other available rewards. This is a new lineage;
fresh_acquisition_only_v1 is completed at150 and remains preserved and stopped.

## Exact reward change

- Add **+1 for each player's own first native physical contact per episode**.
- A touch by the other player does not remove that player's eligibility.
- Both players can earn their own once-only award, including simultaneous touch.
- No repeated-contact, contact-occupancy, or rollout-boundary payouts.
- At substep k of the four-tick decision, aggregate +1*physics_gamma**k.
- A contact before or on a goal tick can stack with the goal. No post-goal award.
- The award applies only to acquisition-start episodes. All current starts are
  acquisition starts. Nonlearning Nexto rewards never enter PPO.
- Touching does not end/reset the episode. Goal/concede remains +10/-10.
  Seven state potentials, sustained control, and inactivity rules stay unchanged.

The +1 value is a prospective implementation choice: a material immediate
acquisition signal, ten percent of the goal magnitude, paid once rather than
farmable by repeated touches. It is not claimed to guarantee learning or to be
policy-invariant shaping. No bonus is withdrawn or retuned during this comparison.

## Fresh initialization and unchanged comparison settings

Construct the actor, entity modules, actor GRU, independent critic MLP/GRU and
empty Adam from scratch. Use the same seed2026090810 as the preceding fresh
experiment, deliberately reproducing its *untrained* initial tensor values for
a controlled comparison. Never load any trained checkpoint as parent. The old
untrained artifact is read only for hash comparison, not for model construction.

Keep the same 32,768-row acquisition bank, 50% current-self-play/50% Nexto worlds,
learner masks, 182 observations, joint90 categorical actions, temperature1,
entropy0.001, 120Hz physics/30Hz decisions, 90-decision rollouts, two PPO passes,
actor LR1e-4, critic LR3e-4, discount and trace. Keep finite/corruption protection;
KL is telemetry only. Do not modify the previous reward implementation or any
old campaign, physics, architecture, demonstrations or observation adapter.

## Validation before campaign optimizer entry

25 focused tests passed, including actual native collision payout, repeated
collision suppression, independent player eligibility, goal availability,
reset/re-arm behavior, timeout semantics and a disposable tiny PPO update.

The actual32,768-world preflight performs rollout, loss/backward and critic
isolation checks with **zero campaign optimizer steps**. It verified5,374 learner
first-touch payouts totaling5,372.430989265442 after within-decision discount;
13,520 Nexto-only awards were excluded by learner masks. There were1,088 repeat
contact decisions with no extra award.

Paired preflight comparison with the previous no-bonus run confirmed identical
initial model tensor hash, identical scenario-bank hash, identical nonreward
telemetry, and exactly unchanged sums for every original reward component.
Only total reward increases, by the new bonus (floating-point accumulation
residual below1e-6). See MATCHED_PREFLIGHT_COMPARISON.json.

Commit/push the sources, authority, package, tests, preflight and initialization;
verify their remote bytes before the first campaign optimizer step.

## Run and monitoring

Retain the same bounded150-update comparison window and fixed deterministic
1,024-start acquisition probes at0 and every10 updates. This is a comparison,
not an indefinite new general-gameplay campaign. Preserve baseline and every
probe checkpoint, plus alternating rolling files each update. The probe is the
unchanged read-only evaluator: easy own-contact within5s, varied within8s,
against active Nexto. It need not add the training-only bonus because it does
not optimize or use reward to choose actions; physics/observations/episode rules
are identical. No training or broad match evaluation runs in the probe.

At every rollout, record learner first-touch award counts, discounted reward,
excluded opponent awards and suppressed repeats, separately from other rewards.
The runner checks payout accounting before optimization. Monitor every2minutes;
notify only newly completed evaluations, or a genuine blocking failure.

Do not infer possession/scoring skill from acquisition success. Compare full
curves against the preceding matched-seed no-bonus run, not only the best point.
At150 finalize evidence and stop; no automatic extension or reward modification.
Respect user STOP, and preserve the last accepted checkpoint on any failure.

Commands with project .venv Python:

- benchmarks/run_fresh_acquisition_touch_v1.py preflight
- benchmarks/run_fresh_acquisition_touch_v1.py freeze
- benchmarks/run_fresh_acquisition_touch_v1.py verify
- benchmarks/run_fresh_acquisition_touch_v1.py run
- benchmarks/run_fresh_acquisition_touch_v1.py report --offset N

Recovery requires explicit same-lineage --resume PATH --resume-sha256 HASH.
It restores weights/Adam/counters/RNG, creates fresh physical episodes and clears
both recurrent memories and first-touch latches. It must not claim continuation
of old in-flight episodes. No numerical/capability failure is silently resumed.

External run directory: G:/dev/RivalSim-runs/fresh-acquisition-touch-v1.
Checkpoints: checkpoints/rival2/fresh_acquisition_touch_v1.
