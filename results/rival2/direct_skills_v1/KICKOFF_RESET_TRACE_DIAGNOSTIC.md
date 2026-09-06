# Focused learning/transfer diagnosis prepared after +300

This is a read-only investigation, not a new training campaign or a reward change.
The live learner continues toward its previously declared +350 review. No native
trace has been executed while preparing this document.

## Evidence that narrows the question

`learning_transfer_review_000300.json` reduces already committed curves and match
logs. It verifies action-count totals and button sums against the independent
logged totals, reconstructs the signed drill bonuses from event counts/weights,
and reconciles first-versus-later goal partitions with complete match scoreboards.
It creates no policy evaluation or optimizer step.

| Full-match checkpoint | Rival / Nexto first goals across 10 matches | Rival / Nexto goals after the first goal |
|---|---:|---:|
| +0 | 0 / 10 | 1 / 256 |
| +50 | 4 / 6 | 13 / 206 |
| +100 | 4 / 6 | 8 / 203 |
| +150 | 4 / 6 | 20 / 197 |
| +200 | 6 / 4 | 27 / 183 |
| +250 | 8 / 2 | 36 / 181 |
| +300 | 8 / 2 | 21 / 195 |

At +300 Rival scores the first goal from initial layouts 0/1/2/3 on both sides,
matching the short drill's successful layouts. Only one match has Rival score
the second goal. The first goal is not an independent whole-match test, and
later episodes have far more exposure. Nonetheless this sharp first/post-goal
gap is a concrete transfer question, not merely a flat scalar training curve.

Read-only source inspection identifies differences to measure, not assumed bugs:

- Skill/training Nexto episode activation clears its previous action and neural
  counter; the complete-match adapter keeps its neural cadence across goals and
  only resets its kickoff-sequence index. The latter is explicitly documented
  as stock-bot behavior. Its causal effect has not been measured here.
- Initial kickoff states are built from a CPU snapshot; post-goal states use the
  accepted native selective reset. Both describe standard kickoffs, but exact
  observed state equality must be checked rather than inferred from their names.
- Rival hidden state is explicitly zeroed at every full-match goal reset. The
  native reset clears previous-action fields and episode/contact timers. The
  trace will measure these values immediately before policy inference.
- The observation builder concatenates/stacks fresh tensors. The old native
  preflight also recorded zero log-probability replay error, so no evidence of a
  rollout observation-aliasing failure was found by this read-only inspection.

Do not change Nexto scheduling, standard reset state or Rival observations based
only on these correlations. Do not relabel the full-match regression as a simulator
bug without a native reproduction and an independently justified correction.

## Other measured learning signals

From updates 201–250 to 252–300 (excluding the fresh-episode resume transient):

- Forward-throttle requests: 82.39% → 82.69%.
- Reverse-throttle requests: 8.02% → 7.57%.
- Jump requests: 1.727% → 1.725%.
- Boost requests: 63.44% → 58.56%.
- Handbrake requests: 4.72% → 4.02%.
- Mean categorical entropy: 0.7756 → 0.6400 nats.

These are stochastic learner requests, not measured flips, applied boost
consumption, purposeful recoveries or good control. Jump incidence does not
support blaming this block on increased random jumping. Declining entropy may
matter to exploration, but it is not by itself a proven cause or a license to
change the frozen coefficient during the block.

Absolute direct-event-plus-approach volume versus absolute terminal goal reward
is about 17.1% in finishing/Nexto, 11.5% in challenge/Nexto, 20.3% in defense/Nexto
and 17.3% in kickoff/Nexto for 252–300. The JSON reports exact lower/upper bounds
for terminal within-decision discount. Positive and negative goal rewards are
not cancelled in this denominator. These are reward magnitude comparisons, NOT
gradient attribution. Local easy bonuses can still bias learning even when
terminal reward has the larger aggregate absolute magnitude.

No per-task actor gradients, value explained variance or task-conditioned GAE
are retained by the current curve. Their causal contributions remain unmeasured.
Do not invent those diagnostics from the aggregate loss/entropy fields.

## Prospective native trace

Helper: `benchmarks/trace_rival2_direct_skills_kickoff_reset.py`.

1. Use a completed scheduled checkpoint/evaluation, initially +300, only after
   the learner stops at an accepted boundary under the +350 review decision.
   Preserve the newest accepted rolling checkpoint; this diagnostic is never
   a reason to resume from the older +300 model.
2. Acquire the exact existing exclusive GPU lease and an owned evaluation stream.
   Refuse overlapping training, unknown authority hashes or an existing output.
3. Run the unchanged ten-match deterministic evaluator once with the exact saved
   checkpoint, fixed initial layouts/sides and original regulation/overtime cap.
4. Wrap only its existing policy-action method, calling that method exactly once.
   Copy the first eight 30-Hz decisions (32 physics ticks) of each initial and
   post-goal kickoff. No input, state, timing, hidden value, reward or opponent
   control is rewritten. Other worlds in each copied row are explicitly masked.
5. Record pre-policy 182 observations, pre/post recurrent hidden state, actual
   selected Rival action, physical ball/car state, individual wheel contacts,
   native lifecycle layout/selector, match kickoff/goal counters, and Nexto's
   previous action, neural counter and kickoff-sequence index. Initial native
   layout metadata can be -1; the original match starting layout is also retained.
6. Require exact equality with every previously saved raw match value, summary
   and hidden-reset count. Also require unchanged model/checkpoint hashes,
   finite arrays and lossless NPZ readback. Record failure and disallow trace
   interpretation if parity fails; do not relax the comparison.
7. Compare initial and later starts of the same layout and side field-by-field,
   then compare deterministic controls and opponent scheduling state. Report
   observed differences separately from hypotheses about why the match diverges.

The maximum recording allocation is bounded by the unchanged evaluator's total
decision limit. The helper writes no campaign state, checkpoint or existing
evaluation. It never constructs an optimizer, accepts a PPO update, or changes
the frozen runtime sources. Existing +150/+250 finishing traces are not rerun.

```powershell
# ONLY after a verified accepted-boundary pause and release of the GPU lease:
.venv/Scripts/python.exe benchmarks/trace_rival2_direct_skills_kickoff_reset.py --update 300 --output results/rival2/direct_skills_v1/kickoff_reset_trace_000300
```

CPU tests cover original-call preservation, copied input/output lifetime, flat
wheel-buffer shape, exact capture window, completed-world masking, and exception
propagation/restoration, alongside the existing bounded trace/archive tests.
An additional guard test verifies that a busy learner lease prevents creation
of any evaluation stream, runner or output directory. All 25 pass with an
isolated pytest temporary directory. The initial combined
invocation again encountered Windows default-temp fixture permissions (22 pass,
two setup errors); both generated XML records are retained, including raw
traceback whitespace. These tests do not establish native replay success.

The helper and this prospective method must be committed, pushed and read back
before the first native diagnostic execution. Reward/PPO/model authority remains
unchanged. If +350 recovers match performance, review the measured evidence before
interrupting it; the trace is prepared, not an unconditional stop request.

## Follow-up source finding after the +350 review

The +350 score recovered to +250, so the conditional pause did not trigger.
The next decision is the published +400 review. While the learner continued,
CPU source inspection identified a specific input-asymmetry candidate:

- `rivalsim/vehicle_state.py:170` initializes individual `wheel_contact` flags
  to zero for a new world, including the full evaluator's initial kickoff.
- `rivalsim/ssl_foundation_v1.py:848` explicitly clears them in each curriculum
  reset. The same reset also clears world-contact flags, wheel/contact counts
  and handbrake interpolation state.
- `rivalsim/kernels/rival2.py:803` (`rival2_interval_reset`) does not receive the
  wheel-contact buffer. The later strict-dash reset reads that buffer; it does
  not clear it. With no scenario template in `CandidateMatchRunner`, the
  curriculum's wheel-state clearing is not applied after full-match goals.
- `Rival2TensorBridge._car_block` reads those individual flags directly into
  both car observation blocks. The full-match runner builds the reset observation
  and selects its next action before another physics tick recomputes contacts.

This establishes a source-level difference between the reset paths. It does not
establish the actual retained values, whether they differ for a particular
kickoff, whether the deterministic action changes, or how much of the later-goal
deficit it explains. Other reset state and Nexto cadence remain competing or
interacting explanations. The existing trace already records wheel flags, both
182-field observations, hidden state and selected action; no extra native replay
or instrumentation change was made.

`kickoff_wheel_source_audit.json` binds six inspected source hashes and four
CPU AST checks. Those checks verify source structure, not native reset correctness
or behavioral causality. The first inline inspection invocation had a shell-quote
syntax error before execution; a corrected AST-only invocation produced the
artifact. Neither invocation constructed a policy, optimizer or simulator.

If the prepared trace is run, explicitly compare wheel flags at age zero for
matching layout/side starts before attributing differences to hidden memory or
Nexto. Do not automatically force these flags to zero or one: a correction must
first establish the intended reset-state semantics and preserve physical validity.
No training source, reward, physics, checkpoint or evaluation result was changed.
