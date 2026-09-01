# Rival 2.0 — Human BC V5

Pull latest `origin/main` and execute this task end-to-end.

This is a **minimal correction and stronger-training retry** of Human BC V4. Do not turn it into another research or validation project.

The required base result is the V4 blocked evidence commit:

`9bb0dd0be8beedfeed5fe2c4e24938b166a88646`

The V4 failure is already understood: the candidate boost-saturation guard required `< 0.95` even though the required Human BC V1 parent itself is approximately `0.96–0.98` saturated on several simulator groups. All four V4 proposals otherwise passed KL, critic, finite-output, frozen-parameter, tail, and human-distribution checks. V4 accepted zero steps and never opened its untouched simulator test.

Do not investigate V2/V3/V4 again. Do not use old failed checkpoints or opened tests as validation. They are historical diagnostics only.

## Goal

Actually train Human BC V5 from the accepted Human BC V1 parent, allow materially more movement toward the human demonstrations than V3/V4, preserve the meaningful tail protections already implemented in V4, and select the most human-like checkpoint that remains inside the real retention safety limits.

Human imitation is the primary objective. Retention exists to prevent destructive policy drift, not to keep the actor nearly identical to BC V1.

## Parent

Train from exactly:

`checkpoints/rival2/human_bc_v1/rival2_human_bc_v1.pt`

SHA-256:

`560C2414C17039DC920126EA148BF73FE6CC4677EE440F043599A7E1C76D2874`

Do not initialize from V2, V3, or V4.

Keep the shared trunk and critic frozen for this campaign. Train the actor head only. Use a fresh AdamW optimizer. No PPO optimizer state and no prior BC optimizer state.

## Reuse V4 simulator authority; do not manufacture more

Reuse the already-created V4 simulator assets directly:

- existing complete simulator validation corpus/split;
- existing orientation-sensitive stratum;
- existing hard-tail training candidate pool/replay design;
- existing stress-validation corpus;
- existing still-sealed V4 untouched simulator test authority.

Do **not** generate a V5 replacement for any of these merely because the campaign number changed.

The V4 untouched simulator test was never student-evaluated and remains sealed. Treat it as the V5 final untouched simulator test. Keep it sealed until one final V5 checkpoint has been selected using training/validation only.

Do not evaluate any previously opened V2 or V3 simulator test.

Do not add another validation corpus, another historical regression matrix, or another campaign-comparison gate.

## Fix the one known V4 blocker

Replace the absolute simulator **boost probability saturation** candidate rule with a BC-V1-parent-relative rule.

For each simulator validation group:

- if the exact BC V1 parent boost-saturation fraction is `<= 0.95`, the trained candidate must remain `<= 0.95`;
- if the exact BC V1 parent is already `> 0.95`, the trained candidate may retain the inherited condition but must not materially worsen it;
- freeze the worsening allowance at **+0.005 absolute saturation fraction** relative to the exact BC V1 parent on that same group.

So the candidate limit is:

`max(0.95, BC_V1_parent_saturation + 0.005)`

This exception applies only to inherited boost saturation. Do not broadly relax unrelated distribution-health checks.

At the final one-time untouched-test access, evaluate the selected V5 checkpoint and BC V1 parent on that test in the same final evaluation pass and apply the same parent-relative boost-saturation rule. Do not use that result for tuning.

Do not require saturation to monotonically improve every validation interval. The rule prevents **new collapse**; it does not make repairing BC V1's inherited boost confidence a prerequisite for learning.

## Keep the real hard retention guard

The unchanged hard safety limit remains:

`all-perspective maximum per-sample actor KL <= 2.0`

Apply it to complete validation, stress validation, and the final untouched simulator test.

Keep finite-output checks and frozen trunk/critic integrity.

Do not add a second stricter maximum-sample-KL eligibility wall below `2.0`.

In particular, remove V4's separate `1.0` maximum-sample-KL selection-margin rejection. A validation-safe candidate may move anywhere below the real `2.0` hard limit. KL can still influence checkpoint ranking softly.

The orientation-tail telemetry remains useful, but do not reject an otherwise hard-safe candidate because of V4's separate `0.5` single-orientation-channel selection margin. The total hard per-sample actor KL limit is authoritative.

## Make BC meaningfully less conservative

V4's retention objective was:

`2*mean(K) + 4*mean(total barrier) + 4*top-1%-CVaR(total barrier) + 4*top-1%-CVaR(orientation barrier)`

That is too preservation-heavy for this campaign.

Use this V5 retention objective instead, keeping the same definitions, activation thresholds, temperatures, and top-1% CVaR construction:

`0.5*mean(K) + 1.5*mean(total barrier) + 2.0*top-1%-CVaR(total barrier) + 2.0*top-1%-CVaR(orientation barrier)`

Keep:

- total sample barrier activation threshold `0.5`;
- total sample temperature `0.05`;
- orientation activation threshold `0.125`;
- orientation temperature `0.0125`;
- hard-tail replay/mining machinery from V4.

Do not introduce additional retention terms.

## Optimizer / duration

Use actor-only AdamW with:

- initial LR: `5e-5`;
- weight decay: `1e-5`;
- betas: `(0.9, 0.999)`;
- epsilon: `1e-8`;
- gradient clip norm: `1.0`;
- maximum accepted supervised steps: `10,000`;
- validation interval: **128 accepted optimizer steps**.

If a transactional interval violates an actual hard guard, roll it back and retry at:

1. `5e-5`
2. `2.5e-5`
3. `1.25e-5`

Do not descend into a long ladder of tiny learning rates. If all three fail the same genuine hard guard, stop and report that blocker.

A parent-relative boost-saturation value inside its frozen allowance is **not** a failure and must not trigger rollback.

## Human objective

Keep the existing frozen human demonstration dataset, observation adapter V2, exact 120 Hz actions, whole-attempt splits, mechanic-aware sampling, Smooth-L1 analog objective, BCE button objective, and positive-mechanic adjudication.

Do not add or relabel demonstrations in this task.

Let the actor learn from them.

## Checkpoint selection should favor imitation

Use hard guards as hard guards, not as most of the optimization objective.

Rank eligible checkpoints with these weights:

- gameplay validation RMSE ratio: `0.36`;
- mechanic validation RMSE ratio: `0.36`;
- mean per-mechanic-label RMSE ratio: `0.18`;
- complete-validation mean-KL soft ratio: `0.03`;
- complete-validation max-KL / hard-limit ratio: `0.02`;
- stress-validation mean-KL soft ratio: `0.03`;
- stress-validation max-KL / hard-limit ratio: `0.02`.

Total human-imitation weight = `0.90`.
Total retention-ranking weight = `0.10`.

Retention hard limits remain mandatory regardless of score.

For human eligibility:

- both gameplay and mechanics must improve over BC V1;
- at least 60% of mechanic labels must strictly improve;
- at least 80% of mechanic labels must remain within **3%** of BC V1 or better;
- do not require a fixed 5% improvement in each human family before a checkpoint is eligible. Any strict family improvement is enough; let the score select the best useful checkpoint.

## Plateau / training length

Do not stop this campaign after only a small amount of actor movement.

- Do not allow plateau termination before **3,072 accepted steps**.
- After 3,072 accepted steps, use **16 consecutive 128-step validation boundaries** without a material combined-score improvement of `0.0005` as the plateau stop.
- Maximum remains 10,000 accepted steps.

The selected checkpoint may be earlier than the final executed step if validation shows it was better.

## Tests and integrity

Keep this bounded.

Before training, run the focused tests required to verify the V5 changes:

- parent-relative boost-saturation logic;
- V4 authority reuse and sealed-test identity;
- revised retention coefficients;
- revised optimizer/LR schedule;
- frozen trunk/critic and actor-only optimizer membership;
- unchanged hard max sample KL `2.0`;
- removal of the extra `1.0`/`0.5` selection-margin rejections.

Then train.

Do not build another large static audit package before optimizer construction.

Do not add new gates simply because V2, V3, or V4 failed for different known reasons.

## Final test discipline

After selecting one final V5 checkpoint from training/validation only:

1. evaluate the untouched human test once;
2. open the reused sealed V4 simulator test once;
3. evaluate BC V1 parent + selected V5 candidate in that same simulator-test pass for parent-relative boost saturation;
4. apply the unchanged `max sample KL <= 2.0` hard guard;
5. do not reopen training or checkpoint selection afterward.

If it passes, V5 is PASS.
If it fails a genuine hard final-test guard, V5 is BLOCKED and report the exact failed condition.

## Prohibited

Do not change:

- rewards;
- PPO;
- RivalSim physics;
- mechanic definitions/rewards;
- action/observation contracts;
- observation adapter;
- raw recordings;
- frozen human data/splits.

Do not train from a blocked checkpoint.

## Output

Create:

`checkpoints/rival2/human_bc_v5/rival2_human_bc_v5.pt`

Commit and push all implementation, concise evidence, training curve, selected checkpoint, and final result to `origin/main`.

## Codex final response

Return only:

- PASS / BLOCKED;
- final Git commit SHA;
- selected V5 checkpoint path + SHA;
- accepted supervised steps executed and selected step;
- gameplay validation RMSE, BC V1 -> V5;
- mechanic validation RMSE, BC V1 -> V5;
- mechanic labels improved / total;
- complete-validation max KL;
- stress-validation max KL;
- untouched-test max KL;
- parent -> V5 boost saturation on complete/stress/test all-perspective groups;
- stop reason;
- recommended next step.

Keep detailed tables, curves, hashes, and diagnostics in Git. Do not dump a large report into the thread.