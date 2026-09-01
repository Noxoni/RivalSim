# Human BC V3 blocked audit

## Verdict

Human BC V3 is **BLOCKED**. The actor-only campaign trained and selected a
validation-safe checkpoint, but the new untouched simulator test failed the
prospectively frozen maximum per-sample actor KL guard:

- frozen limit: `2.0`;
- new untouched test maximum: `2.1052461326179`;
- new untouched test mean: `0.003074094972877279`;
- samples above `0.5`: 7 of 838,912;
- samples above `1.0`: 2 of 838,912;
- samples above `2.0`: 1 of 838,912.

The test result was not used to reselect, retune, or continue training. The
human test and new simulator test were each opened exactly once after final
validation-only selection. The previously opened V2 simulator test was never
used by V3 selection or acceptance.

## Training and selection

The campaign executed 2,048 accepted actor-only steps and stopped because the
prospectively frozen validation score plateau patience expired. No training
guard stopped the run and no transactional retry was required. The selected
checkpoint is the validation-only candidate at +1,152 accepted steps:

- gameplay validation complete-action RMSE: `0.5600362420082092` to
  `0.5204758644104004`;
- mechanic validation complete-action RMSE: `0.5355024933815002` to
  `0.4887794852256775`;
- mechanic labels improved: 13 of 13;
- mechanic labels nonregressed: 13 of 13;
- complete-validation actor mean KL: `0.0031983704771104673`;
- complete-validation maximum sample KL: `0.36273455690025`;
- complete-validation samples above `0.5`, `1.0`, or `2.0`: zero.

The untouched human test, evaluated only after selection, also improved:

- gameplay complete-action RMSE: `0.5419303178787231` to
  `0.5170602202415466`;
- mechanic complete-action RMSE: `0.559705376625061` to
  `0.509960412979126`.

## Tail finding

The failed untouched-test maximum occurred in the
`current_policy_applicable` group, not the historical-opponent or
counterfactual-opponent groups. Group maxima were:

- current-policy-applicable: `2.1052461326179`;
- counterfactual opponent: `0.5493945232703801`;
- historical opponent: `0.5493945232703801`;
- teacher-defined low-variance stratum: `0.2054667761840392`.

Across the seven test samples above KL `0.5`, the channel contribution was
approximately 51.0% roll, 20.9% steer, 12.1% yaw, 8.8% pitch, 4.2% throttle,
2.3% boost, 0.6% handbrake, and 0.02% jump. The maximum individual-channel KL
was roll at `1.4574626684188843`. These statistics come from the single frozen
test evaluation already recorded; this audit performs no second evaluation.

This differs from the V2 failure mechanism the V3 strata targeted. V3
successfully bounded historical-opponent and teacher-low-variance tails, but a
different rare current-policy-applicable orientation-control state remained
outside the complete validation coverage.

## Integrity

- shared trunk and critic hash remained exactly
  `44DD21E20FAF9D07C58009D791564B7CD7A7D5FB85EF02A17E4FC99F58C250AA`;
- critic RMSE and maximum absolute drift were both zero;
- source BC V1 checkpoint, observation adapter, historical PPO optimizer,
  native recordings, dataset, and split identities remained unchanged;
- no PPO update, reward change, mechanic-definition change, observation-adapter
  change, or action/observation-contract change occurred;
- the checkpoint SHA-256 is
  `CE2A9671C3768710F5DECD456E28D1265910CB5934D0FB0BB819478E369E939B`;
- the checkpoint is blocked diagnostic evidence, is not PPO-resumable, and must
  not replace the accepted Human BC V1 parent.

## Recommended next step

Create a new prospective BC V4 authority. Use this now-opened V3 test only as
diagnostic evidence, expand teacher-only training/validation tail coverage for
rare current-policy-applicable orientation-control states using simulator
training data, and bind another disjoint untouched simulator test before any
optimizer step. Do not weaken the `2.0` hard guard and do not continue from this
blocked V3 checkpoint.
