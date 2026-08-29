# Rival human BC continuation V2 review

Verdict: **BLOCKED**.

## Outcome

The immutable selected Human BC V1 parent reproduced under the frozen V2
authority. Cumulative step 192 was accepted on validation and selected after 32
additional optimizer steps. It reduced validation complete-action RMSE from
0.5600362420082092 to 0.5499602556228638 for gameplay and from
0.5355024933815002 to 0.5246223211288452 for mechanics. The frozen combined
selection score improved from 0.9174817773822057 to 0.9172371760057916.

The selected checkpoint passed every validation retention and output-
distribution guard. Its simulator-validation actor mean KL was
0.004958888399414718 and critic maximum absolute drift was
0.4957442283630371 against the frozen 0.5 limit.

The next cumulative step-224 boundary was attempted from the selected step-192
state at the four prospectively frozen learning rates 3e-5, 1.5e-5, 7.5e-6,
and 3.75e-6. Each candidate remained finite and actor-retention-safe, but each
failed only the frozen critic maximum-absolute-drift guard, with respective
drifts 0.5892395973205566, 0.5515842437744141, 0.5476627349853516, and
0.526914119720459. Each retry restored the same saved step-192 transactional
state, and final selection restored the saved best state. Training stopped on
the guard rather than on validation plateau or the emergency ceiling.

## Once-only test evaluation

Only after validation selection, the V2 candidate opened the human and
simulator test splits exactly once. Human test complete-action RMSE improved
from 0.5419303178787231 to 0.535649836063385 for gameplay and from
0.559705376625061 to 0.5467718839645386 for mechanics. Simulator-test actor
mean KL was 0.0050083039635960055; all actor channel, mean, maximum-sample,
finite-output, critic-RMSE, and output-distribution checks passed.

The simulator-test critic maximum absolute drift was 0.5182976722717285,
exceeding the frozen 0.5 guard. This single held-out failure makes the final
checkpoint ineligible even though the human-imitation metrics improved. The
checkpoint is preserved as an inspection artifact and must not be treated as an
accepted deployment checkpoint.

## Contract and scope audit

- The V1 source checkpoint and frozen observation adapter hashes were unchanged.
- Native source hashes and train/validation/test splits were unchanged.
- Training retained equal gameplay/mechanic human batch allocation and the
  frozen bounded mechanic hierarchy sampler; its maximum realized oversampling
  ratio remained 3.0722898230088496 under the 4.0 cap.
- The selected model and optimizer state are resumable, but further continuation
  from this blocked artifact is not authorized by this result.
- No PPO update, reward change, mechanic-definition change, raw-recording
  mutation, observation/action-contract change, closed-loop mechanic framework,
  or Rocket League-to-RivalSim mechanic reconstruction occurred.

## Recommendation

Do not deploy the V2 checkpoint for visual/opponent testing yet. First inspect
the held-out critic maximum-drift tail and decide on a prospectively authorized
retention correction that preserves the frozen test result. Do not tune against,
rerun selection on, or weaken the existing test guard using this opened split.
