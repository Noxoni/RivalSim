# Human Sequence Seed v1 — final report

## Verdict

**BLOCKED — the validation-selected recurrent imitation checkpoint did not
demonstrate functional closed-loop gameplay. PPO was not started.**

## Selected Stage-1 checkpoint

- Path: `checkpoints/rival2/human_sequence_seed_v1/rival2_human_sequence_seed_v1.pt`
- SHA-256: `B77A059ECB31DE59A964FE2A368F40C3F367DE0C028E29325F0FF6F763BAF292`
- Model-tensor SHA-256: `B9E4E2D285723FFC88CB08F9B54C7EA3FBE71FE101F326FB105313E655A9D4F1`
- Initialization: fresh random recurrent model, seed 2,026,090,201
- Prior checkpoint loaded: no
- Selected supervised step: 400
- Steps executed: 2,400
- Stop: `validation_complete_action_rmse_plateau`
- Baseline validation complete-action RMSE: 0.6421332171098845
- Selected validation complete-action RMSE: 0.4426745468066241
- Untouched test complete-action RMSE: 0.41012141192787555
- Test evaluations after selection: exactly one

Checkpoint ranking used only the lowest validation complete-action RMSE. The
step-400 checkpoint was preserved while training continued through the frozen
plateau window.

## Addendum alignment and input integrity

- The kickoff/sequence addendum at `afa4afb36fea60fd566d22487e42ec138c51b3e3`
  was applied before the first optimizer step.
- The 58,306-frame source produced 20 whole playable segments and 41,732
  supervised frames. Frozen countdown and post-goal frames were excluded from
  loss and recurrent burn-in.
- The whole-segment chronological split is 32,363 train / 4,772 validation /
  4,597 untouched test frames.
- The shared 182-field `RIVAL2_HUMAN_SEQUENCE_OBS_VIEW_V1` retains 71 direct
  physical fields and hard-zeros 111 fields in both domains, including every
  `previous_action.*` field. Observation Adapter V2 is not used.
- Projected human/native matched-state retained-field RMSE is
  8.078202995420725e-09; maximum absolute error is 5.960464477539063e-08.
- The critic received zero optimizer steps. No old model weights, mechanic
  demonstrations, reward optimization, retention objective, or PPO were used.

## Deterministic 120 Hz closed-loop evaluation against Nexto

The exact selected checkpoint was evaluated for 256 balanced standard-kickoff
episodes. Rival used tanh actor means plus thresholded button logits at 120 Hz,
with no action sampling. Recurrent hidden state started at zero on every native
playable kickoff and remained continuous within the episode.

| Metric | Result |
|---|---:|
| Rival touches | 52 |
| Episodes with a Rival touch | 52 / 256 |
| Rival first touches | 0 |
| Rival challenge possession exchanges | 52 |
| Rival forward-velocity contacts | 52 |
| Rival forward-displacement possessions | 0 |
| Rival goals | 0 |
| Nexto goals | 256 |
| No-touch truncations | 0 |
| Mean Rival speed | 1,437.4989403889047 uu/s |

Zero no-touch truncations is not positive Rival evidence: Nexto scored in every
episode before that timeout. Rival moved and occasionally contacted the ball,
but it never won the first touch, established forward ball displacement, or
scored. Functional closed-loop gameplay was not demonstrated.

## Five zero-hidden native kickoff outputs

| Layout | throttle | steer | pitch | yaw | roll | jump | boost | handbrake |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| diagonal left | 0.095364 | 0.061471 | 0.021886 | 0.026992 | 0.041906 | 0 | 0 | 0 |
| diagonal right | 0.235132 | 0.101335 | -0.021108 | 0.067182 | 0.041325 | 0 | 0 | 0 |
| off-center left | 0.158176 | 0.110093 | 0.016281 | 0.067870 | 0.046043 | 0 | 0 | 0 |
| off-center right | 0.160278 | 0.106953 | 0.016119 | 0.066352 | 0.045600 | 0 | 0 | 0 |
| center | 0.172415 | 0.108016 | 0.028140 | 0.070189 | 0.042466 | 0 | 0 | 0 |

## Evaluation runtime correction

The first closed-loop launch stopped before gameplay because a recurrent hidden
tensor created under PyTorch inference mode was reset outside inference mode.
The evaluator-only correction performs the same zero reset under inference mode.
It did not modify or reselect the checkpoint, run an optimizer, inspect the
human test split, or change simulation behavior. The complete deterministic
evaluation then ran successfully.

## Next step

Do not start PPO from this checkpoint. Diagnose why the aligned recurrent model
still emits weak zero-hidden kickoff controls and fails to turn held-out sequence
imitation into closed-loop ball acquisition; do not infer success from RMSE alone.
