# Fresh Human Seed with no previous-action input — final report

## Verdict

**BLOCKED — the validation-selected imitation checkpoint did not demonstrate
functional closed-loop gameplay. PPO was not started.**

## Selected Stage-1 checkpoint

- Path: `checkpoints/rival2/fresh_human_seed_no_previous_action_v1/rival2_fresh_human_seed_no_previous_action_v1.pt`
- SHA-256: `DE1B16086405744FCB2FF23FB7384FDD1AB0273384E33E1E1EB2314083271DDE`
- Model-tensor SHA-256: `6843A0EE311560CFCE14E267B4F29AED3A166B17457E506C61C304471DEAFE97`
- Initialization: new random Rival 2 model, seed 2,026,090,106
- Prior model checkpoint loaded: no
- Selected supervised step: 200
- Steps executed before the frozen plateau stop: 5,000
- Stop: `validation_plateau_after_minimum_5000_steps`
- Baseline validation complete-action RMSE: 0.6113542782419022
- Selected validation complete-action RMSE: 0.41044817576431636
- Untouched test complete-action RMSE: 0.3884424160849771
- Test evaluations after selection: exactly one

The validation metric reached its minimum at step 200, then worsened while training loss
continued to fall. The checkpoint at step 200 was preserved throughout and selected only
by the frozen held-out validation complete-action RMSE.

## Input and training integrity

- Exactly 58,306 gameplay frames were loaded; mechanic-practice frames loaded: zero.
- Split remained chronological 46,644 / 5,831 / 5,831.
- All 58,306 samples had the eight `previous_action.*` fields zeroed and marked
  unavailable before Adapter V2, then hard-zeroed after adapter reconstruction and pad
  overlay.
- The selected policy config independently hard-zeros the same indices inside policy
  `forward`, immediately before the shared trunk.
- The critic received zero Stage-1 optimizer steps.
- No reward, retention, old-policy KL, mechanic label/objective, simulator-state
  preservation objective, BC/PPO parent, or PPO optimizer was used.

## Deterministic 120 Hz closed-loop evaluation against Nexto

The selected checkpoint was loaded directly. Rival used deterministic tanh actor means
and Boolean thresholding at 120 Hz. Gaussian and Bernoulli sampling were both disabled.
The evaluation used 256 paired standard-kickoff episodes, balanced across Rival side and
five kickoff layouts.

| Metric | Result |
|---|---:|
| Completed episodes | 256 / 256 |
| Rival episodes with a touch | 0 / 256 |
| Rival physics touch onsets | 0 |
| Rival first touches | 0 |
| Rival challenge possession exchanges | 0 |
| Rival contacts accelerating the ball toward the opponent goal | 0 |
| Rival possessions moving the ball toward the opponent goal | 0 |
| No-touch resets | 0 |
| Rival goals | 0 |
| Nexto goals | 256 |
| Mean Rival speed | 1,164.7689114924142 uu/s |

The absence of no-touch resets is not positive Rival evidence: Nexto made 1,262 touches
and scored in all 256 episodes. Rival moved, but never contacted the ball, never won a
first touch, never completed a challenge exchange, and never moved the ball toward the
opponent goal. Functional gameplay was therefore not demonstrated.

## Evaluation bookkeeping correction

The first evaluation trace is retained as
`deterministic_nexto_closed_loop_invalid_lifecycle.json` and excluded from the verdict.
It refreshed both the 120 Hz action and the legacy four-tick lifecycle interval every
tick, preventing valid terminal bookkeeping. The focused evaluation-only correction
decoupled 120 Hz observation/action refresh from the unchanged four-tick lifecycle
interval. It did not alter, retrain, or reselect the checkpoint and did not inspect the
human test split. The corrected trace is `deterministic_nexto_closed_loop.json`; all 256
episodes terminated by a recorded goal.

## Next step

Do not start PPO from this checkpoint. Investigate why the chronological human trajectory
supports low one-step held-out action RMSE yet fails under deterministic closed-loop state
distribution shift. Per the task instruction, no workaround or additional training was
attempted here.
