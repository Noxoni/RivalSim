# Rival2 Gameplay V3 bounded PPO smoke (479 to 489)

Status: `GAMEPLAY_V3_BOUNDED_PPO_SMOKE_COMPLETE`

This is the real ten-update mixed-opponent Gameplay V3 continuation. It stopped exactly at update 489. No reward, classifier, simulator-physics, PPO, network, observation, action, or curriculum coefficient was changed.

## Identity

- Training implementation commit: `c564af9d31aa618c6eb39cf10a5451bd2cabc9ff`.
- Evaluation implementation commit: `d18da26af095802adea575eb3dcfaddd45ba654e`.
- Source checkpoint SHA-256: `3B994E118A9498713DC6115D38F061958A900EA8F4D00CE568F916942E851D9A`.
- Final checkpoint: `checkpoints/rival2/gameplay_v3_smoke/rival2_gameplay_v3_iteration_489_resume.pt`.
- Final checkpoint SHA-256: `10D97428B3F1CC2E307040314D1DD1A924BD82975D4B88C0F73C3FC2716DCF54`.
- Final iteration/policy: `489` / `489`.
- Final sample counter: `3,711,438,222`.

## PPO safety by accepted update

| update | proposals | accepted | early stop | LR start | LR end | backoffs | retries | max minibatch KL | mean KL | retention KL |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 480 | 6 | 3 | yes | 0.0001000 | 0.0000250 | 2 | 3 | 0.014951 | 0.009243 | 0.019326 |
| 481 | 13 | 10 | yes | 0.0001000 | 0.0000250 | 2 | 3 | 0.017789 | 0.005110 | 0.019325 |
| 482 | 7 | 4 | yes | 0.0001000 | 0.0000250 | 2 | 3 | 0.018291 | 0.018175 | 0.019393 |
| 483 | 7 | 4 | yes | 0.0001000 | 0.0000250 | 2 | 3 | 0.006524 | 0.006200 | 0.019404 |
| 484 | 173 | 172 | no | 0.0001000 | 0.0000500 | 1 | 1 | 0.009779 | 0.006153 | 0.013791 |
| 485 | 174 | 172 | no | 0.0001000 | 0.0000250 | 2 | 2 | 0.018230 | 0.004413 | 0.004677 |
| 486 | 174 | 174 | no | 0.0001000 | 0.0001000 | 0 | 0 | 0.013516 | 0.011620 | 0.007938 |
| 487 | 172 | 172 | no | 0.0001000 | 0.0001000 | 0 | 0 | 0.019690 | 0.007125 | 0.008635 |
| 488 | 174 | 172 | no | 0.0001000 | 0.0000250 | 2 | 2 | 0.013709 | 0.004356 | 0.004284 |
| 489 | 173 | 172 | no | 0.0001000 | 0.0000500 | 1 | 1 | 0.014103 | 0.005416 | 0.005905 |

All checkpoints rearm the policy/shared-trunk LR to `1e-4`; the critic remains at `3e-4`. Every accepted step passed the soft 0.02 minibatch and retention budgets. The four early stops (480-483) are normal soft-budget exits. No hard 0.10 minibatch or 0.05 completed-update guard fired.

## Reward scale across all ten training rollouts

- Mechanics / absolute gameplay reward: `0.001494`.
- Bad-flip penalty / absolute gameplay reward: `0.003841`.
- Mechanics / progress: `0.012822`.
- Bad-flip / progress: `0.032969`.
- Maximum single-rollout mechanics/gameplay ratio: `0.074068`.
- Maximum single-rollout bad-flip/gameplay ratio: `0.058729`.

Update 480 began from a fresh kickoff population and had exactly zero progress, so its mechanics/progress ratio is undefined; the raw ledger retains the zero denominator instead of using it as behavioral evidence. Across all ten rollouts, both new terms remain far below one percent of absolute gameplay reward.

## Controlled 479 versus 489 shadow

The primary comparison uses the exact iteration-479 assignments, frozen opponent snapshots, and RNG context with only the hashed iteration-489 model substituted. Both source and policy checkpoints remained byte-identical and no PPO update ran.

| metric | iteration 479 | iteration 489 | relative change |
|---|---:|---:|---:|
| Touches/min | 20.073142 | 19.432768 | -3.19% |
| Flip-active touches/min | 12.634272 | 12.291598 | -2.71% |
| Unnecessary contacts/min | 6.073602 | 5.703018 | -6.10% |
| Unnecessary / flip-touch fraction | 0.480724 | 0.463977 | -3.48% |
| Mechanics/progress | 0.013925 | 0.013405 | -3.73% |
| Bad-flip/progress | 0.027342 | 0.026777 | -2.07% |

Rival goal share moved from `0.535270` to `0.538136`. Goal-ended episode fraction moved from `0.941406` to `0.921875`. No-touch truncations moved from `0.0` to `2.0` of 256 episodes.

The policy used slightly fewer flip-active ball contacts and also converted a larger share of the remaining flip contacts into legitimate cases. Therefore the observed mechanism is **both**, not only fewer flips or only relabeling/conversion.

Mechanics did not collapse broadly: raw detected events moved from 851 to 812 and paid events from 848 to 812; redirects, speedflips, half-flips, and car resets were maintained or increased, while pogos, successful dashes, pinches, and ball resets declined in this short natural sample. Touch acquisition declined 3.19%, which is a modest regression to monitor, not a collapse.

## Recommendation

`CONTINUE_GAMEPLAY_V3_TRAINING`

PPO remained stable, both V3 terms stayed small, unnecessary flip-through rate and fraction declined, and legitimate mechanics/exemptions remained active. Continue only under the existing gates while monitoring the modest touch-rate, no-touch, and selected mechanics declines.

Do not treat this smoke as authorization to continue beyond iteration 489 under this task. A future continuation should preserve the same safety gates and monitor touches/min, no-touch endings, successful-dash/pogo retention, and goal-ended episode fraction closely.
