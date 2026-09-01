# Rival 2.0 — Fresh Human Seed Training v1

Execute this task end-to-end from latest `origin/main`.

This is a NEW LINEAGE. It is **not BC V6**. Do not name/version/report it as BC V6.

Goal:
1. fresh randomly initialized Rival;
2. supervised imitation using only the reviewed 58,306-frame full-match gameplay recording;
3. select strictly by lowest held-out human action RMSE;
4. start fresh self-play PPO from that selected checkpoint;
5. evaluate before/after PPO against Nexto.

Existing V1-V5 BC and V5-rooted PPO artifacts are history only. Preserve them but do not load or continue them.

## Stage 1: fresh supervised imitation

Use only the already-reviewed full-match gameplay sequence with exactly **58,306 paired native-120-Hz state/action frames**. Exclude every mechanic-practice recording.

Reuse the existing frozen Observation Adapter V2 only as preprocessing needed to express recorded state in the current 182-field `RIVAL2_OBS_V2_120HZ` contract. It supplies no behavioral target and remains frozen.

Exact action target order:
`throttle, steer, pitch, yaw, roll, jump, boost, handbrake`.

Create `Rival2ActorCritic(Rival2PolicyConfig())` from fresh deterministic random/orthogonal initialization. **Do not load model weights from any previous Rival, BC, PPO, acquisition, gameplay, Nexto, or Wisp checkpoint.**

Train the shared 3x512 SiLU trunk plus actor analog-mean rows and button-logit rows. Do not train the critic in Stage 1.

The five actor log-std outputs are not human labels. During Stage 1 set actor rows 5:10 weight to 0 and bias to `-1.0`, and prevent gradients from modifying those rows. Unfreeze them normally for PPO.

### Dataset split
Freeze a temporal, non-overlapping split of the 58,306 frames:
- first 80% train;
- next 10% validation;
- final 10% untouched test.

Record exact boundaries and hashes. Shuffle only training frames. Never optimize on validation/test.

### Objective
Prediction is:
`[tanh(mu[0:5]), sigmoid(button_logits[0:3])]`.

Optimize only:
`mean((predicted_8_channel_action - recorded_8_channel_action)^2)`.

No reward objective. No mechanic-label objective. No simulator retention. No KL to an old policy. No teacher. No hard-tail mining. No simulator-state replay. No historical-policy sampling. No old-policy log-std target.

Mechanic labels must not affect sampling, loss, eligibility, checkpoint ranking, or stopping.

### Optimizer
Fresh AdamW:
- LR `3e-4`;
- weight decay `1e-5`;
- betas `(0.9, 0.999)`;
- epsilon `1e-8`;
- grad clip `1.0`;
- batch `4096`.

LR may be reduced if held-out RMSE genuinely stalls; minimum `1e-5`. Do not create a tuning campaign.

### Selection/stopping
Every 100 optimizer steps evaluate the complete validation split.

**The only checkpoint ranking metric is complete-action validation RMSE. Lowest wins.** Nothing else contributes.

Target: about `0.30` validation RMSE or better.

- maximum 30,000 optimizer steps;
- no plateau stop before 5,000;
- after 5,000: plateau = 30 validation checks without RMSE improvement >= `0.0001`;
- if RMSE reaches `<=0.30`, continue until 10 validation checks without >=`0.0001` improvement, then stop;
- if RMSE is still improving, continue toward the ceiling.

NaN/Inf or a genuinely broken optimizer may stop training. Do not invent extra behavioral gates.

Save selected Stage-1 checkpoint:
`checkpoints/rival2/fresh_human_seed_v1/rival2_fresh_human_seed_v1.pt`

Checkpoint provenance must state fresh initialization and no prior Rival checkpoint loaded.

After selection, evaluate untouched human test once. Do not reopen selection afterward.

## Stage 2: self-play PPO

Bootstrap only from the selected Stage-1 model tensors and policy config.

Before PPO:
- reinitialize critic head fresh;
- fresh PPO optimizer/RNG/counters;
- unfreeze full trunk + full actor including log-std + critic;
- do not load any historical optimizer/RNG/counters.

Reuse current `RIVAL2_REWARD_GAMEPLAY_120_V2` on `main` exactly; do not redesign it. This is the already-agreed reward with raw speed occupancy removed, boost-use occupancy removed, supersonic `0.000075/tick`, controlled possession up to `0.00015/tick`, uncontrolled flip-through `-0.005` with contested-50/power-contact/retained-control exemptions, and named-mechanics reward zero.

Use existing `RIVAL2_PPO_120HZ_V1` configuration and initial PPO LR `1e-4`.

Opponent regime: **current-policy self-play only**. Nexto probability 0, Wisp 0, historical Rival 0. Both current-policy sides trainable.

Run **600 accepted PPO updates**. Count only accepted updates.

Save resumable snapshots at:
`30,60,90,...,480,500,510,540,570,600`.

Update 500 is explicit even though it is not divisible by 30.

Snapshot directory:
`checkpoints/rival2/fresh_human_seed_v1/ppo/`

A transactionally rejected PPO update may roll back and use the existing bounded LR backoff. Do not turn one rejection into a new research project.

## Final evaluation
After update 600, using the existing native 120-Hz evaluation harness under identical conditions:
1. evaluate the selected Stage-1 human-seed checkpoint against Nexto;
2. evaluate PPO update 600 against Nexto;
3. report the direct comparison.

Do not train against Nexto in this package. Human RMSE after PPO may be diagnostic only; it is not a PPO gate.

## Scope
Do not change physics, obs/action contracts, Observation Adapter V2, raw recording, Gameplay 120 V2 reward values, or named-mechanics rewards.

Do not create retention machinery. Do not compare Stage-1 eligibility to old Rival checkpoints. Do not require mechanic labels to improve. Do not use prior failed campaigns as acceptance tests.

Intended lineage:
**fresh random Rival -> gameplay imitation -> self-play PPO**.

Evidence root:
`results/rival2/fresh_human_seed_v1/`

Keep only useful evidence: source/split manifest, Stage-1 curve + selected/test metrics, checkpoint hashes, PPO curve, snapshot manifest, update-600 hash, and pre/post-PPO Nexto evaluation.

Commit and push completed work to `origin/main`.

## Final response
Return only:
- PASS/BLOCKED;
- final commit SHA;
- Stage-1 checkpoint path + SHA;
- Stage-1 steps executed + selected step;
- best validation RMSE;
- untouched human-test RMSE;
- whether ~0.30 was reached;
- PPO accepted updates;
- update-600 checkpoint path + SHA;
- Nexto result before PPO;
- Nexto result after PPO;
- stop reason/blocker;
- recommended next step.

Detailed metrics stay in Git.