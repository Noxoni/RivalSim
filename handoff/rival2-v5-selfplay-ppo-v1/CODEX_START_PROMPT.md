# Rival 2.0 — V5-rooted Self-Play PPO V1

Pull latest `origin/main` and execute this task end-to-end.

Required starting lineage:

- Human BC V5 result commit: `6f87b26c85f5e413c914116a7804e9aee09b771b`
- V5 checkpoint: `checkpoints/rival2/human_bc_v5/rival2_human_bc_v5.pt`
- V5 checkpoint SHA-256: `F9100E543F48B1AD9E447179DFC2022774F039AD8D47F9FBF07359B7E1D12FE8`

This task has two parts only:

1. revise the clean 120 Hz gameplay reward to favor fast controlled possession;
2. run a long self-play PPO continuation rooted in the V5 model weights.

Do not turn this into another broad validation/research project.

---

# 1. New 120 Hz reward contract

Create a new production reward version derived from `RIVAL2_REWARD_GAMEPLAY_120_V1`, e.g.:

`RIVAL2_REWARD_GAMEPLAY_120_V2`

Preserve the existing 120 Hz observation/action/episode contracts.

The reward remains zero-sum.

## Keep unchanged

Keep these existing competitive terms:

- goal: `+10.0` to scorer / `-10.0` to conceder;
- signed ball progress toward opponent goal: existing `0.5 * deltaY / 5120` formulation;
- demolition: `+0.10`;
- small boost-pad pickup: `+0.001`;
- large boost-pad pickup: `+0.005`;
- save: `+0.75`;
- ordinary touch reward: `0`;
- first-touch reward: `0`;
- named-mechanics reward: `0`;
- generic jump/flip reward or penalty: `0`.

Do not reintroduce the quarantined inferred named-mechanics detectors.

## Remove raw-speed shaping

Set ordinary speed occupancy reward to exactly:

`0.0`

There must be no reward for merely moving quickly below supersonic.

## Remove physical boost-use occupancy shaping

Set boost-use occupancy reward to exactly:

`0.0`

Boost-pad pickup rewards remain unchanged. We do not want to pay the policy merely for holding boost.

## Increase supersonic occupancy

Change supersonic occupancy from:

`0.00005 / 120 Hz tick`

to:

`0.000075 / 120 Hz tick`

Use the existing authoritative supersonic state. Do not infer supersonic from controller inputs.

This term is competitive/zero-sum exactly like the current occupancy shaping:

`BlueSupersonic - OrangeSupersonic`

The intent is to reward actually attaining and maintaining supersonic speed without creating a direct incentive to spam flips.

---

# 2. Add physically defined controlled-possession occupancy

Add a new dense per-player control score using only authoritative physical state.

No named dribble detector, mechanic label, learned classifier, or inferred technique name is allowed.

For each player on each authoritative 120 Hz tick:

```text
car_ball_distance = length(ball_position - car_position)
relative_speed = length(ball_linear_velocity - car_linear_velocity)

proximity = clamp((500.0 - car_ball_distance) / 350.0, 0.0, 1.0)
velocity_match = clamp(1.0 - relative_speed / 1200.0, 0.0, 1.0)

control_score = proximity * velocity_match
control_reward = 0.00015 * control_score
```

Compose competitively:

`BlueControlReward - OrangeControlReward`

and set Orange reward to the negative of the complete Blue reward as usual.

Do not pay this component on a reset transition itself if the environment's reward composition would otherwise observe reset-generated motion/state.

Record telemetry for:

- mean control score;
- fraction of player-ticks with control score > 0;
- fraction with control score >= 0.25;
- fraction with control score >= 0.50;
- cumulative control reward contribution.

Telemetry is diagnostic, not a training gate.

The purpose is simple: keeping the ball physically close while matching its velocity should be valuable, including at high speed. Merely being near a ball that is rapidly escaping should pay little or nothing.

---

# 3. Relax unnecessary flip-through-ball penalty

The existing production guard currently penalizes unnecessary directional-dodge ball contacts at `-0.01` with trusted contested-50 and power-contact exemptions.

Change the penalty to:

`-0.005`

Retain both existing trusted exemptions unchanged:

- contested 50/challenge;
- dodge-powered power contact.

Add one additional purely physical exemption:

`EXEMPT_RETAINED_CONTROL`

After a candidate directional-dodge ball-contact onset, inspect the offender's physical control score during the next **12 authoritative 120 Hz ticks** after the contact.

If the offender reaches:

`control_score >= 0.20`

on any of those post-contact ticks before reset/termination, classify the contact as retained control and do not pay the bad-flip penalty.

The contested-contact association logic can retain its existing shorter timing window. The candidate only needs to remain pending long enough to resolve the new retained-control condition correctly.

Precedence should be:

1. contested 50 exemption;
2. power-contact exemption;
3. retained-control exemption;
4. unnecessary flip-through-contact penalty.

Do not add a generic flip penalty. Do not punish flips that never contact the ball.

Add telemetry counts for the new retained-control exemption.

---

# 4. Reward tests

Add focused deterministic tests proving:

- speed occupancy contributes exactly zero;
- boost-use occupancy contributes exactly zero;
- authoritative supersonic occupancy contributes exactly `0.000075/tick` before competitive subtraction;
- controlled-possession formula matches the contract at representative distances/relative speeds;
- distant ball gives zero control reward;
- close ball with large relative velocity gives little/zero control reward;
- close velocity-matched ball gets the intended reward;
- control component is zero-sum;
- reset-generated state cannot farm control reward;
- bad-flip penalty is `-0.005`;
- existing contested and power exemptions still work;
- retained-control exemption works from authoritative post-contact state;
- a flip-through contact that loses control still pays the penalty;
- named-mechanics hot path remains disabled.

Run the existing relevant reward/120 Hz regression tests. Do not build a new historical campaign validation suite.

Commit and push the reward implementation before the long training run so the training checkpoint can bind the exact reward contract/hash.

---

# 5. V5-rooted PPO bootstrap

Human BC V5 is not PPO-resumable. That is expected.

Create a fresh PPO training state using **only the exact V5 model weights** as the model initialization.

Do not load:

- a BC optimizer;
- an old PPO optimizer;
- V2/V3/V4 blocked checkpoints;
- the accidental old 10-hour PPO descendant;
- historical policy-pool state.

Initialize a fresh PPO optimizer/state.

The full model is trainable during PPO:

- shared trunk;
- actor;
- critic.

The V5 checkpoint remains immutable as the root artifact.

Record the exact source checkpoint SHA/model tensor identity in every campaign checkpoint.

---

# 6. Pure self-play only

This training phase is self-play only.

No Nexto, Wisp, historical Rival snapshot, or other opponent family may contribute PPO training samples.

Use the same current policy on both sides of the environment and train both canonical perspectives.

In the existing trainer this should correspond to no historical-opponent assignment (`historical_chance = 0`) and an empty historical pool, or an equivalent explicit pure-current-policy self-play path.

Verify on the first rollout that all trainable samples come from current-policy self-play on both sides.

This is a one-time configuration check, not a recurring acceptance framework.

---

# 7. PPO configuration

Stay on native 120 Hz and the established `RIVAL2_PPO_120HZ_V1` physical-time configuration:

- worlds: `32768` target if memory permits as already validated;
- rollout horizon: `128`;
- gamma: `0.9987476493904754`;
- GAE lambda: `0.9872585449014338`;
- clip range: `0.20`;
- value loss coefficient: `0.50`;
- entropy coefficient: `0.0`;
- max gradient norm: `0.50`;
- PPO epochs: `2`;
- minibatch size: `65536`.

Use a fresh full-policy learning rate of:

`1.0e-4`

Use the existing transactional PPO KL corruption guard:

- minibatch hard KL: `0.10`;
- completed-update mean KL: `0.05`.

A hard KL rejection must rollback that update exactly. If repeated hard rejections occur, reduce the current optimizer LR to `5e-5`; if necessary reduce once more to `2.5e-5`.

Do not add a BC-to-V5 KL leash or behavior-cloning loss to PPO. V5 is preserved separately; PPO is allowed to change the policy meaningfully.

Do not add new soft retention walls inherited from the BC campaigns.

---

# 8. Training length and snapshots

Train for **at least 500 accepted PPO updates**.

Except for unrecoverable numerical/runtime corruption, do not terminate the campaign before accepted update 500.

Snapshot every **30 accepted PPO updates**:

`30, 60, 90, ... 480, 510, 540, ...`

Also save an explicit checkpoint at **accepted update 500**, even though 500 is not divisible by 30.

Checkpoint directory:

`checkpoints/rival2/v5_selfplay_ppo_v1/`

Use clear names such as:

- `rival2_v5_selfplay_ppo_u0030.pt`
- `rival2_v5_selfplay_ppo_u0060.pt`
- ...
- `rival2_v5_selfplay_ppo_u0500.pt`
- `rival2_v5_selfplay_ppo_final.pt`

Each snapshot must be genuinely PPO-resumable and include fresh optimizer state, RNG state, accepted-update count, sample accounting, V5 lineage, and the new reward contract hash.

## Continue beyond 500

At accepted update 500, do not automatically stop.

If training remains numerically healthy, continue automatically to at least **600 accepted updates**.

At 600, continue in 30-update blocks while the recent training telemetry still shows useful development and no obvious collapse, up to a hard campaign ceiling of **750 accepted updates**.

Use the existing gameplay telemetry plus the new reward telemetry to make this operational decision. Relevant signals include goals, touches, no-touch behavior, supersonic occupancy, control score, bad-flip rate, policy KL, and action-distribution health.

Do not invent a complicated scalar acceptance score for continuation. If the evidence at 600 is ambiguous rather than clearly degrading, stop at 600 and preserve all snapshots.

---

# 9. Training telemetry

Persist concise per-update or aggregated telemetry sufficient to see what the new shaping is doing:

- accepted/rejected PPO updates;
- current LR;
- policy loss / value loss;
- completed-update KL;
- goals and conceding events;
- legitimate touches;
- no-touch resets/timeouts;
- supersonic occupancy;
- control-score statistics listed above;
- control reward contribution;
- unnecessary flip-through contacts;
- contested exemption count;
- power-contact exemption count;
- retained-control exemption count;
- action distribution summary.

This telemetry is for diagnosis and snapshot comparison. Do not turn every metric into a stop gate.

---

# 10. Post-training evaluation

Do not train against Nexto.

After the self-play campaign has completely stopped, use the existing established Rival-vs-Nexto evaluation harness to evaluate progression.

At minimum evaluate:

- untouched V5 root;
- update 150;
- update 300;
- update 450;
- update 500;
- update 600 if it exists;
- final checkpoint if different.

If it is cheap under the existing harness, evaluate all 30-update snapshots as well.

Use the established evaluation protocol/seeds rather than creating a new one.

Report enough to compare win rate, goals/score differential, touches/control telemetry, supersonic behavior, and bad-flip behavior against the V5 baseline.

Nexto evaluation is post-training measurement only. Do not resume or alter PPO based on the Nexto result in this task.

---

# 11. Scope

Do not change:

- observation contract;
- action contract;
- 120 Hz cadence;
- arena physics;
- human demonstrations;
- behavior-cloning data/splits;
- observation adapter;
- named-mechanics definitions/rewards.

Only change what is necessary for:

- the new 120 Hz reward contract;
- its focused tests/telemetry;
- V5-rooted pure self-play PPO bootstrap/training;
- snapshotting;
- post-run evaluation.

Do not use failed BC campaigns as validation authorities.

---

# 12. Git / final artifacts

Commit and push detailed evidence to `origin/main`.

Preserve:

- reward contract/version/hash;
- V5 source identity;
- training config;
- training curve/telemetry;
- snapshot manifest with SHA-256 for every saved snapshot;
- update-500 checkpoint;
- final resumable checkpoint;
- post-training Nexto evaluation summary.

## Codex final response

Return only:

- PASS / BLOCKED;
- final Git commit SHA;
- new reward version + contract SHA;
- accepted PPO updates completed;
- final LR;
- update-500 checkpoint path + SHA;
- final checkpoint path + SHA;
- snapshot count;
- V5 -> final supersonic occupancy;
- V5 -> final controlled-possession/control-score telemetry;
- V5 -> final unnecessary-flip rate and exemption breakdown;
- V5 vs Nexto result;
- best evaluated PPO snapshot vs Nexto result;
- stop reason;
- recommended next step.

Keep detailed test output, curves, hashes, and tables in Git. Do not dump a giant report into the thread.
