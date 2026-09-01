# Rival 2.0 — Human Sequence Seed v1

## Authority

Execute this task from the latest `origin/main` containing diagnostic commit:

`f24f7506a68e9d094aabbb1cea85af3c299368a3`

This is a **new lineage** named **Human Sequence Seed v1**. It is not BC V6/V7/etc.

The prior no-previous-action Stage-1 result is diagnostic history only. Do not continue it and do not load its weights.

## Why this task exists

The focused diagnostic established two concrete blockers:

1. Human-demo observations and native RivalSim observations are materially different. The largest differences are lifecycle/kickoff semantics and Adapter V2 reconstructed car state; the same physical state changed the selected policy's deterministic action by complete-action RMSE ~0.153.
2. Removing `previous_action.*` exposed temporal ambiguity. The recording contains nearly identical physical observations with materially different human actions, so a memoryless one-frame policy is insufficient.

Fix those two blockers directly. Do not add unrelated process or preservation machinery.

---

# Goal

Train a **fresh randomly initialized recurrent Rival policy** from the reviewed 58,306-frame full-match human gameplay trajectory so that:

- the policy sees the **same deterministic observation view** during human imitation and native RivalSim execution;
- no human previous controller action is ever provided as input;
- the policy gets temporal context from its own recurrent hidden state rather than from `previous_action.*`;
- checkpoint selection is based only on held-out human action imitation;
- the selected checkpoint is then run deterministically against Nexto before any PPO work is allowed.

**Do not start PPO in this task.**

---

# Immutable source data

Use only the existing reviewed full-match gameplay recording:

- session UUID: `CD6E7DB1-2761-4B8B-BD37-F21C7F135722`
- source frames: `58,306`
- gameplay label: `nexto_1v1`

Do not load mechanic-practice demonstrations.

Do not load any previous Rival, BC, Human Seed, or PPO model weights.

Use a new deterministic initialization seed and record it in authority/results.

---

# Part 1 — One shared observation view

Create a frozen observation-view contract named:

`RIVAL2_HUMAN_SEQUENCE_OBS_VIEW_V1`

Keep the structural Rival observation size at **182 floats**, but make the policy consume only fields that can be represented consistently from both the native recording and RivalSim.

## Required retained information

Retain these categories when they can be deterministically mapped to the exact existing Rival normalization:

### Ball
Retain all 9 ball fields:
- position xyz
- linear velocity xyz
- angular velocity xyz

### Self and opponent car
Retain, for both cars:
- position xyz
- linear velocity xyz
- forward xyz
- up xyz
- angular velocity xyz
- boost amount, using a direct deterministic recorder-to-contract normalization rather than Adapter V2 estimation
- on_ground
- has_jumped
- has_double_jumped
- four wheel-contact flags
- is_supersonic
- jump_available when it is deterministically derivable from recorded native state

### Relative physical state
Retain all 12 relative ball/opponent position and velocity fields.

## Required zeroed information

Hard-zero these fields in **both** domains:

- all 8 `previous_action.*` fields
- all 68 boost-pad active/cooldown fields for this lineage
- all 7 lifecycle fields for this lineage
- car fields whose human value is only approximate/unavailable and cannot be deterministically mapped to the same native Rival value, including timing/proxy fields and ambiguous dodge/flip state

The purpose is not to maximize the number of nonzero inputs. The purpose is:

> the same physical state must produce the same policy input regardless of whether it came from the human recording or native RivalSim.

Do **not** run Observation Adapter V2 over the policy input for this lineage. Adapter V2 is the source of a demonstrated train/deploy domain difference and must not estimate missing state here.

If an intended retained field does not match between the direct human reconstruction and a matched native RivalSim state, either fix the deterministic conversion if the source semantics support it or zero that field in both domains. Do not invent a learned repair.

Implement the projection/mask in one shared code path or frozen contract and enforce it immediately before the recurrent policy encoder so native RivalSim cannot accidentally expose extra fields.

Also apply the same projection to the materialized human observations before training.

## Minimal parity check

Before training, rerun the existing matched-state diagnostic logic on representative kickoff/early-play states using **the projected observation view**.

For every retained field, human and native values must agree to normal float/reconstruction tolerance. Any remaining disagreement must be resolved by a deterministic semantic correction or by removing that field from the retained view.

This is a preflight for the actual fix, not a new regression framework. Record the final retained/zeroed field list and aggregate matched-state RMSE.

---

# Part 2 — Recurrent policy

Do not modify the behavior of the existing feed-forward Rival policy used by historical campaigns.

Implement a separate recurrent policy for this lineage, e.g. `rivalsim/rival2_recurrent_policy.py`.

Use this architecture unless a direct repository constraint requires a mechanically equivalent implementation:

1. projected 182-dim observation
2. `Linear(182, 512)` + SiLU encoder
3. one-layer `GRU(input_size=512, hidden_size=512)`
4. `Linear(512, 512)` + SiLU post layer
5. actor head with the existing 13-channel Rival hybrid layout
6. critic head with one scalar output for future PPO compatibility

Initialization should follow the spirit of the existing Rival policy: stable orthogonal initialization, small actor/critic output gains.

During Stage 1:
- train encoder + GRU + post layer + action-producing actor parameters
- critic receives zero optimizer steps
- exploration/log-std parameters are not human action targets; keep them fixed at a documented sane value for later use

The recurrent hidden state is the only temporal memory. The model must not consume prior human controller inputs.

Reset hidden state at true episode/kickoff reset boundaries and at explicit dataset split boundaries.

---

# Part 3 — Sequential human imitation

Preserve the existing chronological split boundaries:

- first 80% train
- next 10% validation
- final 10% untouched test

Do not randomly scatter adjacent individual frames across splits.

Train as a sequence problem, not independent shuffled frames.

Use truncated backpropagation through time with bounded contiguous windows. A reasonable implementation is:

- 256-tick contiguous sequence windows
- up to 64 ticks of preceding context/burn-in when available inside the same episode/split
- compute the supervised action loss on the non-burn-in portion
- never cross a true reset or split boundary with hidden state

Equivalent sequential batching is acceptable if it preserves the same semantics.

The supervised target is still the exact eight-channel recorded human action:

`throttle, steer, pitch, yaw, roll, jump, boost, handbrake`

Use a straightforward differentiable imitation objective over the deterministic action prediction. Do not introduce retention, simulator preservation, mechanic labels, reward, old-policy KL, or any old Rival teacher.

## Selection

Checkpoint ranking must be **only**:

`lowest held-out validation complete-action RMSE`

No secondary score may override a lower validation action RMSE.

Train long enough to establish a real validation plateau. Preserve the best checkpoint while continuing through the plateau decision.

Open the untouched final 10% test exactly once after checkpoint selection and report complete-action RMSE.

Also report train/validation curves and per-channel errors as diagnostics, but they do not alter selection.

---

# Part 4 — Deterministic closed-loop evaluation

After Stage-1 checkpoint selection, do **not** start PPO.

Load the exact selected recurrent checkpoint into native RivalSim using the same `RIVAL2_HUMAN_SEQUENCE_OBS_VIEW_V1` projection.

Run deterministic actions:
- analog = actor means through tanh
- buttons = learned logits thresholded at 0.5
- no Gaussian sampling
- no Bernoulli sampling

Maintain recurrent hidden state continuously through each episode and reset it only at the same real episode/kickoff boundaries defined for the policy.

Evaluate against Nexto using the existing deterministic 256-episode / balanced-standard-kickoff evaluation approach from the prior focused check.

Report at minimum:
- Rival touches
- episodes with at least one Rival touch
- first touches/challenge exchanges if available
- Rival and Nexto goals
- mean Rival movement speed
- no-touch truncations with correct interpretation
- five initial kickoff action vectors

This is the decisive Stage-1 playtest.

If the model is still obviously nonfunctional in closed loop, stop and report the behavior. Do not begin PPO and do not invent another training campaign.

If it demonstrably participates in gameplay—approaches/challenges the ball, produces meaningful touches, and attempts to progress or score—preserve the exact selected checkpoint and report it as the candidate human seed for the next PPO task.

Do not create an arbitrary numeric acceptance wall beyond reporting the actual behavior. The purpose is to see whether the bot genuinely plays.

---

# Forbidden in this task

- no previous human action as observation input
- no Observation Adapter V2 inference in the final Human Sequence observation path
- no mechanic-practice data
- no old Rival/BC/Human Seed/PPO weights
- no retention loss
- no policy-preservation KL objective
- no simulator-state preservation objective
- no reward optimization during imitation
- no PPO
- no Nexto/Wisp/historical samples during imitation
- no named-mechanics objectives
- no broad new validation/test families beyond what this task directly requires

---

# Repository artifacts

Use a new lineage directory, for example:

- `results/rival2/human_sequence_seed_v1/`
- `checkpoints/rival2/human_sequence_seed_v1/`

Commit:
- observation-view contract/implementation
- recurrent policy implementation
- training runner
- minimal tests for observation projection, recurrent reset/shape behavior, and deterministic action interface
- frozen source/split authority
- training curve
- selected checkpoint
- untouched-test result
- deterministic Nexto closed-loop result
- concise final report/artifact manifest

Large transient run state may remain outside Git as in existing campaigns, but the selected checkpoint and required evidence must be committed using existing project conventions.

Push the completed work to `origin/main`.

---

# Execution rule

This is an implementation **and actual training/evaluation task**. Do not stop after writing code or tests.

Run the Stage-1 recurrent imitation campaign end-to-end and run the deterministic closed-loop Nexto evaluation before returning.

If a real implementation/runtime blocker occurs, fix the smallest identified blocker and continue. Do not manufacture additional gates or unrelated process.

---

# Final response format

Keep the final response short:

- `STATUS: PASS` or `STATUS: BLOCKED`
- final commit SHA
- selected checkpoint path + SHA-256
- validation RMSE
- untouched test RMSE
- projected human/native matched-state RMSE
- deterministic Nexto summary: touches, episodes-with-touch, goals for/against, mean speed
- whether functional closed-loop gameplay was demonstrated
- one-line next step
