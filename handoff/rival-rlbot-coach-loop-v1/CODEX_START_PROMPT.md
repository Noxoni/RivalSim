# Rival — RLBot Human Coach Loop v1

## Start here

This task **extends the separate RLBot play package that is currently being built for Rival**.

Before implementing anything:

1. Fetch the latest `origin/main`.
2. Confirm that the completed RLBot package for playable Rival is present and identify the exact runtime checkpoint/model it loads.
3. If that RLBot work is not present yet, **STOP** and report only that this handoff is waiting on the RLBot package. Do not recreate, replace, or independently redesign the RLBot integration.
4. Rebase/merge this handoff onto that completed RLBot work, then implement from there.

This handoff was authored from main commit:

`8f5ace7d3c99c824bdd10ee1769973f0bf17a50a`

At that point the current deployment candidate is:

`checkpoints/rival2/unified_capability_distillation_v5/rival2_unified_capability_v5.pt`

SHA-256:

`955C93BF538BC913CC2E42F42E3B0EDC4CCDB1065DA9581FB88D84C363B7C216`

V5 is one recurrent unified policy with one recurrent hidden state and no runtime router. However, **the completed RLBot package is the runtime authority**. If that package intentionally loads an equivalent exported artifact or a newer approved descendant, use exactly what it loads rather than creating another deployment choice.

---

# Goal

Turn playable RLBot Rival into a simple human-coaching loop:

> **play Rival → capture native RLBot-domain human demonstrations/corrections → short recurrent supervised fine-tune → load child checkpoint → play again**

This is an operator workflow, not a new research campaign.

The user should be able to play normal Rocket League matches with Rival through RLBot, collect useful human action targets in the exact runtime observation domain, train briefly between matches, and then immediately play the updated child checkpoint without rebuilding Rival.

Do not start PPO in this task.

---

# Core design

## 1. Keep the existing RLBot package intact

Do not make a second RLBot bot, second observation stack, second action adapter, or second launcher unless the existing package has no extension point at all.

Hook this feature into the **actual policy inference boundary** used by the completed RLBot package.

The tensor recorded as `observation` must be the exact 182-float tensor presented to the deployed policy after all RLBot-side preprocessing/projection/normalization.

The action layout is the existing eight-channel Rival controller:

`throttle, steer, pitch, yaw, roll, jump, boost, handbrake`

Record at the **actual policy inference cadence**. The target is native 120 Hz if the RLBot package runs the policy at 120 Hz. If the runtime delivers a different real cadence, record those actual policy ticks. **Never invent 120 Hz samples by interpolation or duplicated frames.**

---

## 2. Normal sparring should produce human demonstration data

The primary v1 workflow is a normal user-vs-Rival RLBot match.

While Rival runs normally, maintain a second **shadow policy state for the human player's perspective**:

- use the same exact observation builder/runtime preprocessing as deployed Rival;
- swap self/opponent perspective correctly so the observation means “what Rival would see if it occupied the human player's car”;
- maintain an independent recurrent hidden state for this human-perspective shadow;
- reset that hidden state using the same real reset/kickoff semantics as the deployed policy;
- run the same parent model deterministically in shadow mode every policy tick;
- capture the user's actual controller input aligned to that same policy tick as the supervised target.

This gives us human demonstrations directly from the live Rocket League/RLBot domain rather than reconstructing them later through Observation Adapter V2.

The user must be able to play normally. Do not require a special replay editor or offline reconstruction to collect the basic training data.

### Human controller capture

Use the smallest reliable input path available in the completed RLBot package/environment. If it already exposes the local player's controls, reuse it. Otherwise add a lightweight controller reader local to the coach feature.

Sample the controller at the policy tick and convert it once into the exact eight-channel Rival action layout. Record both the raw/normalized input metadata needed for diagnosis and the final eight-channel target.

Do not alter the user's controls or interfere with the normal human car while recording.

---

## 3. Do not recreate the old previous-action teacher-forcing bug

This is mandatory.

Current Rival contracts may include `previous_action.*` in the structural 182-field observation. Never populate the human-perspective training observation with the **previous human controller action** and then ask the model to predict the current human action.

Instead, for the human-perspective shadow policy:

- maintain the shadow model's **own previous deterministic action**;
- use that model action for any runtime `previous_action.*` semantics required by the deployed observation path;
- record the human action only as the supervised target;
- never feed previous human actions into the policy as an imitation shortcut.

If the completed RLBot package already hard-zeros previous-action fields, preserve that behavior. Do not re-enable them.

During training, preserve the same rule. Prefer sequentially regenerating any active `previous_action.*` slots from the train-time model's own detached previous prediction as the sequence is replayed. Never substitute the previous human target.

---

## 4. Record append-only sessions

Create one append-only session directory per played match/session using a simple format such as:

`data/rival_rlbot_coach/<session_id>/`

Use repository conventions if an equivalent data root already exists.

Each session must contain a manifest identifying at least:

- session UUID/id and timestamp;
- Git commit;
- RLBot package/runtime version if available;
- exact parent checkpoint path and SHA-256;
- model/config identity;
- observation/action contract identities available from the runtime;
- real policy cadence observed;
- player/team/car mapping;
- number of total policy ticks;
- number of valid human-target ticks;
- reset boundaries;
- whether the session completed cleanly.

For every policy tick needed by the human-perspective training sequence, persist at minimum:

- tick/sequence index;
- exact runtime-domain observation physical/preprocessed values needed to reconstruct the policy input;
- `reset_before` or equivalent real recurrent reset marker;
- shadow model action;
- human target action;
- target-valid mask;
- relevant timing/cadence metadata.

You may also record Rival's live policy observation/action trace in the same session because it is useful context, but **Rival's own bot actions are not supervised training targets**.

Bot-only samples may carry recurrent context. They must not silently become imitation labels.

Use a compact binary format (`npz`, tensor chunks, or an existing project format) plus a small JSON manifest. Do not build a database service.

---

# Optional focused correction mode

Add a lightweight `coach`/demonstration mode only if it can reuse the completed RLBot package without disrupting normal sparring.

The useful v1 behavior is simple: run a Rival-controlled slot in a mode where the user's controller can drive that slot while the exact Rival policy observation path continues to run in shadow and records human targets.

This lets the user deliberately demonstrate a kickoff, recovery, aerial approach, possession sequence, etc. from the Rival perspective.

Do **not** make exact rewind/state restoration of an arbitrary prior live-game mistake a dependency of v1. If the current RLBot package already has reliable game-state restore support, it may be reused, but do not turn this task into a replay-engine project.

The normal user-vs-Rival human-perspective capture is sufficient for the core acceptance path.

---

# Training: short correction fine-tune

Add one small supervised trainer for the captured RLBot-domain human sequences.

## Parent

The parent is always the exact checkpoint currently active in the RLBot package when the session was recorded.

Never silently train a session against a different parent lineage. If multiple sessions have different parents, group them by compatible parent or require an explicit selection.

Preserve the parent checkpoint unchanged and write a child checkpoint.

## Recurrent replay

Train sequentially. Do not shuffle independent frames and destroy recurrence.

Reconstruct recurrent hidden state from real reset boundaries and carry it forward through the sequence. Truncated BPTT/chunking is fine, but hidden state must carry across contiguous chunks and detach only where appropriate.

Only human-target ticks contribute supervised action loss. Non-target ticks may be replayed to establish temporal context but have zero imitation loss.

Do not initialize hidden state independently at every human-target window unless there is a real recorded reset there.

## Previous-action handling

If the deployed model consumes previous-action fields, regenerate those slots autoregressively from the model's own detached previous deterministic prediction during training. Do not use previous human targets.

All other observation values should come from the exact RLBot-domain capture path rather than Adapter V2 reconstruction.

## What to train

For the current V5-style unified model, default to the smallest useful correction:

- freeze the existing feed-forward/base trunk, base actor, and critic;
- train the recurrent context path (`context_encoder`, recurrent context/GRU, and context actor/residual) only;
- critic receives zero optimization steps;
- do not add reward, KL retention, PPO, or a second teacher policy.

If the completed RLBot package uses a different approved recurrent class, adapt this mechanically to its equivalent recurrent/action-producing parameters while keeping the same intent: **small supervised correction of the active recurrent policy, not a wholesale retraining campaign**.

Use the existing differentiable hybrid-action imitation conventions where available:

- analog targets train the analog action outputs;
- button targets train button logits with a differentiable binary objective;
- log-std/exploration outputs are not human controller targets.

Keep defaults intentionally short — one or a few passes over the newly selected session data at a conservative learning rate. Expose the step/epoch count as a command-line option. Do not create a validation bureaucracy around each coaching iteration.

If there are zero valid human target ticks, exit cleanly and do not create a fake trained checkpoint.

---

# Promotion / reload loop

The operator loop must be fast.

Use the existing RLBot launch/config mechanism and add the smallest checkpoint-selection mechanism necessary so that a child checkpoint can become the next active Rival without rebuilding the package.

A simple local active-checkpoint pointer/config file is acceptable.

Required behavior:

1. play/record a session;
2. run the coach trainer on the latest or explicitly selected session(s);
3. trainer writes a child checkpoint + manifest that names its parent and training data;
4. activate that child for the next RLBot launch (or perform a safe between-match reload if the current package already supports it);
5. preserve an easy way to point back to the parent if the user dislikes the result.

Do not hot-swap network weights in the middle of an active recurrent trajectory unless the existing runtime already has a safe reset/reload mechanism. Between matches is sufficient.

---

# Operator surface

Keep this simple. After inspecting the existing RLBot package, expose no more than roughly these operations through its existing CLI/config style:

- **play / spar + record** — normal user-vs-Rival match with human demonstration capture;
- **coach** — optional deliberate human-control demonstration mode if practical with the existing package;
- **train** — short supervised fine-tune from latest or selected captured sessions and activate the child;
- **rollback/status** — show active parent/child and restore the previous parent checkpoint.

Do not build a new GUI for v1.

Document the exact commands in a short operator README after implementation.

---

# Explicitly out of scope

Do not:

- reimplement or replace the in-flight RLBot package;
- change the deployed observation/action contract just to make coaching easier;
- use Observation Adapter V2 to reconstruct newly captured RLBot training inputs;
- feed previous human actions into the policy;
- train on Rival's own mistake actions as imitation labels;
- invent or interpolate missing 120 Hz frames;
- add a replay database/service;
- build a large UI;
- redesign rewards;
- start PPO;
- add named-mechanic reward/detector work;
- create a broad new evaluation framework;
- reopen old failed campaigns or regression suites;
- turn this into another full-model training campaign.

Fix only concrete implementation blockers encountered in this loop.

---

# Minimal acceptance

Implementation is complete when all of the following work on top of the completed RLBot package:

1. Rival still launches and plays through the existing RLBot package.
2. A normal human-vs-Rival session can record a human-perspective sequence using the **same RLBot runtime observation path** and correctly aligned eight-channel human controller targets.
3. The capture proves that previous human controller actions are not being fed back as `previous_action.*` imitation inputs.
4. Recurrent reset/sequence boundaries are preserved.
5. The trainer loads a recorded session, applies loss only to human-target ticks, and produces a child of the active checkpoint without modifying the parent.
6. The next RLBot launch can load that child without rebuilding the bot package.
7. Parent rollback works.
8. No PPO/reward campaign is started.

If a physical human/controller session is unavailable inside the Codex execution environment, validate the wiring with the smallest deterministic input fixture necessary, but **do not fabricate human training data and call it a real coaching result**. Finish the implementation so the user's next actual match creates real data immediately.

---

# Commit artifacts

Commit the implementation to `origin/main` after rebasing onto the completed RLBot package.

Include only the useful artifacts:

- RLBot capture/coach integration;
- compact session format + manifest code;
- recurrent supervised coach trainer;
- active-checkpoint / rollback mechanism;
- short operator README with exact commands;
- minimal tests/smokes needed to prove action alignment, previous-action non-leakage, recurrent reset handling, and child checkpoint load.

Do not commit ordinary user match recordings by default unless the repository already has a deliberate data-artifact convention for them. Keep runtime coaching sessions local and record their hashes/metadata in child checkpoint manifests.

---

# Execution rule

This is an implementation task, not a proposal document.

Once the RLBot package is present, implement the loop end-to-end and run the smallest available smoke proving capture → train → child load.

Do not wait for another architecture review. Do not add unrelated gates. If a real blocker appears, fix that blocker or stop with the specific blocker.

---

# Final response

Keep the Codex return short:

- `STATUS: PASS` or `STATUS: BLOCKED`
- final commit SHA
- active parent checkpoint
- child checkpoint path + SHA-256 if a child smoke was created
- exact operator commands for play/record, train, and rollback/reload
- real human target ticks captured, if a human session was available
- one-line blocker or next step
