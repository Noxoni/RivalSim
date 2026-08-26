# Rival 2.0 Nexto Port + Full-Match Runner

## Authority and sequencing

This handoff is authorized by the user as the next Rival 2.0 milestone after the already-active final-45B behavioral trajectory evaluation.

Starting repository state for this authorization:

- `Noxoni/RivalSim`
- parent HEAD before this handoff: `df295da1bcaec07170465f22fdc512b66fdd7538`
- final Rival checkpoint: `checkpoints/rival2/overnight/rival2_overnight_final_6h_resume.pt`
- expected Rival checkpoint SHA-256: `4DC158DC2A9D16B79FB5FE7D868E3B50928AB113B55DFCC753F3734F8D87372E`
- Rival policy version: 5,403
- Rival cumulative agent decision samples: 45,323,649,024

First finish the existing `handoff/rival2-behavioral-eval/README.md` evaluation if it has not already been completed in the current lineage. Publish that evidence, then continue directly into this handoff without requesting another approval.

Do not change Rival training, reward, PPO, model, observation, action, or simulator physics as part of this milestone.

## Mission

Port the public Nexto policy into RivalSim as a faithful, GPU-native, batched external policy; build a reusable 120 Hz full-match policy scheduler/runtime; and run the first full 5-minute Rival-vs-Nexto benchmark.

The implementation must be reusable later as an external fixed opponent for training, but **do not train Rival against Nexto in this milestone**.

The desired runtime is conceptually:

`RivalSim GPU state -> Rival obs -> Rival policy @ 30 Hz -> Rival controls`

and, in the same worlds:

`RivalSim GPU state -> Nexto obs -> Nexto model @ 15 Hz -> Nexto lookup action -> Nexto controls`

with Nexto's stock hard-coded kickoff controller operating at its original 120 Hz cadence.

Everything after initialization should remain device-resident unless a final evidence export intentionally copies compact results to the host.

---

# 1. Pin the exact public Nexto opponent

Use the public upstream repository exactly at:

- repository: `Rolv-Arild/Necto`
- commit: `2e6ed7d6ed2b352e8ff529d4a12a0c9c70c28cca`

Relevant upstream files at that commit:

- `rlbot-support/Nexto/nexto-model.pt`
  - Git blob SHA: `1eb5de5660a1fe00860289cb6a17938985094196`
  - size: `1,852,625` bytes
- `rlbot-support/Nexto/agent.py`
  - Git blob SHA: `44b0abbbf7e43820aadf474179c3507a2c48efe5`
- `rlbot-support/Nexto/nexto_obs.py`
  - Git blob SHA: `9e92de91e77ca23ed47bf1f8e99d5a861ee93203`
- `rlbot-support/Nexto/bot.py`
  - Git blob SHA: `f0ccdd03e0139ad636642079c69cf972a83c2d1e`
- upstream `LICENSE`
  - Git blob SHA: `cbe5ad1670406e4402217edfb82d2c56af7e8631`
  - license text identifies CC BY-NC-SA 4.0.

Preserve upstream attribution and license information in an isolated `third_party/nexto/` area or equivalent. Record the exact downloaded model SHA-256 in RivalSim evidence. Do not silently replace the public model with another Nexto checkpoint.

Do not modify the Nexto model weights.

---

# 2. GPU-native Nexto adapter

Implement a reusable batched Nexto policy adapter whose hot path is CUDA-resident.

## 2.1 Model

Load the pinned public `nexto-model.pt` TorchScript actor and run it on the RivalSim CUDA device in eval/no-grad mode.

If the old TorchScript artifact requires compatibility work under the current PyTorch version, solve that without changing the policy's numerical behavior or weights. The final adapter must not fall back to per-world CPU inference.

## 2.2 Exact 1v1 observation semantics

Reproduce the public Nexto observation semantics from `nexto_obs.py` for the 1v1 case directly from RivalSim tensors.

The adapter must construct the same logical `q`, `kv`, and mask inputs that the public model receives, including:

- self car entity;
- opponent car entity;
- ball entity;
- all 34 boost entities;
- Nexto's exact entity ordering;
- exact position / linear-velocity / forward / up / angular-velocity feature layout;
- boost amount semantics;
- demo flag semantics;
- on-ground semantics;
- `has_flip` semantics;
- previous action in the query;
- orange-side inversion;
- Nexto normalization constants;
- Nexto relative-coordinate rotation.

For `has_flip`, match `rlgym-compat==1.0.2` / Nexto-era semantics rather than inventing a new definition. The intended semantic is:

`not has_flipped AND not has_double_jumped AND air_time_since_jump < DOUBLEJUMP_MAX_DELAY`

Use Nexto's exact hard-coded boost locations/order. RivalSim's pad storage order must be mapped to Nexto's order by verified coordinates; do not assume the orders are identical without checking.

Nexto's boost entity availability feature must match the upstream observation builder exactly.

Do not route the production hot path through NumPy or `rlgym_compat` objects.

## 2.3 Exact Nexto action table

Reproduce `Agent.make_lookup_table()` exactly on device.

Validate and record:

- exact number of discrete actions;
- exact action-table contents;
- a stable content hash.

The public benchmark mode uses Nexto `beta=1`: deterministic argmax over the public model logits, then the exact lookup-table action.

The resulting eight controller channels are passed directly to RivalSim:

`throttle, steer, pitch, yaw, roll, jump, boost, handbrake`.

## 2.4 Native policy cadence

Outside kickoff:

- Rival remains native 30 Hz: new action every 4 physics ticks;
- Nexto remains native 15 Hz: new neural action every 8 physics ticks;
- each policy's last action is held between its own update ticks.

Do not force Nexto to 30 Hz and do not force Rival to 15 Hz.

## 2.5 Stock Nexto hard-coded kickoff

Preserve Nexto's public hard-coded kickoff controller from `bot.py` exactly.

Important: this sequence is advanced at **120 Hz**, not at Nexto's 15 Hz neural cadence.

In a 1v1 kickoff Nexto is the kickoff taker. Reproduce the upstream `KICKOFF_CONTROLS` / `KICKOFF_NUMPY` sequence exactly and apply it per physics tick while the kickoff controller is active. Resume the ordinary 15 Hz policy scheduler when the kickoff controller is finished / no longer applicable.

Record a stable hash of the ported kickoff control sequence and validate it against the pinned source sequence.

---

# 3. Targeted fidelity validation for the Nexto port

This is one of the few places where targeted validation is mandatory because a bad adapter would invalidate every benchmark and any later training against Nexto.

Do not turn this into general RivalSim release ceremony.

Required targeted checks:

1. **Observation parity**
   - compare the GPU adapter against a direct NumPy/source-faithful reference on a broad set of sampled 1v1 RivalSim states;
   - include both Blue and Orange Nexto viewpoints;
   - include grounded, aerial, demoed, low/high boost, pad-active/pad-inactive, and all five kickoff layouts;
   - report elementwise max absolute error separately for `q`, `kv`, and mask;
   - target exact or float32-close parity (`<= 1e-6` where ordinary floating arithmetic permits).

2. **Model/action parity**
   - run the same observation batches through the pinned model in the source-faithful reference path and the CUDA adapter;
   - report logit error;
   - require 100% deterministic argmax action-index agreement on the parity corpus.

3. **Action-table parity**
   - exact table contents and row order.

4. **Kickoff-sequence parity**
   - exact controller sequence and length.

5. **No timed hot-path host transfers**
   - after initialization, ordinary batched Nexto inference + action selection must not require per-step H2D/D2H state copies.

If any of these fail, fix the adapter before running the official matchup.

---

# 4. Reusable 120 Hz full-match runtime

Build a match runtime separate from the frozen Rival2 training episode semantics.

The current training contract intentionally ends/resets on goals, 15-second no-touch truncations, and the 45-second hard episode limit. Those rules must **not** control the full-match runner.

Do not change the frozen training behavior to accomplish this.

## 4.1 Scheduler

The full-match engine advances RivalSim physics at 120 Hz and maintains per-world controller state.

At each physics tick:

- advance/override Nexto's hard-coded kickoff controller if active;
- update Rival neural action only on its 4-tick cadence;
- update Nexto neural action only on its 8-tick cadence when not overridden by kickoff;
- hold each policy's last controller state otherwise;
- step RivalSim one physics tick.

The runner may optimize safe portions later, but correctness of mixed 30 Hz / 15 Hz / 120 Hz behavior comes first.

## 4.2 Match state

Maintain a device-resident per-world match state with at least:

- regulation ticks remaining;
- Blue score;
- Orange score;
- overtime flag;
- match-done flag;
- winner;
- goal count;
- current kickoff layout / kickoff state;
- per-policy scheduler counters/state.

Existing RivalSim lifecycle score state may be reused if it remains correct across goal kickoffs and is not reset accidentally.

## 4.3 Regulation and overtime

For this benchmark, define one match as:

- 5:00 of simulated active regulation time = 36,000 physics ticks at 120 Hz;
- goals update the score and trigger the normal RivalSim kickoff reset, but **do not terminate the match**;
- the 15-second no-touch truncation is disabled;
- the 45-second training episode limit is disabled;
- regulation ends at the first runner boundary at/after the 36,000th active physics tick;
- if the score is not tied, the match ends;
- if tied, enter overtime from a fresh kickoff and end on the next goal.

The current milestone does not need Rocket League's zero-second airborne continuation rule. Record this simplification explicitly in the evidence so it is not confused with exact RL match-clock semantics.

## 4.4 Goal reset behavior

Use the accepted RivalSim kickoff reset machinery to reset cars, ball, boost pads, demo state, and per-kickoff state after a goal while preserving the **match score and match clock**.

Do not use the Rival2 training `reset_mask` to reset a world merely because its old 15-second or 45-second truncation condition fired.

## 4.5 Rival observation during full matches

The frozen 45B Rival policy still receives its existing `RIVAL2_OBS_V1` only. Do not add score or time to the policy for this benchmark.

Maintain the existing observation fields (including previous action, kickoff indicator, no-touch/episode age semantics) in a coherent way across each goal-to-goal segment, but no full-match training-contract change is authorized.

## 4.6 Reusable policy interface

Structure the match scheduler so a fixed external policy such as Nexto can later be assigned to selected worlds/opponent slots without rewriting the simulator.

The Nexto adapter should expose a batched action interface and own only Nexto-specific observation/cadence/kickoff state.

Do **not** integrate Nexto into PPO/self-play training yet.

---

# 5. Match telemetry

Reuse the behavioral-trajectory telemetry from the preceding evaluation where practical.

For Rival and Nexto separately, publish at least:

- goals;
- goals per match;
- touches;
- touch share;
- kickoff first-touch wins;
- kickoff goals;
- next-touch possession / opponent-handoff rates;
- touch direction / ball displacement categories from the toucher's canonical perspective;
- wall/backboard continuation where available from the behavioral evaluator;
- goal-entry X/Z placement and goal-mouth distribution;
- demos;
- regulation wins;
- overtime wins;
- total wins/losses;
- average and median goal differential.

Also preserve every canonical match scoreline and starting kickoff layout/side assignment.

Do not label backward, lateral, wall, or backboard touches intrinsically bad.

---

# 6. Official first Rival-vs-Nexto benchmark

After the adapter parity gates and match runner are working, run two clearly separated evaluation suites.

## 6.1 Canonical deterministic deployment suite — primary result

Use:

- frozen final 45B Rival checkpoint;
- Rival deterministic action (`tanh(mu)`, button probability >= 0.5);
- pinned public Nexto with `beta=1`;
- stock Nexto hard-coded kickoff;
- full 5-minute match runtime above.

Run exactly the canonical side/layout matrix:

- 5 possible starting kickoff layouts;
- Rival as Blue and Rival as Orange;
- total: **10 full matches**.

Because both policies and RivalSim are deterministic, do not pretend repeated identical copies of these ten trajectories provide independent statistical evidence. Publish all ten exact scorelines.

## 6.2 Stochastic Rival robustness suite — secondary result

Run a larger batched suite with:

- Nexto unchanged at deterministic `beta=1` + stock kickoff;
- Rival using its ordinary stochastic hybrid action sampling;
- fixed reproducible evaluation RNG seeds;
- side assignments balanced 50/50;
- starting kickoff layouts balanced across all five layouts.

Target **4,096 full matches** if memory/runtime is reasonable. If 4,096 simultaneous worlds is not practical with Nexto's attention model, use the largest clean power-of-two batch that fits without changing either policy, and record the selected size/reason.

This secondary suite is explicitly a policy-distribution robustness measurement, not the headline deployment matchup.

No hyperparameter tuning or retries based on who is winning.

---

# 7. Performance/evidence

Record for the finished Nexto + match stack:

- Nexto observation-build throughput;
- Nexto model inference throughput at the chosen batch size;
- full mixed-policy match simulation throughput;
- peak CUDA memory;
- timed-loop H2D/D2H state transfer bytes;
- exact Rival checkpoint SHA;
- exact pinned Nexto upstream commit;
- exact Nexto model SHA-256;
- exact Nexto action-table hash;
- exact Nexto kickoff-sequence hash.

Publish human-readable results under something like:

`docs/RIVAL2_NEXTO_RESULTS.md`

and machine-readable evidence under:

`results/rival2/nexto/`

Include a compact match ledger for all ten canonical matches and aggregate statistics for the stochastic suite.

---

# 8. Hard boundaries

This milestone authorizes:

- completing the already-authorized behavioral trajectory evaluation if still pending;
- importing/pinning the public Nexto model and required provenance/license assets;
- implementing a faithful GPU-native Nexto observation/action adapter;
- implementing exact Nexto stock kickoff behavior;
- building the reusable full-match scheduler/runtime;
- targeted Nexto fidelity tests;
- running the official Rival-vs-Nexto benchmark suites;
- publishing evidence.

This milestone does **not** authorize:

- training Rival against Nexto yet;
- changing Rival rewards;
- removing/changing ball-progress shaping;
- changing Rival PPO;
- changing Rival policy architecture;
- changing `RIVAL2_OBS_V1` or `RIVAL2_ACTION_V1`;
- starting entity-attention Rival 2.1/3.0 work;
- building the viewer;
- beginning v0.6;
- general release/regression/lint ceremony unrelated to this port.

Do not stop for approval between the behavioral evaluation, Nexto port, targeted parity work, match-runner build, and official matchup unless there is a genuine blocker that makes faithful Nexto execution impossible.

When complete, commit all implementation/evidence, push `origin/main`, and return the final commit SHA plus the behavioral-eval summary and Rival-vs-Nexto results.