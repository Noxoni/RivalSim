# Entity-aware joint-control development candidate

This implements the user's explicit entity-awareness addition within the active
SSL-development goal. It is **one neural policy**, not a specialist router.
It does not prove SSL competence. Old failed/stalled runs remain preserved.

## Input and network

The existing 182-field, team-canonical native observation is unchanged. Separate
encoders represent self, opponent, ball, 34 pads and history/lifecycle context.
Cars share their encoder with different self/opponent type embeddings. Pad
geometry comes from the native Soccar map and existing canonical remap; pad
availability/cooldown comes from the observation. No inferred mechanics, scenario
identity, future state, added sensors or target controller actions are supplied.

One 64-wide, four-head self-query attention layer reads the 38 tokens. Learned
residuals enter the actor and the existing GRU context. The independent MLP critic
remains separately trainable. Its loss has no actor/attention gradient path.
This version is 1v1: the external observation does not include extra players.

Zero-initialized residual projections preserve the joint-control candidate's
logits, values and hidden state exactly. The attention encoders receive learning
gradients after those output projections begin learning. All are trainable.

## Controls and initialization

The same eight native controls are emitted at 30Hz and held for four 120Hz ticks.
The **internal actor contract changes explicitly** to one categorical distribution
over the pinned 90-action standard lookup table. It is not the old 13-output
Gaussian/Bernoulli contract. No separate ground/air classifier or action mask is
used. Deterministic evaluation uses argmax. Endpoint analog actions are normal
for this parser, not evidence of hybrid-distribution saturation.

The immutable parent is fresh-lineage update597, SHA256
`B0B35CDAF3B3551EC667776EB99C3822F863AAA1F17A0BA2F013B5F216BD87A5`.
No V5/BC weights or Nexto weights initialize this model. Shared features, GRU
and critic copy exactly. The action heads receive a fixed preference projection
using sigma0.65 and existing mean/button logits. This is **not exact hybrid-policy
parity**, integration of action probability mass, or a BC optimizer step.
A fresh Adam optimizer is explicit for this changed parameterization; historical
moments are not reused or projected. This combined representation/optimizer
experiment cannot isolate attention's causal effect from categorical controls.

## Frozen learning and evaluation

Original fresh-30Hz reset bank, 32768 worlds, 90-decision/3-second rollouts, two
complete-sequence PPO passes, actor LR1e-4, critic LR3e-4. Same potential-only
terminal-goal reward. No new rewards or physics changes. Current-v-current, both
sides trainable. Nexto is evaluation-only for this initial development stage.
KL stays telemetry; numerical corruption triggers complete update rollback and
stop at the most recent accepted checkpoint. No KL rejection or retention loss.

Initial evidence budget: 100 accepted updates, evaluations/checkpoints at
0/10/20/50/100 and rolling saves every update. This is a review boundary within
the continuing goal, not an SSL completion claim. Evaluate every boundary on
the original deterministic acquisition/finishing/Nexto development cases, not
new easy cases. The evaluation-only interface maps table actions losslessly into
the old harness's deterministic API; training never uses that interface. It is
tested for all90 actions and keeps the old scenario/metric implementation intact.

At100 review touch coverage, no-touch behavior and scoring together against the
candidate's own initialization and immutable u597. Do not select a lucky early
result, confuse scenario goals with match win rates, or silently extend a frozen
experiment. The goal remains actual strong natural gameplay against opponents.

## Validation and evidence

`native_preflight.json` measures a full32768-world 90-step native rollout and a
complete728-sequence forward/backward without an optimizer step. It verifies
exact action/index correspondence, stored/recomputed likelihood, value accuracy,
finite gradients, untouched parent and initialized model, zero entity-branch
parity and absence of mechanics hot paths. Unit tests cover attention permutation
behavior, every input group, pad canonicalization, recurrent resets, critic
isolation, categorical PPO loss and transactional model/Adam/RNG rollback.

Two implementation-preflight errors were corrected before training: the first
backward attempted cuDNN RNN backward after an eval-mode forward; the preflight
now explicitly uses training mode. The first parity comparison projected the
same heads on different CPU/CUDA devices and detected rounding differences;
the comparison now uses the identical CPU initialization procedure for both
models. That failed report is retained, not relabeled. No trained checkpoint
was changed by either diagnostic. No safety threshold was weakened.

### Initial launch operational correction

The first proposed PPO update was rolled back before any accepted update because
the finite-state check stacked CPU Adam step-counter checks with CUDA parameter
checks. All numerical checks now run grouped by device; neither CPU counters nor
GPU moments are omitted. A CUDA sequence-update regression test exercises this
actual mixed-device optimizer and separately injects nonfinite CPU counters and
GPU moments, both of which are detected. The9-test recovery run passed.

`operational_recovery_v1.json` binds the original source hashes to exactly three
corrected source files, without changing the original training authority. The
correction, tests, failure evidence and source hashes are committed and pushed
before resuming the exact verified zero-update checkpoint. Original model and
empty optimizer parity are verified. `launch_zero_before_recovery.pt` preserves
the checkpoint bytes referenced by the initial evaluation and failure, because
rolling files are intentionally reused. `failure_initial_cpu_adam_check.json`
is historical, not evidence that a subsequently resumed healthy process failed.

The first combined test invocation also encountered an OS permission error in
the machine-wide pytest temporary directory after20 tests passed. That XML is
retained; rerunning with a new isolated external temporary directory passed all
21 focused tests. No policy or reward code was changed to address temp access.

Commands from repo root:

```
.venv\Scripts\python.exe benchmarks/validate_rival2_ssl_entity_policy.py
.venv\Scripts\python.exe benchmarks/run_rival2_ssl_entity_joint_control.py prepare
# Commit and push authority, implementation, tests, initial checkpoint and preflight.
.venv\Scripts\python.exe benchmarks/run_rival2_ssl_entity_joint_control.py verify
.venv\Scripts\python.exe -u benchmarks/run_rival2_ssl_entity_joint_control.py run
```

External state: `G:\dev\RivalSim-runs\ssl-entity-joint-control-v1`.
`STOP` requests an accepted-boundary stop. Resume requires explicit `--resume`
and `--resume-sha256` for the same candidate; model/Adam/RNG/counters resume,
while simulator episodes/hidden restart fresh as declared, not exact world-state
resumption. Never resume the old paused run automatically.

## Primary-source context

The pinned public Nexto TorchScript in `third_party/nexto` contains EARLPerceiver
and two MultiheadAttention blocks; its observation builder uses separate entity
rows. Rival's implementation is its own architecture, not copied Nexto weights.
[Necto/Nexto source](https://github.com/Rolv-Arild/Necto).
The [official RLGym training example](https://rlgym.org/Rocket%20League/training_an_agent/)
uses a lookup action parser. Those are architectural precedents, not evidence
that this candidate will acquire SSL gameplay or permission to copy their
different direct-behavior rewards.
