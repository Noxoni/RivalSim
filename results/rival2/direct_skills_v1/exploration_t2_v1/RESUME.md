# Temperature-only continuation: +554 to +600

Authority/runtime/preflight commit:
`8ce15e88828ac275fe7375d7919000c1218969c8`.
All 24 files were read back from fetched remote main before launch.
The separate amendment SHA256 is
`CE9E6EB5DDC1F4BD50975F47C0E1FFF7A0D3198FA9C24C254CA6A9C79B1A45D4`.
The original 20-source package remains unchanged.

The actual learner PID35488 (launcher18312) started at
2026-09-06T09:10:31Z. Verify the live PID/command line rather than relying on
this historical record. Logs are in the existing external direct-skills-v1
directory: `resume_exploration555.stdout.log` and `.stderr.log`.
The recognized agent pause marker was archived as `STOP.review550.completed`;
no user stop was removed. The existing ten-minute monitor was updated to the
new runner, separate amendment and review boundary, not duplicated. It keeps
the user's new-evaluation-only reporting preference. The OpenAI Docs workflow
was used for that scheduled-task update.

## Resume identity

Source: `checkpoints/rival2/direct_skills_v1/paused_000554.pt`, SHA256
`CF5C9D022FFB4EF18DBD728206D78A3601AB8C9128BFA0F0E9ADE31BCF961BAD`.
The strict loader preserves model tensors, Adam moments, parameter groups and
counters. All four saved RNG streams are restored. As in the original runner,
physical episodes restart and recurrent hidden state starts at zero.

The CPU observer missed the short pre-first-step checkpoint window: it first
saw555, then556 in the alternate rolling slot. These are retained observer
results, not training failures. **No direct live pre-first-step byte comparison
is claimed.** The strict loader and zero-step native preflight independently
check source state/model/Adam preservation. No learner restart was needed.

## First accepted continuation audit

Preserved checkpoint: `checkpoints/rival2/direct_skills_v1/exploration_resume_000559.pt`
SHA256 `2878F5F388697E5BCBDFEC3150A98D650A38416500A40B96E20FBCAEEE8A9DCF`.
The CPU audit passes22 checks: contiguous555–559, exact parent/amendment,
unchanged model schema/reward/native contracts/optimizer groups, finite model
and Adam, all RNG state present, exact one-third Nexto learner decisions, and
correct sample/physics/Adam accounting. No KL rejection occurred.

- Five additional updates,698 optimizer steps;131,736 cumulative Adam steps.
- 22,118,400 additional learner decisions;2,472,837,120 direct-skills total.
- Peak allocated CUDA memory16,321,776,128 bytes (about15.2GiB).
- Completed-update mean KL maximum0.0025583654; KL is telemetry only.
- Collected mean entropy555–559:0.7103,0.9307,0.9791,0.9768,0.8257.

Those are learning-health/exploration observations, not a deterministic gameplay
improvement. In particular555 starts from fresh physical episodes, so its early
goals/touches must not be compared directly with the steady554 buffer.

## Operation and review

Use the existing virtualenv Python and the **new** entry point:

```powershell
.venv\Scripts\python.exe benchmarks/run_direct_skills_exploration_v1.py verify
.venv\Scripts\python.exe -u benchmarks/resume_direct_skills_exploration_v1.py --resume <verified-amended-latest.pt> --resume-sha256 <exact-SHA256>
```

Use the [latest-checkpoint recovery guard](RECOVERY_GUARD.md), which dispatches
the frozen training implementation without changing it. Do not omit the resume
identity or use the old untempered runner for amended checkpoints. The artifact's
`model` remains compatible with the original deterministic inference class;
the additional distribution/authority metadata is mandatory for PPO resume.
No fresh optimizer, temperature-free likelihood, reward change or KL stop is
permitted as an operational retry.

At600 the runner saves the permanent checkpoint, completes the same fixed skill
and full-match evaluations, sets `stopped_for_exploration_review`, and creates an
agent review STOP. Inspect actual scoring/conceding, finishing, ongoing-ground
play and follow-ups against550/500 before another block. Do not automatically
restart, cherry-pick an older model, or call increased entropy a success.
The user's goal remains active until actual capability is demonstrated or the
user stops it; a review boundary is not a goal completion claim.
