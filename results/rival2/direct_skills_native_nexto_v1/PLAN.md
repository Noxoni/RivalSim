# Corrected-opponent development and learning block

The previous goal turn made verified progress: the native-v5 Nexto controller
was implemented, validated at 32,768 worlds without learning, committed and
remotely read back. SSL capability remains unproven. Do not repeat the completed
bridge/timing/observation investigations or use old Nexto win rates as progress.

## Baseline first

The prospective baseline protocol was published at
`02c78cb30b7273014db3b4ceba985d7be665c675`; all four files were remotely read back
before execution. Two preserved policies get identical full development matches:
reference600 and finishing650. Both now face the opt-in native-v5 controller,
including its explicit kickoff sampling mode, with the same fixed seed.
The Rival policy itself uses raw deterministic joint90 argmax at 30 Hz.

The parent rule was frozen before seeing results: wins, then aggregate goal
difference, then fewer concedes; an exact tie favors the later finishing650.
No search over other checkpoints, threshold tuning or training during selection.
The fixed set is a development comparison, not an untouched test or native game.

## Real learning, after selection and zero-step preflight

Use the selected checkpoint's exact model, Adam moments/counters and four saved
RNG states. Do not initialize from V5, reset the model, or reuse the old Nexto
controller caches. Fresh physical episodes and zero recurrent hidden are
explicit; the new opponent has a separately seeded, checkpointed RNG.

Preserve the latest finishing-goal contract and pressure curriculum, T2
on-policy exploration, entity-aware recurrent policy, independent critic,
32,768 worlds, 120 Hz physics/30 Hz decisions, 90-decision rollout, two PPO
passes, actor LR1e-4/critic LR3e-4 and finite/corruption protection. KL remains
telemetry only. No new reward detector or mechanic definition.

Worlds remain half self-play/half corrected Nexto; only one car contributes
learner samples in Nexto worlds, so exactly one third of learner decisions face
Nexto. Source starts remain 30% natural, 20% challenge, 20% finishing, 15% defense
and 15% kickoff. The existing pressure bank retains hard finishing cases.

Train a 25-update block, checkpoint every accepted update, preserve entry/first
and permanent10/25 files, and evaluate at10 and25 with this exact corrected
opponent. The early evaluation is to expose regressions before another long run.
The frozen block ends for review; this does not redefine or complete the SSL goal.

Do not call elapsed wall time, training safety, fixture success or increased
reward demonstrated gameplay improvement. Report actual goals/concedes,
contacts, kickoff contacts, repeat-contact identities and no-contact worlds.
No-touch resets are disabled in regulation evaluation, so zero resets is not a
learned achievement. Repeat contacts are not possession-duration measurements.

## Recovery and persistence

Runner: `benchmarks/run_direct_skills_native_nexto_v1.py`.
External state: `G:/dev/RivalSim-runs/direct-skills-native-nexto-v1`.
Source/evidence identities and actual parent will be committed in
`training_authority.json` and `training_package.json` before the first step.
Resume only the exact latest accepted file/hash in this new lane. Physical
episodes restart explicitly, but optimizer and policy/opponent RNG persist.
Never clear user STOP files, restart an already completed block, or silently
rerun a partially started evaluation. Operational failures need an audit;
nonfinite/capability failures must not be bypassed by weakening safeguards.

Raw JSON evidence bytes are preserved through Git attributes because selection
and checkpoint metadata bind artifact SHA-256. Source identities normalize
line endings; baseline protocol identity is its canonical JSON content hash.

The existing monitor will be updated in place after launch, using OpenAI Docs
guidance for durable scheduled-task prompts. It will report new completed
evaluations or meaningful failures, not repeat unchanged status.
