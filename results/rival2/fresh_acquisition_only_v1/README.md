# Fresh acquisition-only diagnostic

User request: test whether a newly initialized Rival can learn contact acquisition
without the other scenario families. The mixed-scenario campaign was stopped at
accepted update 156; its checkpoint and Adam are preserved, not reused.

## What changes

- Fresh random model and fresh Adam, seed 2026090810. No trained checkpoint is loaded
  to initialize the experiment, including BC, V5 or sustained-gameplay weights.
- Every reset starts in the acquisition family. The same underlying 16,384 source
  cases from the earlier acquisition bank each appear in both opponent groups.
- No kickoff, shooting, defending, natural-game or other family reset is sampled.
- No curriculum retirement, automatic promotion or full-match evaluation campaign.

## What deliberately stays the same

The 182-field entity-aware recurrent actor, independent recurrent critic, joint90
action parser, 120 Hz physics / 30 Hz decisions, 90-decision rollout, two passes,
actor LR 1e-4 / critic LR 3e-4, clipping, finite checks and KL-as-telemetry behavior.
The same opponent mix: 50% current self-play / 50% active native-v5 Nexto worlds.
The same goal-first reward and seven potentials, small sustained-control payment
and inactivity penalty. A first touch is neither a new reward nor an episode end:
episodes continue to a goal or 45 seconds without either player contacting the ball.

This interprets "only the touch-acquisition scenario" as removing the other reset
families, not simultaneously replacing reward and termination with a different
first-touch task. The latter would be another experimental change, confounding
the requested comparison. Both learned self-play sides remain trainable; Nexto
is inference-only. There are no scripted controllers or demonstrations.

## Bounded observation window and evaluation

Measure the untrained baseline, then the unchanged 1,024-start deterministic
acquisition probe every 10 accepted updates, up to 150 accepted updates. This is
a diagnostic ceiling, chosen to cover the previous run's early gains and later
regression interval, not an indefinite replacement campaign. STOP and finite-state
failure end it earlier. No settings may be retuned during this experiment.

Report easy own-contact success within 5 seconds, varied within 8 seconds, and
conditional contact times. Save all first-contact tick/failure records. These are
development checks, not evidence of sustained possession, kickoff wins or SSL.
Rolling checkpoints every update and permanent checkpoints at every probe retain
fresh optimizer lineage and contracts. In-process probes do not reset training
worlds or histories. Process resume has the unchanged fresh-episode/cleared-memory
semantics and requires the exact latest same-lineage file and hash.

Improvement would demonstrate that the isolated setup is learnable. It would not
prove which part of the previous mixed setup caused regression: initialization and
scenario mix both differ. A matched fresh mixed-scenario control and additional
seeds would be needed for stronger causal attribution. Do not automatically launch
those or change the reward based on this result.

## Commands and evidence

Use project `.venv` Python with `benchmarks/run_fresh_acquisition_only_v1.py`:
`preflight`, `freeze`, `verify`, `run`, or `report --offset N`.
`run --resume PATH --resume-sha256 HASH` is recovery within this lineage only.
Sources, fresh initialization, tests and actual 32,768-world rollout/backward
preflight are committed and remotely verified before optimizer entry. Focused
tests use independent disposable models; the preflight performs zero optimizer
steps. The official initialization remains immutable.

External state: `G:/dev/RivalSim-runs/fresh-acquisition-only-v1`.
Checkpoints: `checkpoints/rival2/fresh_acquisition_only_v1`.
All older campaigns and their STOP files remain untouched and stopped.

Pre-training tests caught an attempted assignment to the immutable scenario-batch
dataclass. The new bank builder now creates a replacement batch rather than
mutating that container. The rerun passed all 20 focused tests; this occurred
before initialization/preflight or any campaign optimizer step.
