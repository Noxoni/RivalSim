# Stationary standard kickoffs: user amendment after update 76

The user requires kickoff training to start from the stationary positions that
naturally occur, without forward-momentum assistance. This amendment supersedes
only the inherited kickoff initialization, including the source bank restored
after the temporary acquisition curriculum is retired. All other training and
evaluation semantics remain unchanged.

## Cause

`sustained_acquisition_v1.training_starts` inherited
`direct_skills_kickoff_race_v1.scenarios`, which added 250..750 uu/s of focal-car
forward velocity to half of dedicated kickoff rows before acquisition replacement.
Natural-family kickoff rows and all full-match evaluations were already stationary.
That historical curriculum was explicitly preserved at the sustained restart,
but is incompatible with this new user instruction. Historical code and evidence
remain untouched rather than relabeled as genuine standing starts.

## Correction

`standing_training_starts` restores every surviving `kickoff_indicator` row to its
exact existing `make_standard_kickoff_state` layout. This covers dedicated and
natural-family kickoff rows, before and after acquisition retirement:

- Five standard diagonal, offset-back and center-back layouts, both teams.
- Both cars have exactly zero linear and angular velocity.
- Native spawn orientation, grounded state and 100/3 starting boost.
- Centered stationary ball, clear jump/flip state and previous inputs.
- Rival controls itself from the first actionable tick.

There is no speedflip script, opening controller prefix, first-touch bonus or
opening-specific reward. The existing sustained episode continues after a touch
until an actual goal, or the existing 45-second no-contact inactivity condition.
The full-match evaluation already tests natural stationary kickoffs and is unchanged.

All non-kickoff states, family counts, focal sides, layout assignments, acquisition
starts/probes, opponents, PPO, exploration, rewards, actor/critic and runtime
contracts are unchanged. The full-bank audit verifies that `car_vel` on the
previously assisted rows is the only changed state field. The post-retirement
bank also receives this correction, preventing assistance from returning later.

## Continuation and prospective authority

Training stopped cleanly at accepted update 76. Its original rolling file was copied
without reserialization to `checkpoints/rival2/sustained_acquisition_v1/plus_000076.pt`:
SHA-256 `FBC1715C787CDD9DAFD1434D0B9DBEF9D717F9187C6858D1D5C25A3E12EF7C73`.
It contains 336,199,680 learner decisions and 11,098 Adam steps. This is not a
random restart; the learned model and complete optimizer state are preserved.

`standing_kickoff_package.json` supplements the original immutable authority and
package. It binds the exact stopped checkpoint, correction code, tests, audit and
both corrected reset-bank hashes. The modified runner checks unchanged original
sources, verifies its replaced historical source against the original implementation
commit, and verifies all current amendment bytes against `origin/main` before
optimizer entry. It cannot silently restart from +27 or another unbound old file.
New checkpoints and update records carry the amendment identity; recovery requires
that identity and the exact latest checkpoint SHA. The original authority is not
rewritten as if earlier training had used stationary kickoffs.

Process resume has the existing documented semantics: model/Adam/counters/RNG are
restored, physical episodes start fresh, both recurrent memories clear. Abandoned
episodes are not counted as failures and are not spliced into GAE. Thereafter
histories and physical games continue across updates and diagnostics as before.

## Focused verification

`standing_kickoff_tests.xml` records 21 passing tests: whole-state exact native
kickoff parity, unchanged non-kickoff data and metadata, active/retired banks,
forbidden resume/downgrade cases, actual CUDA initialization and actual goal/reset
readback before the next action, plus existing sustained reward, GAE, memory,
retirement and disposable PPO regressions. Disposable tests use separate random
models and do not update the accepted checkpoint.

`standing_kickoff_audit.json` verifies every kickoff row in both full 32,768-world
source banks and records old/new assisted-row counts and speed. No new task reward,
mechanic detector or solution controller is introduced. This verification establishes
correct starts, not that the policy has learned to win kickoffs.

Run/resume remains `benchmarks/run_sustained_acquisition_v1.py run --resume PATH
--resume-sha256 HASH` with the exact latest rolling file. The monitor retains its
two-minute schedule and completed-evaluation-only reporting.
