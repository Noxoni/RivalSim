# Actual training entry and first accepted update

This is learning-entry evidence, not a gameplay promotion. The corrected
development baseline remains poor: finishing650 lost all ten matches, scoring
7 and conceding 196. See `BASELINE.md`; legacy-controller wins are not comparable.

## Prospective authority and launch

Training authority and the selected baseline were committed and pushed at
`05c18d111950c4e89131a9055214bdbbc3eaa400`. All sixteen commit files were read
back from GitHub. Raw JSON SHA-256 values were additionally checked against
remote bytes before launch. Authority identity:

`00634163128D126033C6C7F0626380638A75A4BD2AAD3A8FC26FC6106FCEE001`

The hidden runner started at 2026-09-06 14:37:20 UTC. Launcher PID40600 and
actual Python worker PID43740 were observed; these are historical identities,
not a substitute for checking the current process and campaign state.

Command: `.venv/Scripts/python.exe -u -B benchmarks/run_direct_skills_native_nexto_v1.py run`.
External state: `G:/dev/RivalSim-runs/direct-skills-native-nexto-v1`.

This block starts from the exact finishing650 model and Adam, not fresh
weights. Shooting/finishing remains **20% of scenario starts**, alongside
30% natural play, 20% challenge, 15% defense and 15% kickoff. All reward,
architecture, PPO and scenario settings remain the frozen parent settings.
The opponent implementation changes to the validated native-v5 controller.

## Entry integrity

`entry_integrity.json` confirms exact model, full Adam state/groups, optimizer
counters, four saved RNG states and source checkpoint hash at entry. Physical
episodes and recurrent hidden state begin fresh, as explicitly documented in
the authority; this is not exact physical-world continuation.

Immutable entry: `checkpoints/rival2/direct_skills_native_nexto_v1/entry_000000.pt`.
SHA-256: `209474F84AB899C79F2ECF429718DD4BAEDD7FE017CA56CF7C8B11397F4D7F9E`.

## First actual accepted update

Immutable first checkpoint:
`checkpoints/rival2/direct_skills_native_nexto_v1/first_000001.pt`.
SHA-256: `2A9223B5BD0E1514AD2A673A44778C3E74FD2D3983178AB29BA711312DAE8DF2`.

The CPU-only audit in `first_update_audit.json` passes all21 checks, including
changed model weights, finite model and Adam tensors, correct lineage and
unchanged PPO/reward contracts. The audit itself constructs no optimizer and
takes zero optimizer steps. Its first-record read avoids racing the live
append-only log tail.

- Child offset1 / cumulative651; Adam step144636 ->144772:136 actual steps.
- 4,423,680 trainable decisions; 1,474,560 face corrected Nexto (one third).
- 11,796,480 physical world ticks; 2,195 real goals and matching resets.
- Rollout15.599s; PPO33.086s, measured for this update only.
- Completed-update mean KL0.00227305; maximum sample KL7.273745.
  KL is telemetry only: zero KL rejections, not a claimed safety failure.
- Training telemetry:23.400 contacts/min and mean movement speed1200.491uu/s.
  These scenario/exploration measurements are **not** deterministic match results.

Three focused entry tests passed (zero failures/errors) in `entry_tests.xml`.
The prior36 focused tests and zero-step GPU preflight remain preserved in
`training_tests.xml` and `training_preflight.json`; they were not overwritten.

## Running state and next review

At 14:42:31 UTC, five updates were accepted, the actual worker was alive and
optimizing, and stderr was empty. This observation is not the final block result.
Read `campaign_state.json` and `latest.json` for current accepted progress.

The existing `monitor-corrected-ssl-ppo-v2` heartbeat was updated in place to
follow this exact run. It reports only new completed evaluations or meaningful
problems; the baseline is already marked reported. The new notification cursor
lives in the external run directory. It must not resume a legacy runner.

Next deterministic corrected-controller comparison is at child10, then25.
The block stops at25 for review, preserving the latest accepted checkpoints.
No SSL, native-gameplay recovery or finishing breakthrough is claimed here.
