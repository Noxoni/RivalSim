# Rival 120 Hz human-demonstration dataset V1

## Scope and authority

This lane is a read-only adapter and frozen split. It performs no PPO, behavior cloning,
optimizer step, reward change, mechanic detection, or model mutation. Its immutable review
authority is commit `08a55e8326940271689847b316fa096a7fed3c71`, read directly with
`git show` rather than from mutable working-tree copies.

The native session files remain the source of truth. The committed dataset manifest stores
their per-file byte sizes and SHA-256 hashes, the aggregate source-file-set hashes from the
authority review, and Git identities for the authoritative review artifacts. The builder
refuses a missing, changed, corrupt, or manifest-hash-invalid source session.

## Frozen cohorts and split policy

The manifest freezes all 110 high-confidence mechanic behavior-cloning candidate attempts
from review V2. The split unit is the entire attempt: no attempt is divided by frame, and
source ranges are checked for overlap. Assignment is deterministic, per mechanic label, and
based on a versioned SHA-256 ordering seed. Common labels reserve approximately 10% each for
validation and test; labels with three to nine candidates reserve one complete attempt for
each; a two-attempt label reserves one for validation; a singleton remains in train.

All 85 nonpositive reviewed mechanic attempts (71 failed and 14 ambiguous) are retained by
source UUID, label, frame range, and verdict, but are explicitly excluded from the initial
BC-positive cohort.

The complete 58,306-frame `nexto_1v1` session is frozen as general gameplay, not as named
mechanics. It is partitioned into whole regions at recording start, native sequence/physics
discontinuities, kickoff/round resets, local-car rebinds, and respawns. Complete regions are
assigned deterministically to train/validation/test. Split changes therefore occur only at a
recorded hard boundary; neighboring ordinary frames never cross a split.

## Adapter contract

`ReadOnlyTrajectoryAdapter` exposes both ordered complete trajectories and individual
samples. Each source frame produces:

- one exact, read-only `float32[8]` `RIVAL2_ACTION_V2_120HZ` target in contract order;
- a read-only `float32[182]` partial `RIVAL2_OBS_V2_120HZ` vector;
- a read-only 182-field exactness mask;
- a complete observation only when every field is source-exact;
- the exact source sequence used for `previous_action`, when one exists; and
- explicit blocked field names and reasons otherwise.

No action is averaged, subsampled, interpolated, combined, or temporally reduced. For the
first frame of each requested span the adapter reads the preceding native source frame when
it is contiguous and no lifecycle boundary intervenes. It excludes `previous_action` only
at a true recording/lifecycle/discontinuity boundary and never fabricates it.

Arrays returned by the adapter are read-only. The source dictionaries and native recording
files are never mutated.

## Exact-observation gate

The committed native recordings do **not** contain enough information to reconstruct a
complete exact 182-field Rival observation. Consequently the adapter intentionally emits no
complete supervised `(observation, action)` pair: `observation` is `None`, unresolved partial
entries are `NaN`, and `usable` is false. This is a source-contract block, not a split or
serialization failure.

The recurring blockers are:

- all 34 canonical boost-pad identities, active flags, and cooldown timers;
- RivalSim-only per-car state/timer semantics (`is_jumping`, `has_flipped`, `is_flipping`,
  `dodge_available`, demolition timer/state, jump/air/flip/boosting timers);
- Rival policy memory (`time_since_boosted` and `sticky_ticks`) and exact supersonic time;
- exact episode-age and no-touch-age origins when recording begins mid-lifecycle;
- every opponent and opponent-relative field in single-car Freeplay recordings;
- a small number of frames whose native wheel records repeat index zero instead of uniquely
  identifying all four wheel positions; and
- only at real boundaries, the prior action and pre-boundary event history.

Native boost, component, event, and wheel data are preserved and remain useful for future
work, but the authority review did not prove their semantics identical to the corresponding
RivalSim internal fields. The adapter therefore does not promote approximate mappings to
exact ones. Adding placeholders, zero-filled pads, a synthetic opponent, inferred timers, or
a fabricated prior action would violate this contract.

## Balanced-sampling interface

`sampling_metadata.json` exposes counts and buckets in the hierarchy
`split -> mechanic label -> whole attempt -> frame`. A later trainer can choose between
natural frame sampling and mechanic-aware label/attempt sampling. No oversampling
coefficients or aggressive reweighting are selected here.

## Rebuild and verification

From the repository root with the project virtual environment:

```powershell
.\.venv\Scripts\python.exe benchmarks\build_rival2_human_demo_dataset_v1.py
.\.venv\Scripts\python.exe benchmarks\build_rival2_human_demo_dataset_v1.py --verify-only
.\.venv\Scripts\pytest.exe -q --basetemp .tools\pytest_human_adapter `
  tests\test_human_demo_training_adapter.py tests\test_human_demo_recording.py
.\.venv\Scripts\ruff.exe check rivalsim\human_demo\__init__.py `
  rivalsim\human_demo\training_adapter.py `
  benchmarks\build_rival2_human_demo_dataset_v1.py `
  tests\test_human_demo_training_adapter.py
```

Override `--source-root` only to point at another byte-identical copy of the reviewed native
sessions. The output deliberately records a relocatable UUID-based source locator rather than
freezing a machine-specific absolute path.

## Committed artifacts

The canonical evidence is under `results/rival2/human_demo_dataset_v1/`:

- `dataset_manifest.json`: authority, source hashes, positive and excluded cohorts, splits,
  source ranges, and per-trajectory adapter scan results;
- `adapter_audit.json`: contract hashes, no-learning audit, exact action totals, blockers,
  and fail-closed verdict;
- `statistics.json`: split, label, attempt, region, frame, and usability counts;
- `sampling_metadata.json`: later-trainer sampling interface without selected weights;
- `artifact_manifest.json`: deterministic artifact hashes; and
- `verification_evidence.json`: focused test, lint, source, split, deterministic-rebuild,
  and remote-persistence evidence for this frozen build.

The manifest and adapter implementation are valid and deterministic. The training adapter is
not BC-ready until a prospective source/observation contract supplies every missing exact
field or the observation contract is explicitly changed under separate authority.
