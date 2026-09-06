# Focused Blue/Orange check after +500

This is read-only diagnosis while the unchanged Direct Skills learner continues
toward the scheduled +550 evaluation. It is not a new training authority,
acceptance threshold, native rollout, or deployment claim.

## Why check

Under the same corrected match-reset method, +451 to +500 improved the five
Blue cases from 34/76 goals/concedes to 45/63. Orange went from 25/80 to 25/102.
That asymmetry merits checking inputs and sampling; five deterministic cases
per side do not establish a team/canonicalization defect.

## Results

- Eleven immutable checkpoints, +0 and every 50 through +500: Nexto-facing
  learner assignments are 49.6399% to 50.2869% Blue at these instants. +500 has
  8,224 Blue and 8,160 Orange Nexto-facing learners. Every snapshot has the
  intended fixed even Nexto slots and exact complementary Nexto player index.
- The source assigns the learner side from the active scenario's focal side
  at physical reset. Self-play trains both players. There is no fixed Blue-only
  learner or inference-only-side selection mistake in these inspected paths.
- Reusing the existing 32-world native reset fixture, CPU execution of the
  unchanged production observation builder matches the archived GPU observation
  within 5.9604645e-8 across all 182 fields.
- Rotate each world 180 degrees, swap the players, rotate vector/quaternion
  state, and remap pad cooldowns: equivalent canonical observations differ only
  in self/opponent forward-x roundoff, at most 1.1920929e-7.
- The immutable +500 policy, with matching zero recurrent history, gives the
  same deterministic joint action in all 64 player perspectives. Largest logit
  difference is 7.6293945e-6; value 1.4305115e-6; hidden 5.0663948e-7.
- Model/fixture/checkpoint unchanged; no optimizer or native simulator step.
  All calculations and model tensors are on CPU. Importing the simulator module
  enumerates Warp devices; Torch CUDA is not initialized by this diagnostic.

## Limits and decision

Checkpoint slot counts are instantaneous, not a reconstruction of cumulative
per-side decision exposure or per-role episode duration. That telemetry was not
saved at every decision. The fixture checks reset inputs, not all learned
recurrent histories, team-mirrored physics trajectories, Nexto's actor inputs,
or Nexto post-kickoff scheduling. It also does not resolve the separately
documented external RLBot wheel-state approximation.

**No obvious Rival canonicalization or side-assignment defect was found in this
bounded check. The full-match Orange regression remains unexplained.** Keep
the active learner's frozen semantics unchanged and examine both sides again
at +550. Do not call the asymmetry a proven bug or a proven sampling fluctuation.

## Reproduction

From the repo root:

```powershell
.venv/Scripts/python.exe benchmarks/audit_rival2_direct_skills_team_symmetry.py
```

The script validates the fixed +500 and fixture hashes, compares all saved
assignment checkpoints, checks fixture/model immutability, and refuses a changed
result on rebuild. Machine-readable evidence:
`team_symmetry_diagnostic_000500.json`.

Two focused transform tests exercise half-turn involution and a deliberately
missing pad remap with nonuniform synthetic cooldowns. Synthetic values are
unit-test inputs only, never presented as new gameplay observations. Existing
entity-policy tests additionally cover canonical pad geometry, attention,
recurrent reset semantics, and independent critic gradient isolation.

Validation completed: **6 focused CPU tests passed**, ruff check/format passed,
and two diagnostic builds produced identical JSON bytes. All 20 frozen runtime
source hashes still match `package.json`. Evidence SHA-256:
`F58786DE12D5EC1BA2D962EAEA0FD2D062FCCFDDD9291C7EE8C6470F3696EA04`.
JUnit evidence: `team_symmetry_tests_000500.xml`.
