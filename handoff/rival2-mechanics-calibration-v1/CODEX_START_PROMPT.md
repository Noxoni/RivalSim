# Codex Start Prompt — Rival 2.0 Mechanics Calibration V1

Work in `Noxoni/RivalSim`.

1. Pull the latest `main` and record its exact HEAD SHA. Do **not** reset or discard newer work.
2. Verify handoff source commit `1da8557f32a94e6a8e96d1acbb0103656e203e27` is an ancestor of your working HEAD.
3. Read `handoff/rival2-mechanics-calibration-v1/README.md` completely.
4. Read all five mechanics authority documents listed in that README completely before implementing.
5. Execute the handoff exactly as bounded: calibration and read-only detector work only. **No Rival training, no reward activation, no policy/PPO/observation/action/physics/lifecycle changes.**
6. Build deterministic RivalSim-native calibration cases for the nine continuous detectors, derive thresholds from physical separation margins, run held-out cases, run the focused source-exact regressions, then run the 256-episode zero-reward shadow gate.
7. Do not force a detector to pass. If physically justified features cannot cleanly separate positives from near-misses, mark it `NOT_READY_FOR_REWARD`, preserve the overlap evidence, and continue with the other detectors.
8. Keep thresholds binary event-identity boundaries, never execution-quality scores. A weak genuine mechanic must still classify.
9. Write all required machine-readable evidence under `results/rival2/mechanics_calibration_v1/` and the human-readable report at `docs/RIVAL2_MECHANICS_CALIBRATION_V1_RESULTS.md`.
10. Run focused tests only; do not broaden into a full simulator acceptance suite.
11. Commit and push all implementation, tests, calibration artifacts, and documentation to `main` when complete.
12. Return the exact commit SHA plus the reviewer return package specified in section 15 of the handoff README.

Hard stop conditions:

- any proposed training run;
- any non-calibration reward change;
- any policy/PPO/obs/action/physics/lifecycle contract change;
- inability to verify the source commit is an ancestor of current `main`;
- a required calibration result being fabricated rather than measured.
