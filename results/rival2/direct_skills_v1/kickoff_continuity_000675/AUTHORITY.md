# Prospective675 corrected-runtime kickoff continuity diagnostic

Run only after worker20016 has exited the completed650->675 block. Keep its
agent675 review STOP and preserved checkpoint. No training or optimizer.

Source: `checkpoints/rival2/direct_skills_v1/plus_000675.pt`, SHA256
`77E6EF076F7073549F45168FA6FB34BE9A2D4C25E1580F17F7434A59195B32F8`.
Comparison authority: existing `full_match_000675.json` and the unchanged root
runtime package, with `RIVAL2_STANDARD_KICKOFF_CONTACT_CACHE_RESET_V2` semantics.

Question: under the already corrected reset runtime, are the initial and later
same-layout/side kickoff observations, hidden reset and first actions equivalent?
What subsequent differences exist within the first eight policy decisions?
This is motivated by the opening/later disparity documented in675review, which
predates the latest curriculum. It does not establish a new reset defect.

Use unchanged `benchmarks/trace_rival2_direct_skills_kickoff_reset.py`:

```powershell
.venv\Scripts\python.exe benchmarks/trace_rival2_direct_skills_kickoff_reset.py --update 675 --output G:/dev/RivalSim/results/rival2/direct_skills_v1/kickoff_continuity_000675/trace
```

The existing exclusive GPU lease and owned evaluation stream are mandatory.
Ten original fixed matches,36,000regulation ticks,existing bounded overtime.
Only copy the first32physics-tick decision inputs/output and native state after
each kickoff. Delegate every control/reset to the original evaluator. No state
intervention, new opponent, script prefix or controller override. All saved raw
match fields, summary and hidden-reset counts must replay exactly or the trace
is non-interpretable failure evidence, not a basis for modifying the learner.

Freeze raw NPZ/manifest and source identities. CPU reduction pairs by actual
layout and side, distinguishes episode age from global time, and separately
reports input, output, hidden and Nexto cadence differences. Only comparable
uncontaminated local-age sequences may support trajectory claims. Missing
matches must be counted, never fabricated. No wheel intervention is needed if
wheel values are already exact. No new mechanic/possession detector.

Publish this authority and completed675review before execution. Afterward,
preserve all results, including null findings. The diagnostic does not select a
different checkpoint, alter inference, resume an expired campaign, or prove SSL.
