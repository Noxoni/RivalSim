# Comparing post-correction training, not the evaluator change

Use the CPU-only `benchmarks/report_rival2_direct_skills_matches.py` helper for
the +500 and subsequent full-match reviews. It verifies both source integrity
reports and checkpoint hashes, and requires identical corrected-reset version,
runtime package, authority, side assignments and starting layouts. It rejects
untagged historical results and reversed/identical update comparisons.

This is a reporting check, not a new training gate, reward, evaluation workload,
policy detector or automatic PPO stop. The learner is not interrupted. The same
fixed development matches remain in use; this does not certify ranked/SSL skill.

For +500, run after both the skill and full-match files are complete:

```powershell
.venv/Scripts/python.exe benchmarks/report_rival2_direct_skills_matches.py results/rival2/direct_skills_v1/corrected_reset/full_match_000451.json results/rival2/direct_skills_v1/full_match_000500.json --output results/rival2/direct_skills_v1/same_method_000451_to_000500.json
.venv/Scripts/python.exe benchmarks/report_rival2_direct_skills_matches.py results/rival2/direct_skills_v1/corrected_reset/full_match_000000.json results/rival2/direct_skills_v1/full_match_000500.json --output results/rival2/direct_skills_v1/same_method_000000_to_000500.json
```

Do not use the old untagged `full_match_000450.json` as a pure-learning baseline.
Task/skill evaluations still use the unchanged curriculum-reset path and retain
their existing comparisons. No old evidence is overwritten.

Eight focused CPU tests cover mismatched/missing method, authority and runtime
identity; case changes; old-reset inputs; reversed offsets; and deterministic
reduction against the real committed parent/+451 pair. The checked pair improves
goal difference in all ten matches but still loses all ten. This helper retains
both facts and emits no automatic training decision.
