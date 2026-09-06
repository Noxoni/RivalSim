# Legacy Nexto simulator results are not native-fidelity evidence

2026-09-06: a controlled comparison confirmed that `third_party/nexto/adapter.py`
applies neural controls with different timing from the installed native Nexto.
The installed implementation separates a newly computed pending action from
the currently emitted controls; the simulator has applied new predictions
immediately. The model weights themselves are identical.

On unchanged reference600, the old full-match benchmark is 8/10 wins, 169-122
goals. Native compute/emission timing plus the corrected kickoff literal gives
0/10 wins, 34-239 goals on the same fixed ten starts and full regulation.
Both arms reproduce their earlier action traces exactly; no learning occurred.

Full evidence: `results/rival2/nexto_native_full_match_v1/RESULTS.md`.
Four-way timing-versus-table isolation:
`results/rival2/nexto_native_causal_v1/RESULTS.md`.
Actual installed source binding:
`results/rival2/native_nexto_integration_audit_v1/RESULTS.md`.

Older scores remain valid measurements of the **legacy simulator opponent**,
not proof Rival beats actual native Nexto. Do not delete/rewrite those results,
promote a model based on them, or label the diagnostic timing subclass a complete
production/native-fidelity correction. Native Rival results remain poor (0-24,
1-30). Production-safe integration and its focused validation are the next step.
