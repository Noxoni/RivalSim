# Reproduce native bridge preparation checks

From `G:/dev/RivalSim`:

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_entity_native_bridge.py tests/test_ssl_entity_policy.py tests/test_ssl_joint_control_policy.py --basetemp .tmp/pytest-entity-native-fresh
```

Published no-learning capture/export script:
`benchmarks/prepare_entity_native_bridge.py`.
It refuses completed output overwrite and verifies original checkpoint hashes,
the published method, GPU lease and protocol. The committed NPZ captures support
deterministic readers without another simulator run. Do not rerun the whole
script merely to inspect an existing result.

Target interpreter probe:
`G:/dev/RLBot-Rival/.venv/Scripts/python.exe -B benchmarks/check_entity_native_target_runtime.py`.
It reads exports/native archives in RivalSim and creates only its one result
file there, refusing overwrite. No RLBot service/game startup or installation.

`reference600.json` and `finishing650.json` bind artifact/corpus/source hashes.
`input_availability.json` is a complete field-level source audit, not a learned
adapter and not measured native packet domain equivalence. See RESULTS.md for
limitations and concrete next integration work.
