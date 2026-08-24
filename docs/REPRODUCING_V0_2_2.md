# Reproducing RivalSim v0.2.2

These commands reproduce the bounded v0.2.2 static-world result. They do not authorize or run
v0.3 dynamic contacts.

## 1. Environment and collision assets

Use the repository's pinned Python environment and the exact RocketSim Soccar CMFs. In
PowerShell from the repository root:

```powershell
$collisionDir = 'G:\dev\RLBot-Rival\bot\collision_meshes'
$cacheRoot = '.tools\v0.2.2\oracle-cache'
```

The expected combined CMF SHA-256 is:

`2239556BDC74D205CAA6E46A0F6E91FA2C6E4257E84D4F608BA775958B0A5538`

Collision assets are external and ignored. Do not commit extracted game assets.

## 2. Verify the frozen native authority

```powershell
.\.venv\Scripts\python.exe benchmarks\build_v022_oracle_cache.py `
  --collision-dir $collisionDir `
  --cache-root $cacheRoot `
  --verify-only
```

Expected authority identity:

`6B31F9D147D5A19F882F9075C2A9F07C9A7228377A8A118CA2874F67FBD0805B`

This verifies every corpus and trajectory-cache chunk. Regenerate only if the RocketSim
revision, CMF content, generator source/config, seed, or authority settings change.

## 3. Run the representative cached gate

```powershell
.\.venv\Scripts\python.exe benchmarks\run_v022_breadth.py `
  --collision-dir $collisionDir `
  --oracle-cache-root $cacheRoot `
  --work-dir .tools\v0.2.2\pilot-release `
  --sample-count 1024 `
  --device cuda:0
```

The deterministic selection contains 1,043 cases because required pilot cases are added to the
1,024 sampled cases. Expected: `PILOT_PASS`, zero hard mismatches, and zero numeric failures.

## 4. Run the complete acceptance corpus

Run this only after the representative gate is clean:

```powershell
.\.venv\Scripts\python.exe benchmarks\run_v022_breadth.py `
  --collision-dir $collisionDir `
  --oracle-cache-root $cacheRoot `
  --work-dir .tools\v0.2.2\full-release `
  --device cuda:0
```

Expected:

- 39,236 starting states;
- 156,944 checkpoint comparisons at 1/4/8/12 ticks;
- 0 hard mismatch events;
- 0 numeric tolerance failures;
- 0 failed cases;
- `complete_v022_gate_pass: true`;
- classification `PASS_GREEN`.

The runner reads cached native authority only and cannot silently launch RocketSim.

## 5. Reproduce v0.1 regression and repository validation

```powershell
.\.venv\Scripts\python.exe benchmarks\run_parity.py `
  --device cuda:0 `
  --output .tools\v0.2.2\v01-regression-release.json

$env:RIVALSIM_COLLISION_DIR = $collisionDir
New-Item -ItemType Directory -Force .tools\pytest-v022-release | Out-Null
.\.venv\Scripts\python.exe -m pytest -q `
  --basetemp .tools\pytest-v022-release
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\python.exe -m compileall -q rivalsim benchmarks
git diff --check
```

Expected: all 27 v0.1 scenarios pass, all 46 repository tests pass, and all static checks pass.
A repository-local pytest base directory avoids machine-specific access problems in the default
Windows pytest temp root.

## 6. Reproduce stress and performance

```powershell
.\.venv\Scripts\python.exe benchmarks\run_v02_benchmark.py `
  --collision-dir $collisionDir `
  --output .tools\v0.2.2\benchmark-release.json `
  --device cuda:0 `
  --milestone v0.2.2 `
  --parity-file .tools\v0.2.2\full-release\aggregate.json `
  --reference-benchmark results\v0.2.1\benchmark.json
```

The benchmark refuses timing if query correctness or deterministic stress fails. The published
RTX 5090 result is 511,886.15 aggregate simulated game-seconds/s at 262,144 worlds with 0.0913%
CV, zero timed transfers, and `PASS_GREEN`. Hardware and background load may change throughput;
parity, query, stress, residency, and stability gates must remain invariant.

## 7. Rebuild compact acceptance evidence

```powershell
$authority = '6B31F9D147D5A19F882F9075C2A9F07C9A7228377A8A118CA2874F67FBD0805B'
$traceRoot = "$cacheRoot\$authority\deep-traces"

.\.venv\Scripts\python.exe benchmarks\build_v022_acceptance_evidence.py `
  --full-run-dir .tools\v0.2.2\full-release `
  --pilot-run-dir .tools\v0.2.2\pilot-release `
  --benchmark .tools\v0.2.2\benchmark-release.json `
  --v01-regression .tools\v0.2.2\v01-regression-release.json `
  --trace-dir "$traceRoot\C493A13448693870FA25F91F484E0C8785ECB32C1BCC650A40B5D5F46FEB87F2" `
  --trace-dir "$traceRoot\7A70F416941C46B2FDA9B9D31A02189B5EF8874ED733CA7D78E42B14AAF25866" `
  --output-dir results\v0.2.2
```

Focused diagnostic arguments are optional when reproducing only the acceptance summary. The
published `source_port.json` additionally hashes the final F07059-C, E20521, E21358, and E15855
diagnostics from the source-port run.

## 8. Rebuild oracle-data evidence

```powershell
.\.venv\Scripts\python.exe benchmarks\build_v022_oracle_evidence.py `
  --collision-dir $collisionDir `
  --cache-root $cacheRoot `
  --trace-dir "$traceRoot\0931DACD23E3116251421728BD03F2CC313E5C22676B3B2369AF39F102BBEF0F" `
  --goal-diagnostic .tools\v0.2.2\diagnostics\wheel-ray-source-F06155-W.json `
  --cached-pilot-dir .tools\v0.2.2\pilot-release `
  --full-run-dir .tools\v0.2.2\full-release
```

This artifact proves native-cache and deep-trace custody. `parity.json`, not
`oracle_data.json`, is the complete RivalSim acceptance verdict.

## 9. Stop boundary

Stop after v0.2.2 is verified and published. Do not begin ball-world, car-ball, car-car, dynamic
body, game-rule, training, or other v0.3 work without separate authority.
