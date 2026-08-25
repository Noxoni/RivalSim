# Reproducing RivalSim v0.3

Run from the repository root in PowerShell. Native cache generation requires the exact pinned
RocketSim build and external Soccar collision assets. GPU gates and benchmarks require an NVIDIA
CUDA device supported by Warp 1.16.0.

## 1. Environment

```powershell
py -3.14 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-v0.2.txt
.venv\Scripts\python.exe -m pip install -e .[dev]
$env:RIVALSIM_COLLISION_DIR = 'G:\dev\RLBot-Rival\bot\collision_meshes'
```

The CMF directory must contain the exact 16-file Soccar set with combined content SHA-256
`2239556BDC74D205CAA6E46A0F6E91FA2C6E4257E84D4F608BA775958B0A5538`. Extracted game assets are
external and must not be committed.

The installed ordinary RocketSim extension must hash to
`E3EE24CA82445B4BFCC754583F6778D7B0D8B7A7F7D64F872BE8C65E621A63D0`. Phase C/D cache generation
also requires the read-only logical-order diagnostic extension with SHA-256
`A92F6680284A7149843AFC1041C70DB574ED4CAEA5176F3EF0F1E0E216763807`.

## 2. Build or verify native authority

The following commands resume matching chunks, reject mismatched identities, and verify all chunks
before finalization. Cache generation may be lengthy.

```powershell
.venv\Scripts\python.exe benchmarks\build_v03_phase_a_cache.py --collision-dir $env:RIVALSIM_COLLISION_DIR --cache-root .tools\v0.3\phase-a\oracle-cache
.venv\Scripts\python.exe benchmarks\build_v03_phase_b_cache.py --collision-dir $env:RIVALSIM_COLLISION_DIR --cache-root .tools\v0.3\phase-b\oracle-cache
.venv\Scripts\python.exe benchmarks\build_v03_phase_c_cache.py --collision-dir $env:RIVALSIM_COLLISION_DIR --cache-root .tools\v0.3\phase-c\oracle-cache-relational --order-diagnostic-extension .tools\RocketSimPhaseCBranch-build\RocketSim.pyd
.venv\Scripts\python.exe benchmarks\build_v03_phase_d_cache.py --collision-dir $env:RIVALSIM_COLLISION_DIR --cache-root .tools\v0.3\phase-d\oracle-cache-relational --order-diagnostic-extension .tools\RocketSimPhaseCBranch-build\RocketSim.pyd
```

For an already complete cache, add `--verify-only`. Never use an acceptance runner to regenerate
missing data. See `docs/V0_3_ORACLE_CACHE.md` for expected identities.

## 3. Run the four complete cached gates

```powershell
.venv\Scripts\python.exe benchmarks\run_v03_phase_a.py --collision-dir $env:RIVALSIM_COLLISION_DIR --oracle-cache-root .tools\v0.3\phase-a\oracle-cache --output .tools\v0.3\phase-a\full-v03-release.json --device cuda:0 --full
.venv\Scripts\python.exe benchmarks\run_v03_phase_b.py --collision-dir $env:RIVALSIM_COLLISION_DIR --oracle-cache-root .tools\v0.3\phase-b\oracle-cache --output .tools\v0.3\phase-b\full-v03-release.json --device cuda:0 --full
.venv\Scripts\python.exe benchmarks\run_v03_phase_c.py --collision-dir $env:RIVALSIM_COLLISION_DIR --oracle-cache-root .tools\v0.3\phase-c\oracle-cache-relational --output .tools\v0.3\phase-c\full-v03-release.json --device cuda:0 --full
.venv\Scripts\python.exe benchmarks\run_v03_phase_d.py --collision-dir $env:RIVALSIM_COLLISION_DIR --oracle-cache-root .tools\v0.3\phase-d\oracle-cache-relational --output .tools\v0.3\phase-d\full-v03-release.json --device cuda:0 --full
```

Expected complete status is `PASS_GREEN` for all four files, with zero blocking hard/numeric
failures and case counts 31,216, 8,192, 8,192, and 512.

## 4. Mandatory regressions

The v0.2.2 runner consumes its previously frozen cache and has no live fallback:

```powershell
.venv\Scripts\python.exe benchmarks\run_v022_breadth.py --collision-dir $env:RIVALSIM_COLLISION_DIR --oracle-cache-root .tools\v0.2.2\oracle-cache --work-dir .tools\v0.3\regression-v022-full --device cuda:0
.venv\Scripts\python.exe benchmarks\run_parity.py --device cuda:0 --output .tools\v0.3\v01-regression-final.json
git diff --exit-code 6dfd44ad9afeb3d1164da7e0e38c097fb74d07b8 -- results/v0.1 results/v0.2 results/v0.2.1 results/v0.2.2
```

Expected results are 39,236/39,236 v0.2.2 cases, 27/27 v0.1 scenarios, and no prior-evidence
diff. Both 4,608-ray backends are rechecked by the final benchmark.

## 5. Repository checks

Use a repository-local pytest base directory on Windows to avoid unrelated temporary-directory ACL
failures:

```powershell
.venv\Scripts\python.exe -m pytest -q --basetemp .tools\pytest-v03-release
.venv\Scripts\python.exe -m ruff check rivalsim benchmarks tests
.venv\Scripts\python.exe -m compileall -q rivalsim benchmarks tests
git diff --check
```

The published run passed 63 tests.

## 6. Determinism and performance

The benchmark performs both independent 64-world/2,400-tick stress runs, the 4,608-ray backend
gate, the integrated batch sweep, and selected-batch component decomposition:

```powershell
.venv\Scripts\python.exe benchmarks\run_v03_benchmark.py --collision-dir $env:RIVALSIM_COLLISION_DIR --output .tools\v0.3\benchmark-final.json --phase-d-parity .tools\v0.3\phase-d\full-v03-release.json --v022-regression .tools\v0.3\regression-v022-full\aggregate.json --v01-regression .tools\v0.3\v01-regression-final.json --device cuda:0
```

The published best stable complete point is 196,614.39 aggregate simulated game-seconds/s at
131,072 worlds with 1.313% CV and zero timed transfers. The two stress hashes must be identical.

## 7. Compact evidence

After every command above is green:

```powershell
.venv\Scripts\python.exe benchmarks\build_v03_evidence.py --repository-tests 63 --ruff-pass --compile-pass --diff-check-pass
```

The release manifest is generated after the implementation checkpoint commit so it can bind that
commit and the final compact evidence bytes. Large `.tools/v0.3/` data remains untracked.
