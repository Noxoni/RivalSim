# Reproducing RivalSim v0.4

Run from the repository root in PowerShell. Native authority generation requires the exact pinned
RocketSim package/build and external Soccar collision assets. GPU gates and benchmarks require an
NVIDIA CUDA device supported by Warp 1.16.0.

## 1. Environment and custody

```powershell
py -3.14 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-v0.2.txt
.venv\Scripts\python.exe -m pip install -e .[dev]
$env:RIVALSIM_COLLISION_DIR = 'G:\dev\RLBot-Rival\bot\collision_meshes'
```

The collision root must contain the exact 16-file `soccar/` CMF set with combined content
SHA-256 `2239556BDC74D205CAA6E46A0F6E91FA2C6E4257E84D4F608BA775958B0A5538`.
Extracted game assets remain external and must not be committed. The installed ordinary
`RocketSim.pyd` must hash to
`E3EE24CA82445B4BFCC754583F6778D7B0D8B7A7F7D64F872BE8C65E621A63D0`.

Verify the authorized start before applying the v0.4 implementation:

```powershell
git rev-parse b5875c4b853a8ce844d0904e989b1d2a3854d0ac
git diff --exit-code b5875c4b853a8ce844d0904e989b1d2a3854d0ac -- results/v0.1 results/v0.2 results/v0.2.1 results/v0.2.2 results/v0.3
```

## 2. Build or verify v0.4 native authority

```powershell
.venv\Scripts\python.exe benchmarks\build_v04_oracle_cache.py $env:RIVALSIM_COLLISION_DIR --cache-root .tools\v0.4\oracle
```

Expected identity:

`33AA0BA3BC35BC4300E2D2B84A3813CB0AD776479546A50AC3BBC6CE3D3E2562`

If that directory is already complete, the builder verifies and reuses it. Any missing, corrupt,
or current-source-mismatched cache is an error for the gate. The acceptance runner never falls
back to live native collection.

## 3. Run the complete lifecycle gate

```powershell
.venv\Scripts\python.exe benchmarks\run_v04_lifecycle.py $env:RIVALSIM_COLLISION_DIR --cache-root .tools\v0.4\oracle --device cuda:0 --output .tools\v0.4\lifecycle-gate.json
```

Expected status is `PASS_GREEN` for phases A through D and overall. The gate covers 68 pad/car
pickups, both cooldown types and contention orders, six goal boundaries, five kickoff layouts,
eight team/respawn poses, exact tick-360 respawn, and two equal-hash deterministic 64-world,
400-tick lifecycle stress executions with zero timed transfers.

## 4. Inherited native physics gates

The v0.3 caches remain immutable and have no live acceptance fallback:

```powershell
.venv\Scripts\python.exe benchmarks\run_v03_phase_a.py --collision-dir $env:RIVALSIM_COLLISION_DIR --oracle-cache-root .tools\v0.3\phase-a\oracle-cache --output .tools\v0.4\regression-v03-phase-a.json --device cuda:0 --full
.venv\Scripts\python.exe benchmarks\run_v03_phase_b.py --collision-dir $env:RIVALSIM_COLLISION_DIR --oracle-cache-root .tools\v0.3\phase-b\oracle-cache --output .tools\v0.4\regression-v03-phase-b.json --device cuda:0 --full
.venv\Scripts\python.exe benchmarks\run_v03_phase_c.py --collision-dir $env:RIVALSIM_COLLISION_DIR --oracle-cache-root .tools\v0.3\phase-c\oracle-cache-relational --output .tools\v0.4\regression-v03-phase-c.json --device cuda:0 --full
.venv\Scripts\python.exe benchmarks\run_v03_phase_d.py --collision-dir $env:RIVALSIM_COLLISION_DIR --oracle-cache-root .tools\v0.3\phase-d\oracle-cache-relational --output .tools\v0.4\regression-v03-phase-d.json --device cuda:0 --full
```

Expected complete counts are 31,216, 8,192, 8,192, and 512. Phase C and D must pass both complete
native-valid visitation branches without metric mixing.

## 5. v0.2.2, v0.1, and prior-evidence regressions

```powershell
.venv\Scripts\python.exe benchmarks\run_v022_breadth.py --collision-dir $env:RIVALSIM_COLLISION_DIR --oracle-cache-root .tools\v0.2.2\oracle-cache --work-dir .tools\v0.4\regression-v022-full --device cuda:0
.venv\Scripts\python.exe benchmarks\run_parity.py --device cuda:0 --output .tools\v0.4\regression-v01.json
git diff --exit-code b5875c4b853a8ce844d0904e989b1d2a3854d0ac -- results/v0.1 results/v0.2 results/v0.2.1 results/v0.2.2 results/v0.3
```

Expected results are 39,236/39,236 v0.2.2 cases, 27/27 v0.1 scenarios, and zero prior-evidence
differences.

## 6. Repository checks

Use a repository-local pytest base directory on Windows to avoid unrelated temporary-directory
ACL failures:

```powershell
$env:RIVALSIM_COLLISION_DIR = 'G:\dev\RLBot-Rival\bot\collision_meshes'
.venv\Scripts\python.exe -m pytest -q --basetemp .tools\pytest-v04-release
.venv\Scripts\ruff.exe check .
.venv\Scripts\python.exe -m compileall -q rivalsim benchmarks tests tools
git diff --check
```

The published configured run passed 70 tests.

## 7. Ray, complete-path, and reset-heavy benchmark

The benchmark first checks both 4,608-ray backends, consumes all mandatory gate JSON, sweeps the
complete lifecycle path, then measures a deterministic full reset every eight ticks:

```powershell
.venv\Scripts\python.exe benchmarks\run_v04_benchmark.py --collision-dir $env:RIVALSIM_COLLISION_DIR --output .tools\v0.4\benchmark.json --lifecycle-gate .tools\v0.4\lifecycle-gate.json --v03-phase-a .tools\v0.4\regression-v03-phase-a.json --v03-phase-b .tools\v0.4\regression-v03-phase-b.json --v03-phase-c .tools\v0.4\regression-v03-phase-c.json --v03-phase-d .tools\v0.4\regression-v03-phase-d.json --v022-regression .tools\v0.4\regression-v022-full\aggregate.json --v01-regression .tools\v0.4\regression-v01.json --device cuda:0
```

The published complete point is 191,748.10 aggregate simulated game-seconds/s at 131,072 worlds
with 0.856% CV and zero timed transfers. The reset-heavy point is 225,005.06 sim-s/s,
3,375,075.88 reset transitions/s, 0.723% CV, and zero timed transfers. Both ray backends pass.

## 8. Compact evidence and manifest

After every gate above is green:

```powershell
.venv\Scripts\python.exe benchmarks\build_v04_evidence.py --collision-dir $env:RIVALSIM_COLLISION_DIR --cache-root .tools\v0.4\oracle --repository-tests 70 --ruff-pass --compile-pass --diff-check-pass
```

Commit the stable implementation first. With that commit still at `HEAD`, stage the release docs
and compact JSON, then bind them:

```powershell
.venv\Scripts\python.exe benchmarks\build_v04_manifest.py --implementation-commit da34c6d8a9ad4eb6aaced955ef0fe96575e1ec56
```

Large `.tools/v0.4/` authority, regressions, and benchmark intermediates remain untracked. Do not
begin v0.5.
