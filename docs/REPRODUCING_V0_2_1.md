# Reproducing RivalSim v0.2.1

These commands reproduce the v0.2.1 local-transition, coverage, stress, and performance evidence
on Windows PowerShell. They preserve the published v0.1/v0.2 evidence and stop before v0.3.

## 1. Check out the published boundary

Clone `Noxoni/RivalSim`, check out the v0.2.1 evidence commit recorded by `origin/main`, and
verify that its manifest binds implementation commit:

`9939d0736a92cdfa6ce842d7818634e08260dd65`

The final evidence commit is intentionally not stored inside its own manifest because doing so
would be a recursive self-hash. Verify it with `git log`, `git ls-remote`, and the remote readback
reported in the release handoff.

```powershell
git clone https://github.com/Noxoni/RivalSim.git
Set-Location RivalSim
git switch main
git pull --ff-only origin main
git status --short
```

The worktree must be clean before reproduction.

## 2. Create the pinned Python environment

The measured environment used Python 3.14.3. The project supports Python 3.12 or later.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements-v0.2.txt
```

The v0.2.1 implementation deliberately reuses the frozen v0.2 dependency lock: NumPy 2.5.2,
Warp 1.16.0, RocketSim 2.2.1, NVML 13.610.43, psutil 7.2.2, pytest 9.1.1, and Ruff 0.16.4.

## 3. Provide the exact external Soccar CMFs

Collision meshes are local game-derived inputs and must not be committed. Point the environment
variable at a root containing the 16-file `soccar` directory:

```powershell
$env:RIVALSIM_COLLISION_DIR = 'G:\dev\RLBot-Rival\bot\collision_meshes'
```

The loader must report:

- 16 files;
- 4,468 vertices;
- 8,020 triangles;
- combined content SHA-256
  `2239556BDC74D205CAA6E46A0F6E91FA2C6E4257E84D4F608BA775958B0A5538`.

The exact per-file hashes and bounds are in `results/v0.2.1/manifest.json`. Any local path is
acceptable when those bytes match.

## 4. Verify source custody

The implementation oracle revisions are:

- RocketSim: `c2baacb8f4b441dd8505e63c2aeb5a1679b60b02`;
- RocketSimPython: `2da51b1dac7b8127127613a5ff30e490bdd70dd8`.

If the optional reference checkouts exist at `.reference/RocketSim` and
`.reference/RocketSimPython`:

```powershell
git -C .reference\RocketSim rev-parse HEAD
git -C .reference\RocketSimPython rev-parse HEAD
git -C .reference\RocketSim status --short
git -C .reference\RocketSimPython status --short
```

Both must resolve to the hashes above and remain clean.

## 5. Run lint and regression tests

```powershell
.venv\Scripts\ruff.exe check rivalsim benchmarks tests
.venv\Scripts\python.exe -m compileall -q rivalsim benchmarks tests
.venv\Scripts\python.exe -m pytest -q `
  tests\test_arena.py tests\test_world_queries.py tests\test_static_world.py `
  --basetemp G:\dev\RivalSim\.tools\pytest-v021-targeted-reproduction
.venv\Scripts\python.exe -m pytest -q `
  --basetemp G:\dev\RivalSim\.tools\pytest-v021-full-reproduction
git diff --check
```

Expected results are 15/15 targeted and 38/38 full. The targeted group includes two independent
2,400-tick determinism runs and direct big/small boost-pad pickup, cooldown, and reset checks.

## 6. Reproduce the v0.1 live regression

```powershell
.venv\Scripts\python.exe benchmarks\run_parity.py `
  --device cuda:0 `
  --output .tools\v0.2.1-reproduction\v0.1-parity.json
```

Expected: 27 scenarios, with same-equation, live RocketSim, axis/sign, and overall parity all
passing. This output is intentionally ignored; the v0.2.1 manifest stores its compact summary
and SHA-256.

## 7. Reproduce authoritative local parity

```powershell
.venv\Scripts\python.exe benchmarks\run_v02_parity.py `
  --collision-dir $env:RIVALSIM_COLLISION_DIR `
  --output .tools\v0.2.1-reproduction\parity.json `
  --mode gate `
  --device cuda:0 `
  --milestone v0.2.1 `
  --horizons 1 4 8 12
```

Expected:

- 35 scenarios;
- 140 checkpoint comparisons;
- hard mismatch records: 0;
- numeric tolerance failures: 0;
- `parity_gate_pass: true`.

The frozen tolerances remain 10 uu position, 25 uu/s linear velocity, 0.025 rad orientation,
0.1 rad/s angular velocity, 0.01 boost, 0.0001 handbrake value, and 0.05 rad world-contact
normal. Hard on-ground, wheel-contact, world-contact, direction, and normal-sign checks remain
exact.

Do not substitute 30/60/120/300/600 ticks as the hard gate. Those synchronized open-loop
horizons are diagnostic-only under the 2026-08-23 steering adjustment.

## 8. Reproduce the bounded DFH coverage report

```powershell
.venv\Scripts\python.exe benchmarks\run_v021_coverage.py `
  --collision-dir $env:RIVALSIM_COLLISION_DIR `
  --parity-file .tools\v0.2.1-reproduction\parity.json `
  --output .tools\v0.2.1-reproduction\coverage.json `
  --device cuda:0
```

Expected observed coverage is 2 of 8,020 mesh triangles across the existing 35 cases and an
all-mesh topology audit of 23,176 shared directed edges. This is explicitly a bounded,
non-exhaustive prototype. It must not be reported as all-triangle authoritative parity.

## 9. Rebuild the optional native diagnostic oracle

This step is needed for causal trace reproduction, not for the published benchmark. It requires
CMake, Ninja, and MSVC. Initialize a Developer PowerShell or call `vcvars64.bat`, then configure
against the exact RocketSimPython checkout:

```powershell
cmake -S tools\rocketsim_diagnostic `
  -B .tools\rocketsim_diagnostic-reproduction `
  -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DROCKETSIM_BINDING_SOURCE_DIR="$PWD\.reference\RocketSimPython"
cmake --build .tools\rocketsim_diagnostic-reproduction --config Release
```

The executable accepts the external collision root, a supported scenario, and a tick count and
emits JSONL. Example:

```powershell
.tools\rocketsim_diagnostic-reproduction\rocketsim_diagnostic.exe `
  $env:RIVALSIM_COLLISION_DIR powerslide_initiation 12
```

The source-only helper links unmodified pinned source. Do not commit its build directory or CMFs.

## 10. Reproduce stress and the official B0/B1/B2/B3 benchmark

Run performance only after the local parity and regression gates pass:

```powershell
Copy-Item .tools\v0.2.1-reproduction\parity.json `
  .tools\v0.2.1-reproduction\parity-for-benchmark.json
.venv\Scripts\python.exe benchmarks\run_v02_benchmark.py `
  --collision-dir $env:RIVALSIM_COLLISION_DIR `
  --output .tools\v0.2.1-reproduction\benchmark.json `
  --device cuda:0 `
  --milestone v0.2.1 `
  --parity-file .tools\v0.2.1-reproduction\parity-for-benchmark.json
```

The driver first runs and validates two 64-world × 2,400-tick stress passes. It refuses to begin
timing unless full-state hashes match, state is finite/bounded, floor rest is stable, and hot-loop
transfer counters remain zero.

The default sweep measures five decomposed variants at 1,024 through 131,072 worlds and adds
262,144 when endpoint scaling remains worthwhile. The published RTX 5090 optimum is 822,480.77
aggregate sim-s/s at 262,144 worlds with 0.403% CV. Hardware, clocks, background load, and driver
version can change the reproduced rate; fidelity and residency gates must remain invariant.

Classification is:

- `PASS_GREEN`: parity plus at least 500,000 sim-s/s;
- `PASS`: parity plus 100,000 to less than 500,000 sim-s/s;
- `PAUSE_PERF`: parity but less than 100,000 sim-s/s;
- `PAUSE_FIDELITY`: any required local parity failure.

## 11. Rebuild the compact causal index and manifest

The causal index can be regenerated directly:

```powershell
.venv\Scripts\python.exe benchmarks\build_v021_divergence_index.py `
  --baseline results\v0.2\parity.json `
  --final .tools\v0.2.1-reproduction\parity.json `
  --output .tools\v0.2.1-reproduction\divergence_index.json
```

The release manifest is built only from the committed implementation plus the canonical
release evidence paths. For an exact package rebuild, place reproduced JSON at
`results/v0.2.1/`, preserve the two result documents, and run:

```powershell
.venv\Scripts\python.exe benchmarks\build_v021_manifest.py `
  --implementation-commit 9939d0736a92cdfa6ce842d7818634e08260dd65 `
  --output results\v0.2.1\manifest.json
```

The builder refuses changed v0.1/v0.2 evidence, tracked CMFs, a non-0.2.1 package version, or a
non-green published result.

## 12. Stop boundary

Stop after reproducing v0.2.1. Do not add ball-world, car-ball, car-car, bump/demolition,
scoring/reset, RLGym/PPO, Rival policy inference, or other v0.3 work without a separate handoff.
