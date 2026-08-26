# Reproducing RivalSim v0.5

These commands reproduce the v0.5 correctness, learning, performance, inherited regression, and
compact evidence gates on the authorized Windows/CUDA machine. Run them from the repository root
at the v0.5 implementation commit recorded in `results/v0.5/manifest.json`.

## Prerequisites

- Windows 11 x64;
- NVIDIA RTX 5090 and a CUDA-capable driver;
- Python 3.14;
- the exact external RocketSim Soccar CMFs at a local path;
- the ignored frozen native-authority caches under `.tools/v0.2.2`, `.tools/v0.3`, and `.tools/v0.4`;
- the unchanged pinned RocketSim build for the v0.1 live gate.

The release measurement used Python 3.14.3, NumPy 2.5.2, Warp 1.16.0, PyTorch 2.13.0+cu130,
NVIDIA driver 610.62, and an RTX 5090. Install the repository and the official CUDA 13.0 PyTorch
wheel in the project environment:

```powershell
py -3.14 -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
.venv\Scripts\python.exe -m pip install -e .
```

Set the collision root used by tests and shorten later commands:

```powershell
$env:RIVALSIM_COLLISION_DIR = 'G:\dev\RLBot-Rival\bot\collision_meshes'
$cmf = $env:RIVALSIM_COLLISION_DIR
New-Item -ItemType Directory -Force .tools\v0.5 | Out-Null
```

The exact CMF set and v0.4 authority identity are verified by the inherited lifecycle gate. Do not
substitute meshes while comparing the published results.

## v0.5 correctness and learning authority

```powershell
.venv\Scripts\python.exe -m benchmarks.run_v05_acceptance `
  --collision-dir $cmf `
  --output .tools\v0.5\acceptance_raw.json `
  --device cuda:0
```

Expected final line: `PASS_GREEN`. This single deterministic authority covers the zero-copy
bridge, observation corpus/symmetry, hybrid distribution CPU oracle, mechanics4 cadence,
reward/episode transitions, deterministic rollout buffer, GAE CPU oracle, PPO objective CPU
oracle, finite gradient/update checks, exact checkpoint continuation, historical opponents, and
the fixed-seed held-out learning smoke.

## End-to-end Rival 2.0 sweep

```powershell
.venv\Scripts\python.exe -m benchmarks.run_v05_benchmark `
  --collision-dir $cmf `
  --output .tools\v0.5\benchmark_raw.json `
  --device cuda:0 `
  --repeats 5
```

The default authorized list is 8,192, 16,384, 32,768, 65,536, and 131,072 worlds. Each point
warms up before five measured repeats. The JSON separates rollout, GAE, PPO, and complete wall
time and records decision rate, PPO consumption, simulated game time, update freshness, VRAM,
utilization sampling, CV, phase timings, and hot-loop H2D/D2H.

## Inherited v0.4 authority and lifecycle

Verify the existing content-addressed cache and rerun the lifecycle gate:

```powershell
.venv\Scripts\python.exe benchmarks\verify_v04_oracle_cache.py `
  --collision-dir $cmf `
  --cache-root .tools\v0.4\oracle

.venv\Scripts\python.exe benchmarks\run_v04_lifecycle.py `
  --collision-dir $cmf `
  --cache-root .tools\v0.4\oracle `
  --output .tools\v0.5\lifecycle-gate.json `
  --device cuda:0
```

The lifecycle output must be `PASS_GREEN`, use authority identity
`33AA0BA3BC35BC4300E2D2B84A3813CB0AD776479546A50AC3BBC6CE3D3E2562`, and report zero timed
traffic in deterministic stress.

## Inherited v0.3 complete corpora

```powershell
.venv\Scripts\python.exe benchmarks\run_v03_phase_a.py `
  --collision-dir $cmf `
  --oracle-cache-root .tools\v0.3\phase-a\oracle-cache `
  --output .tools\v0.5\regression-v03-phase-a.json `
  --device cuda:0 --full

.venv\Scripts\python.exe benchmarks\run_v03_phase_b.py `
  --collision-dir $cmf `
  --oracle-cache-root .tools\v0.3\phase-b\oracle-cache `
  --output .tools\v0.5\regression-v03-phase-b.json `
  --device cuda:0 --full

.venv\Scripts\python.exe benchmarks\run_v03_phase_c.py `
  --collision-dir $cmf `
  --oracle-cache-root .tools\v0.3\phase-c\oracle-cache-relational `
  --output .tools\v0.5\regression-v03-phase-c.json `
  --device cuda:0 --full

.venv\Scripts\python.exe benchmarks\run_v03_phase_d.py `
  --collision-dir $cmf `
  --oracle-cache-root .tools\v0.3\phase-d\oracle-cache-relational `
  --output .tools\v0.5\regression-v03-phase-d.json `
  --device cuda:0 --full
```

Expected complete counts are 31,216, 8,192, 8,192, and 512. Phase C and D compare each complete
trajectory to one labeled source-valid native visitation branch; metrics are never mixed between
branches.

## Inherited v0.2.2 and v0.1 gates

```powershell
.venv\Scripts\python.exe benchmarks\run_v022_breadth.py `
  --collision-dir $cmf `
  --oracle-cache-root .tools\v0.2.2\oracle-cache `
  --work-dir .tools\v0.5\regression-v022-full `
  --device cuda:0

.venv\Scripts\python.exe benchmarks\run_parity.py `
  --device cuda:0 `
  --output .tools\v0.5\regression-v01.json
```

The v0.2.2 aggregate must report all 39,236 cases selected with zero hard mismatches, zero numeric
failures, and zero failed cases. The v0.1 summary must report 27 scenarios and
`basic_parity_pass=true`.

## Inherited ray, lifecycle, and complete-path benchmark

```powershell
.venv\Scripts\python.exe benchmarks\run_v04_benchmark.py `
  --collision-dir $cmf `
  --output .tools\v0.5\v04-benchmark.json `
  --lifecycle-gate .tools\v0.5\lifecycle-gate.json `
  --v03-phase-a .tools\v0.5\regression-v03-phase-a.json `
  --v03-phase-b .tools\v0.5\regression-v03-phase-b.json `
  --v03-phase-c .tools\v0.5\regression-v03-phase-c.json `
  --v03-phase-d .tools\v0.5\regression-v03-phase-d.json `
  --v022-regression .tools\v0.5\regression-v022-full\aggregate.json `
  --v01-regression .tools\v0.5\regression-v01.json `
  --device cuda:0
```

This reruns both 4,608-ray backends plus complete and reset-heavy v0.4 measurements and refuses a
green aggregate if any supplied inherited gate is not green.

## Repository quality and frozen prior evidence

```powershell
.venv\Scripts\python.exe -m pytest -q --basetemp .tools\pytest-all-v05
.venv\Scripts\ruff.exe check .
.venv\Scripts\python.exe -m compileall -q rivalsim benchmarks tests tools
git diff --check
git diff --exit-code 8a422a86c69f16f0d62073992e515575f88733b5 -- `
  results/v0.1 results/v0.2 results/v0.2.1 results/v0.2.2 results/v0.3 results/v0.4
```

The last command must produce no output. It proves published v0.1–v0.4 evidence bytes were not
changed by v0.5.

## Build compact evidence

After every command above is green, substitute the observed passing test count:

```powershell
.venv\Scripts\python.exe benchmarks\build_v05_evidence.py `
  --repository-tests 78 `
  --ruff-pass --compile-pass --diff-check-pass
```

The evidence builder reads only completed local gate outputs, validates their green status and the
prior-evidence byte boundary, then creates the compact files under `results/v0.5/`.

After committing the implementation, stage the release files except the not-yet-generated
manifest, then bind the exact staged Git blobs to that implementation commit:

```powershell
$implementation = git rev-parse HEAD
git add CODEX_START_PROMPT.md VERSION.md README.md CHANGELOG.md pyproject.toml `
  docs\ROADMAP.md docs\V0_5_RESULTS.md docs\REPRODUCING_V0_5.md `
  docs\RIVAL2_TRAINING_CONTRACT.md results\v0.5
.venv\Scripts\python.exe benchmarks\build_v05_manifest.py `
  --implementation-commit $implementation
git add results\v0.5\manifest.json
```

Commit the human reports and `results/v0.5/manifest.json`, push both commits together, and verify
the remote main SHA with `git ls-remote origin refs/heads/main`. A manifest may be called
`PASS_GREEN` only when every blocking gate is green and `v0_6_begun` is false.
