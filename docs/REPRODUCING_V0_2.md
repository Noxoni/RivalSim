# Reproducing RivalSim v0.2

These commands reproduce the published v0.2 evidence on Windows PowerShell. They require an
NVIDIA CUDA-capable GPU and the exact external Soccar `.cmf` set; the mesh files are not part of
this repository.

## 1. Clone and identify the implementation

```powershell
git clone https://github.com/Noxoni/RivalSim.git
Set-Location RivalSim
git fetch origin main
git checkout main
git rev-parse HEAD
```

The evidence manifest records implementation commit `f236310` and the full source revision.
For historical reproduction, check out that implementation commit before running the gate
scripts; use the final evidence commit when verifying the published JSON and documentation.

## 2. Create the exact Python environment

```powershell
py -3.14 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements-v0.2.txt
.venv\Scripts\python.exe -m pip install -e . --no-deps
```

The published environment uses Python 3.14.3, NumPy 2.5.2, Warp 1.16.0,
`rocketsim==2.2.1`, psutil 7.2.2, nvidia-ml-py 13.610.43, pytest 9.1.1, and Ruff 0.16.4.

`pip check` may report the already documented upstream RocketSim wheel-tag metadata mismatch.
The installed extension must still import, initialize with the exact CMFs, and pass the live
oracle suite; do not reinterpret a working binding as absent solely because of that metadata
warning.

## 3. Supply the exact external collision assets

The published run used the files tracked in `Noxoni/Rival` commit
`36cb14cf645c4f06b668c34d85ce1a500e4b53da` at:

```text
bot/collision_meshes/soccar/mesh_0.cmf ... mesh_15.cmf
```

Set the parent collision directory, not the `soccar` child itself:

```powershell
$env:RIVALSIM_COLLISION_DIR = 'G:\dev\RLBot-Rival\bot\collision_meshes'
```

Any path is acceptable if its 16 files match `results/v0.2/manifest.json`. Validate the exact
parser, hashes, counts, bounds, and CPU/GPU queries with:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_arena.py tests/test_world_queries.py -q `
  --basetemp G:\dev\RivalSim\.tools\pytest-v02-geometry
```

Do not copy or commit the `.cmf` files into RivalSim.

## 4. Run the full tests

```powershell
.venv\Scripts\ruff.exe check rivalsim benchmarks tests
.venv\Scripts\python.exe -m pytest -q `
  --basetemp G:\dev\RivalSim\.tools\pytest-v02-full
.venv\Scripts\python.exe -m compileall -q rivalsim benchmarks tests
git diff --check
```

The published final run passes 37 tests. Asset/CUDA tests skip rather than silently substituting
geometry if `RIVALSIM_COLLISION_DIR` or CUDA is unavailable.

## 5. Reproduce the parity protocol

The historical measurement-only phase must precede any new tolerance selection:

```powershell
.venv\Scripts\python.exe benchmarks/run_v02_parity.py `
  --collision-dir $env:RIVALSIM_COLLISION_DIR `
  --mode measurement `
  --output .tools\v0.2-parity-measurement.json
Get-FileHash .tools\v0.2-parity-measurement.json -Algorithm SHA256
```

The published pre-freeze measurement hash is
`7EB62CF97BE25EA5F7CF6540D9D6350829B0AE7887B09F0B63E3915E937B9BDF`. A new run has a new
timestamp and therefore a different whole-file hash; compare scenario errors and aggregates,
not the timestamp-bearing JSON hash, unless reproducing the original artifact byte-for-byte.

Run the clean gate using the already frozen table:

```powershell
.venv\Scripts\python.exe benchmarks/run_v02_parity.py `
  --collision-dir $env:RIVALSIM_COLLISION_DIR `
  --mode gate `
  --output .tools\v0.2-parity-reproduced.json
```

The expected classification is a failed parity gate, not a pass: 85 scenario/horizon records
contain hard mismatches and 617 numeric checks exceed frozen limits on the published machine.

## 6. Reproduce B0/B1/B2/B3 and stress evidence

```powershell
.venv\Scripts\python.exe benchmarks/run_v02_benchmark.py `
  --collision-dir $env:RIVALSIM_COLLISION_DIR `
  --output .tools\v0.2-benchmark-reproduced.json `
  --repeats 5 `
  --warmup-ticks 16 `
  --graph-block-ticks 8 `
  --ticks 64 `
  --max-worlds 262144 `
  --seed 20260823
```

The script runs the required 1,024–131,072 sweep, extends variants to 262,144 only when the
131,072 endpoint is still materially rising, compares normal and cuBQL ray backends, restores
an identical device checkpoint before every measured repeat, and keeps verification readback
outside timing. It also executes the two-run 2,400-tick determinism/stress gate.

Expect hardware and background-load variation. The published RTX 5090 best B3 point is
162,089,778.61 world-ticks/s, 1,350,748.16 aggregate sim-s/s at 262,144 worlds, with 0.998% CV.
Correctness still forces `PAUSE_RED`.

## 7. Re-run v0.1 regressions

```powershell
.venv\Scripts\python.exe benchmarks/run_parity.py `
  --device cuda:0 `
  --output .tools\v0.1-parity-regression.json

.venv\Scripts\python.exe benchmarks/run_benchmark.py `
  --device cuda:0 `
  --gpu-ticks 16 `
  --cpu-ticks 2 `
  --gpu-repeats 3 `
  --cpu-repeats 3 `
  --warmup-ticks 8 `
  --graph-block-ticks 8 `
  --max-worlds 16384 `
  --seed 20260823 `
  --output .tools\v0.1-benchmark-regression.json
```

The live v0.1 parity gate must remain 27/27 passing. The small benchmark command is a regression
smoke, not a replacement for the immutable published v0.1 performance sweep.

## 8. Verify evidence and asset exclusion

```powershell
.venv\Scripts\python.exe -c "import json; from pathlib import Path; [json.loads(p.read_text(encoding='utf-8')) for p in Path('results/v0.2').glob('*.json')]; print('JSON OK')"
git ls-files '*.cmf' '*.pskx' '*.bin'
git diff 1f7a36cc6165273fb658ba07a8458e8d8e60628a..HEAD -- results/v0.1
git diff --check
```

The asset listing and frozen-v0.1 diff must both be empty. Verify exact evidence blob hashes
against `results/v0.2/manifest.json`.

## Boundary

Stop after reproducing v0.2. Do not add ball-world, car-ball, car-car, boost pads, scoring,
training integration, or any other v0.3+ feature without a new authority package.
