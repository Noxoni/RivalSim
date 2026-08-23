# Reproducing RivalSim v0.1

These commands reproduce the published tests and generate fresh parity/benchmark artifacts.
Run them in a PowerShell session on a CUDA-capable NVIDIA Windows workstation.

## 1. Clone and create the isolated environment

```powershell
git clone https://github.com/Noxoni/RivalSim.git
Set-Location RivalSim
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-v0.1.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

The published evidence used Python 3.14.3 because it was the only registered interpreter on
the workstation. The package declares Python 3.12 or newer, and 3.12 remains the preferred
baseline, but a different interpreter/driver/toolkit is a new measurement environment and
should not be presented as a bit-for-bit reproduction of the published timings.

## 2. Reconstruct the source-custody references

The live parity harness needs only the pinned `rocketsim==2.2.1` wheel installed above. To
inspect the exact source used to derive constants and validate the binding, reconstruct the
ignored reference trees:

```powershell
New-Item -ItemType Directory -Force .reference | Out-Null
git clone https://github.com/ZealanL/RocketSim.git .reference/RocketSim
git -C .reference/RocketSim checkout c2baacb8f4b441dd8505e63c2aeb5a1679b60b02
git clone https://github.com/mtheall/RocketSim.git .reference/RocketSimPython
git -C .reference/RocketSimPython checkout 2da51b1dac7b8127127613a5ff30e490bdd70dd8
```

Verify the installed binding on Windows:

```powershell
Get-FileHash .venv\Lib\site-packages\RocketSim.pyd -Algorithm SHA256
```

The published hash is
`E3EE24CA82445B4BFCC754583F6778D7B0D8B7A7F7D64F872BE8C65E621A63D0`.

## 3. Run static and automated validation

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m compileall -q rivalsim benchmarks tests
.\.venv\Scripts\python.exe -m pytest -q
```

The tests require a working CUDA device and include GPU allocation, CPU/GPU parity, live
RocketSim cases and a 2,400-tick random contact-free stress run.

## 4. Re-run the frozen parity gate

```powershell
.\.venv\Scripts\python.exe benchmarks\run_parity.py `
  --device cuda:0 `
  --output results\v0.1\parity-reproduction.json
```

This runs all 27 scenarios at horizons 1/4/8/30/60/120 against the already-frozen tolerances
in `rivalsim/parity_tolerances.py`. Do not tune tolerances against the reproduction output.
Compare the logical outcomes and numeric error distribution; timestamp and small
floating-point differences can prevent file hashes from matching.

To recreate the pre-tolerance measurement protocol for a new prospective version, first make
both tolerance dictionaries empty in an uncommitted working copy and write the output under
`results/v0.1/raw/`. Do not overwrite the published v0.1 evidence.

## 5. Re-run the benchmark

Close unrelated GPU workloads and run:

```powershell
.\.venv\Scripts\python.exe benchmarks\run_benchmark.py `
  --device cuda:0 `
  --gpu-ticks 6000 `
  --cpu-ticks 360 `
  --gpu-repeats 5 `
  --cpu-repeats 3 `
  --warmup-ticks 120 `
  --graph-block-ticks 8 `
  --max-worlds 1048576 `
  --seed 20260822 `
  --output results\v0.1\benchmark-reproduction.json
```

The script always runs required batch sizes 256 through 16,384. It then doubles while median
world-tick throughput improves by at least 5%, resources permit and `--max-worlds` is not
exceeded. It synchronizes around timing, performs five GPU repeats, keeps telemetry in a
separate untimed pass, and verifies device state outside the timed loop.

The published path uses captured eight-tick CUDA graph blocks with resident fixed controls.
Changing graph block size, controls during a block, tick counts, repeat counts, telemetry
placement, hardware, power state or background GPU use makes the result a different protocol.

## 6. Verify compact evidence hashes

```powershell
Get-FileHash results\v0.1\benchmark.json,results\v0.1\parity.json -Algorithm SHA256
```

Expected published hashes and the ignored raw-measurement inventory are recorded in
`results/v0.1/manifest.json`.
