# RivalSim v0.2.2 RocketSim authority cache

This cache freezes native RocketSim outputs for the deterministic 39,236-case DFH breadth
corpus. The cache itself is an oracle-data package, not a RivalSim parity result. The separately
tracked `results/v0.2.2/parity.json` now records the completed GPU acceptance gate against this
authority.

## Semantic identity and invalidation

The content-addressed cache directory is the SHA-256 of one canonical JSON identity. The
identity binds:

- the pinned primary RocketSim commit, Python-binding commit, package version, and installed
  extension hash;
- the combined collision-geometry SHA-256 plus every source CMF filename, size, and SHA-256;
- the breadth generator source SHA-256, schema, explicit configuration, seed, complete case
  count, and exact float32 corpus SHA-256;
- all relevant authority settings: Soccar, 120 Hz, Octane, one fresh one-car arena per case,
  collision switches, ball placement, initial-state readback, held controls, step order,
  captured ticks, frame fields, and contact semantics.

RivalSim GPU implementation changes, CUDA/Warp compilation, benchmark device, pilot selection,
and result tolerances are deliberately absent from this semantic identity. They do not
invalidate native authority data. A corrupt or missing artifact is repaired under the same
identity; it does not create a new authority identity.

## Complete cached payload

For every corpus case, the cache stores:

- the exact `StateSnapshot` produced by RocketSim's immediate post-`SetState` readback;
- every native frame from tick 1 through tick 12, including position, linear velocity,
  rotation matrix, angular velocity, boost, handbrake value, ground state, four wheel-contact
  booleans, chassis-world contact, and world-contact normal;
- the complete frozen generated case description and case order.

Chunks are compressed NPZ files with embedded identity, range, case IDs, and captured ticks.
The completion manifest hashes the frozen corpus and every chunk. A separate completion marker
hashes the manifest. The breadth runner verifies the identity, completion marker, corpus, and
every chunk it consumes; it has no live-RocketSim fallback.

## Generation and verification

From the repository root in PowerShell:

```powershell
$collisionDir = 'G:\dev\RLBot-Rival\bot\collision_meshes'
$cacheRoot = '.tools\v0.2.2\oracle-cache'

.\.venv\Scripts\python.exe benchmarks\build_v022_oracle_cache.py `
  --collision-dir $collisionDir `
  --cache-root $cacheRoot

.\.venv\Scripts\python.exe benchmarks\build_v022_oracle_cache.py `
  --collision-dir $collisionDir `
  --cache-root $cacheRoot `
  --verify-only
```

Authority simulation steps RocketSim only. The package import may discover the installed
Warp/CUDA runtime, but generation never constructs or steps `StaticWorldSim` and never evaluates
a RivalSim tolerance or acceptance classification. It is resumable by validated 256-case chunks.

## Cached GPU iterations

Every later breadth iteration must name the cache root:

```powershell
.\.venv\Scripts\python.exe benchmarks\run_v022_breadth.py `
  --collision-dir $collisionDir `
  --oracle-cache-root $cacheRoot `
  --work-dir .tools\v0.2.2\pilot-cached `
  --sample-count 1024 `
  --device cuda:0
```

This command may run a selected RivalSim pilot, but it cannot launch RocketSim. Omitting the
cache or presenting a stale, partial, mismatched, or corrupt cache is a hard error.

## Deep native trace cache

`build_v022_deep_trace_cache.py` freezes the failing-case selection from a completed pilot and
runs two native passes per case:

1. Discovery records staged native state, actual contact callbacks, persistent-manifold
   evolution, and exact per-tick Bullet BVH candidate order.
2. Operation capture probes the union of discovered body/face pairs at every tick and records
   source-ordered `btGjkPairDetector` operations, `btVoronoiSimplexSolver` cached points,
   barycentrics and vertex use, exact termination reason, final full-solver GJK result, direct
   EPA witnesses, the pinned nine-guess penetration-depth fallback sequence, contact callbacks,
   and retained manifold faces.

The deep-trace identity nests the complete authority identity and additionally hashes the
source pilot artifacts, frozen failed-case list, trace source, trace executable, and capture
protocol. Trace-instrumentation changes therefore regenerate only the deep traces; they do not
invalidate the complete trajectory authority cache.

```powershell
.\.venv\Scripts\python.exe benchmarks\build_v022_deep_trace_cache.py `
  --collision-dir $collisionDir `
  --cache-root $cacheRoot `
  --pilot-run-dir .tools\v0.2.2\pilot-1024-direct-bullet-v4-relative-transform `
  --diagnostic-exe .tools\rocketsim_diagnostic-ipo-build\rocketsim_diagnostic.exe
```

These traces are comparison oracles for source-port debugging. They are not parity evidence and
must not be reported as a v0.2.2 pass.

## Compact tracked evidence

After generating and independently verifying the complete trajectory cache, deep traces, and a
representative cached GPU pilot, write the compact evidence record:

```powershell
.\.venv\Scripts\python.exe benchmarks\build_v022_oracle_evidence.py `
  --collision-dir $collisionDir `
  --cache-root $cacheRoot `
  --trace-dir $cacheRoot\<authority-identity>\deep-traces\<trace-identity> `
  --goal-diagnostic .tools\v0.2.2\diagnostics\<goal-diagnostic>.json `
  --cached-pilot-dir .tools\v0.2.2\pilot-cached
```

The builder rehashes every authority chunk and deep-trace artifact. It also requires the pilot
to carry the same authority, corpus, and selection identities and to declare
`cached_native_authority_only`. The resulting `results/v0.2.2/oracle_data.json` records
native-cache custody and the representative result without converting the oracle-data artifact
itself into an acceptance claim.

## Complete acceptance evidence

Only after the representative selection is clean, run the full cached corpus with no
`--sample-count`. Then compact and validate the full result, benchmark, v0.1 regression, and
focused source-port traces:

```powershell
.\.venv\Scripts\python.exe benchmarks\run_v022_breadth.py `
  --collision-dir $collisionDir `
  --oracle-cache-root $cacheRoot `
  --work-dir .tools\v0.2.2\full-release `
  --device cuda:0

.\.venv\Scripts\python.exe benchmarks\build_v022_acceptance_evidence.py `
  --full-run-dir .tools\v0.2.2\full-release `
  --pilot-run-dir .tools\v0.2.2\pilot-release `
  --benchmark .tools\v0.2.2\benchmark-release.json `
  --v01-regression .tools\v0.2.2\v01-regression-release.json `
  --output-dir results\v0.2.2
```

`build_v022_acceptance_evidence.py` refuses a non-complete selection, live-oracle execution,
identity mismatch, missing chunk, any blocking failure, a non-green benchmark, a failed stress
or residency gate, or a failed v0.1 regression. The published complete result is 39,236/39,236
cases with zero hard and numeric failures and classification `PASS_GREEN`.
