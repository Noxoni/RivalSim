"""Precompute native operation traces for the current failing v0.2.2 pilot cases."""

from __future__ import annotations

import argparse
import gzip
import io
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rivalsim.arena import ArenaGeometry
from rivalsim.dfh_breadth import (
    BreadthCase,
    build_breadth_catalog,
    generate_breadth_cases,
)
from rivalsim.math import quat_to_matrix
from rivalsim.v022_oracle_cache import (
    RocketSimAuthorityCache,
    build_authority_identity,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    validate_installed_rocketsim_extension,
)

TRACE_SCHEMA_VERSION = 5
TRACE_TICKS = 12
EXPECTED_GEOMETRY_SHA256 = "2239556BDC74D205CAA6E46A0F6E91FA2C6E4257E84D4F608BA775958B0A5538"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collision-dir", required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--pilot-run-dir",
        type=Path,
        help="completed representative pilot whose currently failing cases are frozen",
    )
    selection.add_argument(
        "--case-id",
        dest="case_ids",
        action="append",
        help=(
            "frozen corpus case to trace directly; repeat for additional cases. "
            "The exact ordered selection is included in the trace identity"
        ),
    )
    parser.add_argument("--diagnostic-exe", type=Path, required=True)
    parser.add_argument(
        "--diagnostic-source",
        type=Path,
        default=Path("tools/rocketsim_diagnostic/trace.cpp"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    extension = validate_installed_rocketsim_extension()
    executable = args.diagnostic_exe.resolve()
    source = args.diagnostic_source.resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"diagnostic executable not found: {executable}")
    if not source.is_file():
        raise FileNotFoundError(f"diagnostic source not found: {source}")

    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    if geometry.content_sha256 != EXPECTED_GEOMETRY_SHA256:
        raise RuntimeError(
            f"unexpected collision geometry: {geometry.content_sha256}, "
            f"expected {EXPECTED_GEOMETRY_SHA256}"
        )
    cases = generate_breadth_cases(build_breadth_catalog(geometry))
    identity = build_authority_identity(geometry, cases)
    authority_cache = RocketSimAuthorityCache(args.cache_root, identity, cases)
    if args.case_ids:
        source_selection, failed_ids = _explicit_case_selection(args.case_ids, identity)
    else:
        source_selection, failed_ids = _pilot_failure_selection(
            args.pilot_run_dir.resolve()
        )
    case_by_id = {case.case_id: (index, case) for index, case in enumerate(cases)}
    missing = sorted(set(failed_ids) - set(case_by_id))
    if missing:
        raise RuntimeError(f"pilot failures are absent from the frozen corpus: {missing}")
    selected = tuple(case_by_id[case_id] for case_id in failed_ids)

    trace_inputs = {
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "purpose": "native operation oracle; not a RivalSim acceptance run",
        "authority_identity_sha256": identity["authority_identity_sha256"],
        "case_selection": source_selection,
        "failed_case_ids": list(failed_ids),
        "failed_case_selection_sha256": sha256_bytes(canonical_json_bytes(list(failed_ids))),
        "ticks": TRACE_TICKS,
        "diagnostic": {
            "source_path": "tools/rocketsim_diagnostic/trace.cpp",
            "source_sha256": sha256_file(source),
            "executable_sha256": sha256_file(executable),
        },
        "capture_protocol": {
            "discovery": (
                "one native custom run records staged state, actual contact callbacks, "
                "persistent-manifold evolution, exact per-tick Bullet BVH traversal, "
                "exact vehicle-ray inputs and results, and source-ordered wheel "
                "impulse application replay plus contact, friction, and split-impulse "
                "solver rows with before/after body state"
            ),
            "operations": (
                "one native custom_probe run evaluates the union of discovered body/face "
                "pairs at every tick and records source-ordered GJK, Voronoi cached state, "
                "the complete nine-guess GJK/EPA penetration-depth fallback sequence, "
                "callbacks, and retained manifolds"
            ),
        },
    }
    trace_identity_sha256 = sha256_bytes(canonical_json_bytes(trace_inputs))
    trace_dir = authority_cache.cache_dir / "deep-traces" / trace_identity_sha256
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_identity = {
        "trace_identity_sha256": trace_identity_sha256,
        "identity_inputs": trace_inputs,
    }
    _validate_or_write_json(trace_dir / "identity.json", trace_identity)

    started = time.perf_counter()
    case_manifests: list[dict[str, Any]] = []
    for ordinal, (corpus_index, case) in enumerate(selected):
        case_started = time.perf_counter()
        case_dir = trace_dir / "cases" / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        case_manifest_path = case_dir / "manifest.json"
        existing = _validated_case_manifest(
            case_manifest_path,
            trace_identity_sha256,
            case.case_id,
        )
        if existing is not None:
            case_manifests.append(existing)
            print(
                json.dumps(
                    {
                        "case": case.case_id,
                        "ordinal": ordinal,
                        "status": "resumed",
                    }
                ),
                flush=True,
            )
            continue

        base_values = _custom_values(case)
        discovery_command = [
            str(executable),
            str(geometry.source_root),
            "custom",
            str(TRACE_TICKS),
            *base_values,
        ]
        discovery = _run_native(discovery_command, case.case_id, "discovery")
        pairs = _candidate_pairs(discovery)
        discovery_path = case_dir / "discovery.jsonl.gz"
        _write_deterministic_gzip_bytes(discovery_path, discovery)

        operation_command = [
            str(executable),
            str(geometry.source_root),
            "custom_probe",
            str(TRACE_TICKS),
            *base_values,
            *(str(value) for pair in pairs for value in pair),
        ]
        command_line_length = len(subprocess.list2cmdline(operation_command))
        if command_line_length >= 30_000:
            raise RuntimeError(
                f"native operation command is too long for {case.case_id}: "
                f"{command_line_length} characters"
            )
        operations = _run_native(operation_command, case.case_id, "operations")
        operations_path = case_dir / "operations.jsonl.gz"
        _write_deterministic_gzip_bytes(operations_path, operations)

        command_inputs = {
            "case_id": case.case_id,
            "corpus_index": corpus_index,
            "custom_values": base_values,
            "candidate_pairs": [list(pair) for pair in pairs],
            "ticks": TRACE_TICKS,
        }
        case_manifest = {
            "trace_schema_version": TRACE_SCHEMA_VERSION,
            "trace_identity_sha256": trace_identity_sha256,
            "case_id": case.case_id,
            "corpus_index": corpus_index,
            "command_inputs_sha256": sha256_bytes(canonical_json_bytes(command_inputs)),
            "candidate_pair_count": len(pairs),
            "candidate_pairs": command_inputs["candidate_pairs"],
            "records": {
                "discovery_bvh_traversals": discovery.count(b'"record":"bvh_traversal"'),
                "actual_contact_callbacks": discovery.count(b'"record":"contact_added"'),
                "wheel_apply_replays": discovery.count(
                    b'"record":"wheel_apply_replay"'
                ),
                "vehicle_rays": discovery.count(b'"record":"vehicle_ray"'),
                "solver_rows": discovery.count(b'"record":"solver_row"'),
                "gjk_iteration_streams": operations.count(b'"record":"gjk_iterations"'),
                "gjk_results": operations.count(b'"record":"gjk_probe"'),
                "epa_results": operations.count(b'"record":"epa_probe"'),
                "epa_solver_traces": operations.count(b'"record":"epa_solver_trace"'),
            },
            "artifacts": [
                _artifact(trace_dir, discovery_path),
                _artifact(trace_dir, operations_path),
            ],
        }
        _write_json_atomic(case_manifest_path, case_manifest)
        case_manifests.append(case_manifest)
        print(
            json.dumps(
                {
                    "case": case.case_id,
                    "ordinal": ordinal,
                    "status": "generated_native_trace",
                    "candidate_pairs": len(pairs),
                    "seconds": round(time.perf_counter() - case_started, 3),
                }
            ),
            flush=True,
        )

    manifest = {
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "milestone": "v0.2.2",
        "purpose": "oracle_data_generation_not_rivalsim_acceptance",
        "status": "complete",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "trace_identity_sha256": trace_identity_sha256,
        "authority_identity_sha256": identity["authority_identity_sha256"],
        "installed_rocketsim_extension": extension,
        "case_count": len(case_manifests),
        "case_ids": [item["case_id"] for item in case_manifests],
        "candidate_pair_count": sum(int(item["candidate_pair_count"]) for item in case_manifests),
        "gjk_iteration_stream_count": sum(
            int(item["records"]["gjk_iteration_streams"]) for item in case_manifests
        ),
        "epa_solver_trace_count": sum(
            int(item["records"]["epa_solver_traces"]) for item in case_manifests
        ),
        "cases": [_root_case_entry(trace_dir, item) for item in case_manifests],
        "generation_seconds": time.perf_counter() - started,
    }
    manifest_path = trace_dir / "manifest.json"
    _write_json_atomic(manifest_path, manifest)
    complete = {
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "trace_identity_sha256": trace_identity_sha256,
        "manifest_sha256": sha256_file(manifest_path),
        "status": "complete",
    }
    _write_json_atomic(trace_dir / "complete.json", complete)
    print(
        json.dumps(
            {
                "purpose": manifest["purpose"],
                "status": manifest["status"],
                "trace_dir": str(trace_dir.resolve()),
                "trace_identity_sha256": trace_identity_sha256,
                "case_count": manifest["case_count"],
                "candidate_pair_count": manifest["candidate_pair_count"],
                "gjk_iteration_stream_count": manifest["gjk_iteration_stream_count"],
                "epa_solver_trace_count": manifest["epa_solver_trace_count"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


def _pilot_failure_selection(run_dir: Path) -> tuple[dict[str, Any], tuple[str, ...]]:
    run_path = run_dir / "run.json"
    aggregate_path = run_dir / "aggregate.json"
    if not run_path.is_file() or not aggregate_path.is_file():
        raise FileNotFoundError(f"pilot run is incomplete: {run_dir}")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    failed: set[str] = set()
    chunks = []
    for path in sorted(run_dir.glob("chunk-*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
        failed.update(
            str(record["case_id"]) for record in payload["records"] if not bool(record["pass"])
        )
        chunks.append(
            {
                "file": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    ordered = tuple(
        record["case_id"]
        for path in sorted(run_dir.glob("chunk-*.json.gz"))
        for record in _read_chunk_records(path)
        if record["case_id"] in failed
    )
    if len(ordered) != len(failed):
        raise RuntimeError("pilot failed-case ordering is not unique")
    expected_count = int(aggregate["counts"]["failed_cases"])
    if len(ordered) != expected_count:
        raise RuntimeError(
            f"pilot failure count mismatch: found {len(ordered)}, "
            f"aggregate reports {expected_count}"
        )
    source = {
        "selection_sha256": run["selection_sha256"],
        "corpus_sha256": run["corpus_sha256"],
        "aggregate_sha256": sha256_file(aggregate_path),
        "run_sha256": sha256_file(run_path),
        "chunks": chunks,
        "reported_failed_case_count": expected_count,
    }
    return source, ordered


def _explicit_case_selection(
    case_ids: list[str],
    identity: dict[str, Any],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    ordered = tuple(str(case_id) for case_id in case_ids)
    if not ordered:
        raise RuntimeError("explicit trace selection is empty")
    if len(set(ordered)) != len(ordered):
        raise RuntimeError("explicit trace case IDs must be unique")
    corpus = identity["identity_inputs"]["corpus"]
    source = {
        "mode": "explicit_frozen_corpus_cases",
        "case_ids": list(ordered),
        "case_ids_sha256": sha256_bytes(canonical_json_bytes(list(ordered))),
        "corpus_sha256": corpus["corpus_sha256"],
        "generator_source_sha256": corpus["generator_source_sha256"],
        "generator_config_sha256": corpus["generator_config_sha256"],
        "seed": corpus["seed"],
    }
    return source, ordered


def _read_chunk_records(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)["records"]


def _custom_values(case: BreadthCase) -> list[str]:
    if any(float(value) != 0.0 for value in case.controls[2:7]):
        raise RuntimeError(
            f"diagnostic custom protocol cannot represent controls for {case.case_id}"
        )
    matrix = quat_to_matrix(case.quaternion)
    values = [
        *case.position,
        *case.velocity,
        *case.angular_velocity,
        *matrix[:, 0],
        *matrix[:, 1],
        *matrix[:, 2],
        case.controls[0],
        case.controls[1],
        int(case.controls[7]),
    ]
    return [_float_argument(value) for value in values]


def _float_argument(value: Any) -> str:
    if isinstance(value, int):
        return str(value)
    return format(float(value), ".9g")


def _run_native(command: list[str], case_id: str, phase: str) -> bytes:
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")[-4000:]
        raise RuntimeError(
            f"native {phase} trace failed for {case_id} with exit {result.returncode}: {stderr}"
        )
    if not result.stdout:
        raise RuntimeError(f"native {phase} trace emitted no data for {case_id}")
    return result.stdout


def _candidate_pairs(payload: bytes) -> tuple[tuple[int, int], ...]:
    pairs: set[tuple[int, int]] = set()
    for line in payload.splitlines():
        if not line.startswith(b'{"record":"bvh_traversal"'):
            continue
        record = json.loads(line)
        body = int(record["world_body_index"])
        pairs.update((body, int(face)) for face in record["faces"])
    return tuple(sorted(pairs))


def _validated_case_manifest(
    path: Path,
    trace_identity_sha256: str,
    case_id: str,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("trace_identity_sha256") != trace_identity_sha256
        or payload.get("case_id") != case_id
    ):
        return None
    trace_dir = path.parents[2]
    for artifact in payload.get("artifacts", []):
        artifact_path = trace_dir / str(artifact["path"])
        if not artifact_path.is_file() or sha256_file(artifact_path) != artifact["sha256"]:
            return None
    return payload


def _artifact(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _root_case_entry(root: Path, item: dict[str, Any]) -> dict[str, Any]:
    manifest = Path("cases") / str(item["case_id"]) / "manifest.json"
    return {
        "case_id": item["case_id"],
        "corpus_index": item["corpus_index"],
        "candidate_pair_count": item["candidate_pair_count"],
        "records": item["records"],
        "manifest": manifest.as_posix(),
        "manifest_sha256": sha256_file(root / manifest),
    }


def _validate_or_write_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise RuntimeError(f"trace identity mismatch: {path.resolve()}")
        return
    _write_json_atomic(path, payload)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(json.dumps(payload, indent=2).encode("utf-8") + b"\n")
    temporary.replace(path)


def _write_deterministic_gzip_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with (
        temporary.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        io.BufferedWriter(compressed) as stream,
    ):
        stream.write(payload)
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
