"""Cache one source-level native ball/world trace under the Phase A identity."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from rivalsim.arena import ArenaGeometry
from rivalsim.v03_corpus import generate_phase_a_cases
from rivalsim.v03_oracle_cache import (
    build_phase_a_identity,
    phase_cache_dir,
    sha256_bytes,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collision-dir", required=True)
    parser.add_argument("--oracle-cache-root", type=Path, required=True)
    parser.add_argument("--diagnostic-exe", type=Path, required=True)
    parser.add_argument("--corpus-index", type=int, default=8020)
    parser.add_argument("--ticks", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    _catalog, cases = generate_phase_a_cases(geometry)
    if not 0 <= args.corpus_index < len(cases):
        raise IndexError("trace corpus index outside frozen Phase A corpus")
    case = cases[args.corpus_index]
    identity = build_phase_a_identity(geometry, cases)
    cache_dir = phase_cache_dir(args.oracle_cache_root, identity)
    trace_dir = cache_dir / "deep-traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{case.case_id}-tick{args.ticks}"
    payload_path = trace_dir / f"{stem}.jsonl.gz"
    metadata_path = trace_dir / f"{stem}.json"
    if payload_path.exists() or metadata_path.exists():
        if not payload_path.is_file() or not metadata_path.is_file():
            raise RuntimeError("incomplete Phase A deep-trace cache entry")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["authority_identity_sha256"] != identity["authority_identity_sha256"]:
            raise RuntimeError("deep-trace authority identity mismatch")
        if metadata["payload"]["sha256"] != sha256_file(payload_path):
            raise RuntimeError("deep-trace payload hash mismatch")
        print(json.dumps({**metadata, "status": "CACHED"}, indent=2))
        return 0

    command = [
        str(args.diagnostic_exe.resolve()),
        str(geometry.source_root),
        "ball_custom",
        str(args.ticks),
        *(format(float(value), ".17g") for value in case.position),
        *(format(float(value), ".17g") for value in case.velocity),
        *(format(float(value), ".17g") for value in case.angular_velocity),
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    records = [
        json.loads(line)
        for line in completed.stdout.splitlines()
        if line.strip()
    ]
    kinds = Counter(str(record.get("record")) for record in records)
    required = {"ball_state", "contact_added", "manifold_after_add", "solver_row"}
    if not required.issubset(kinds):
        raise RuntimeError(f"native trace missing required records: {required - set(kinds)}")
    payload = gzip.compress(
        b"".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
            + b"\n"
            for record in records
        ),
        compresslevel=9,
        mtime=0,
    )
    _write_atomic(payload_path, payload)
    repository_root = Path(__file__).parents[1]
    trace_source = repository_root / "tools" / "rocketsim_diagnostic" / "trace.cpp"
    cmake_source = repository_root / "tools" / "rocketsim_diagnostic" / "CMakeLists.txt"
    metadata = {
        "schema_version": 1,
        "milestone": "v0.3",
        "phase": "A_ball_world",
        "status": "COMPLETE_NATIVE_DEEP_TRACE",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "authority_identity_sha256": identity["authority_identity_sha256"],
        "case": {
            "corpus_index": args.corpus_index,
            "case_id": case.case_id,
            "ticks": args.ticks,
            "source_input_sha256": sha256_bytes(
                json.dumps(
                    {
                        "position": case.position.astype(float).tolist(),
                        "velocity": case.velocity.astype(float).tolist(),
                        "angular_velocity": case.angular_velocity.astype(float).tolist(),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ),
        },
        "tooling": {
            "trace_cpp_sha256": sha256_file(trace_source),
            "cmake_lists_sha256": sha256_file(cmake_source),
            "diagnostic_exe_sha256": sha256_file(args.diagnostic_exe),
        },
        "record_counts": dict(sorted(kinds.items())),
        "payload": {
            "path": payload_path.name,
            "size_bytes": len(payload),
            "sha256": sha256_file(payload_path),
        },
    }
    _write_atomic(
        metadata_path,
        json.dumps(metadata, indent=2, sort_keys=True).encode("utf-8"),
    )
    print(json.dumps(metadata, indent=2))
    return 0


def _write_atomic(path: Path, payload: bytes) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        stream.write(payload)
        temp_path = Path(stream.name)
    try:
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
