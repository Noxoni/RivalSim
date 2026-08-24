"""Build compact tracked evidence for v0.2.2 native oracle-data generation."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rivalsim.arena import ArenaGeometry
from rivalsim.dfh_breadth import build_breadth_catalog, generate_breadth_cases
from rivalsim.v022_oracle_cache import (
    build_authority_identity,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    verify_complete_authority_cache,
)

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_RETAINED_GOAL_FACES = [6262, 6235, 6035, 6040]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collision-dir", required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--goal-diagnostic", type=Path, required=True)
    parser.add_argument("--cached-pilot-dir", type=Path, required=True)
    parser.add_argument("--full-run-dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "v0.2.2" / "oracle_data.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    cases = generate_breadth_cases(build_breadth_catalog(geometry))
    identity = build_authority_identity(geometry, cases)
    authority_manifest = verify_complete_authority_cache(
        args.cache_root,
        identity,
        cases,
    )
    cache_dir = args.cache_root.resolve() / str(identity["authority_identity_sha256"])
    trace = _verify_trace_cache(args.trace_dir.resolve(), identity)
    goal = _goal_evidence(args.goal_diagnostic.resolve())
    cached_pilot = _cached_pilot_evidence(
        args.cached_pilot_dir.resolve(),
        identity,
        trace,
    )
    full_acceptance = None
    if args.full_run_dir is not None:
        full_acceptance = _full_acceptance_evidence(
            args.full_run_dir.resolve(),
            identity,
        )
    chunks = authority_manifest["chunks"]
    source_paths = (
        "rivalsim/dfh_breadth.py",
        "rivalsim/v022_oracle_cache.py",
        "benchmarks/build_v022_oracle_cache.py",
        "benchmarks/build_v022_deep_trace_cache.py",
        "benchmarks/build_v022_oracle_evidence.py",
        "benchmarks/build_v022_acceptance_evidence.py",
        "benchmarks/run_v022_breadth.py",
        "tools/rocketsim_diagnostic/trace.cpp",
        "docs/V0_2_2_ORACLE_CACHE.md",
        "docs/V0_2_2_RESULTS.md",
        "docs/REPRODUCING_V0_2_2.md",
    )
    evidence = {
        "schema_version": 1,
        "milestone": "v0.2.2",
        "evidence_kind": "native_oracle_data",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "classification": "ORACLE_DATA_COMPLETE_NOT_ACCEPTANCE",
        "authority": identity,
        "trajectory_cache": {
            "status": authority_manifest["status"],
            "purpose": authority_manifest["purpose"],
            "case_count": authority_manifest["case_count"],
            "captured_ticks": authority_manifest["captured_ticks"],
            "frame_count": authority_manifest["frame_count"],
            "chunk_count": authority_manifest["chunk_count"],
            "compressed_size_bytes": authority_manifest["frozen_corpus"]["size_bytes"]
            + sum(int(chunk["size_bytes"]) for chunk in chunks),
            "frozen_corpus": authority_manifest["frozen_corpus"],
            "chunk_set_sha256": sha256_bytes(canonical_json_bytes(chunks)),
            "manifest_sha256": sha256_file(cache_dir / "manifest.json"),
            "complete_marker_sha256": sha256_file(cache_dir / "complete.json"),
        },
        "deep_native_traces": trace,
        "diagnostic_goal_witness_gate": goal,
        "cached_representative_pilot": cached_pilot,
        "complete_acceptance_reference": full_acceptance,
        "tracked_source": [
            {
                "path": relative,
                "size_bytes": (ROOT / relative).stat().st_size,
                "sha256": sha256_file(ROOT / relative),
            }
            for relative in source_paths
        ],
        "scope_boundary": {
            "native_oracle_generation_complete": True,
            "rivalsim_acceptance_gate_run": full_acceptance is not None,
            "complete_breadth_gpu_gate_run": full_acceptance is not None,
            "representative_cached_gpu_pilot_run": True,
            "v0_2_2_parity_verdict_claimed": full_acceptance is not None,
            "v0_3_begun": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "classification": evidence["classification"],
                "authority_identity_sha256": identity["authority_identity_sha256"],
                "case_count": authority_manifest["case_count"],
                "deep_trace_case_count": trace["case_count"],
            },
            indent=2,
        )
    )
    return 0


def _cached_pilot_evidence(
    pilot_dir: Path,
    authority_identity: dict[str, Any],
    trace: dict[str, Any],
) -> dict[str, Any]:
    aggregate_path = pilot_dir / "aggregate.json"
    run_path = pilot_dir / "run.json"
    for path in (aggregate_path, run_path):
        if not path.is_file():
            raise FileNotFoundError(f"cached representative pilot is incomplete: {path}")
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    run = json.loads(run_path.read_text(encoding="utf-8"))
    aggregate_run = aggregate["run"]
    expected_authority = authority_identity["authority_identity_sha256"]
    if (
        aggregate_run.get("oracle_execution") != "cached_native_authority_only"
        or aggregate_run.get("oracle_authority_identity_sha256") != expected_authority
        or run.get("oracle_execution") != "cached_native_authority_only"
        or run.get("oracle_authority_identity_sha256") != expected_authority
    ):
        raise RuntimeError("representative pilot did not use the expected cached authority")
    if (
        aggregate_run.get("selection_sha256") != trace["pilot_selection_sha256"]
        or aggregate_run.get("corpus_sha256")
        != authority_identity["identity_inputs"]["corpus"]["corpus_sha256"]
    ):
        raise RuntimeError("representative pilot selection or corpus binding changed")
    chunks = []
    for path in sorted(pilot_dir.glob("chunk-*.json.gz")):
        chunks.append(
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not chunks:
        raise RuntimeError("representative pilot has no measured chunks")
    return {
        "oracle_execution": aggregate_run["oracle_execution"],
        "authority_identity_sha256": expected_authority,
        "corpus_sha256": aggregate_run["corpus_sha256"],
        "selection_sha256": aggregate_run["selection_sha256"],
        "selected_case_count": aggregate["counts"]["selected_starting_states"],
        "checkpoint_comparisons": aggregate["counts"]["checkpoint_comparisons"],
        "hard_mismatch_events": aggregate["counts"]["hard_mismatch_events"],
        "numeric_failure_events": aggregate["counts"]["numeric_failure_events"],
        "failed_cases": aggregate["counts"]["failed_cases"],
        "pass_fail_by_horizon": aggregate["pass_fail_by_horizon"],
        "gate": aggregate["gate"],
        "aggregate_sha256": sha256_file(aggregate_path),
        "run_sha256": sha256_file(run_path),
        "chunk_count": len(chunks),
        "chunk_set_sha256": sha256_bytes(canonical_json_bytes(chunks)),
        "claim_boundary": (
            "representative cached-path validation only; the complete acceptance verdict is "
            "recorded separately"
            if aggregate["gate"]["selected_run_pass"]
            else "representative cached-path validation only; blocking discrepancies remain"
        ),
    }


def _full_acceptance_evidence(
    run_dir: Path,
    authority_identity: dict[str, Any],
) -> dict[str, Any]:
    aggregate_path = run_dir / "aggregate.json"
    run_path = run_dir / "run.json"
    for path in (aggregate_path, run_path):
        if not path.is_file():
            raise FileNotFoundError(f"complete acceptance run is incomplete: {path}")
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    run = aggregate["run"]
    counts = aggregate["counts"]
    gate = aggregate["gate"]
    if (
        run.get("selection_kind") != "complete"
        or run.get("selected_case_count") != 39236
        or run.get("oracle_execution") != "cached_native_authority_only"
        or run.get("oracle_authority_identity_sha256")
        != authority_identity["authority_identity_sha256"]
        or run.get("corpus_sha256")
        != authority_identity["identity_inputs"]["corpus"]["corpus_sha256"]
    ):
        raise RuntimeError("complete acceptance run identity or selection mismatch")
    if (
        counts.get("hard_mismatch_events") != 0
        or counts.get("numeric_failure_events") != 0
        or counts.get("failed_cases") != 0
        or not gate.get("complete_v022_gate_pass")
        or gate.get("classification") != "PASS_GREEN"
    ):
        raise RuntimeError("complete acceptance run is not PASS_GREEN")
    chunks = [
        {
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(run_dir.glob("chunk-*.json.gz"))
    ]
    if len(chunks) != 154:
        raise RuntimeError("complete acceptance run does not contain 154 chunks")
    return {
        "oracle_execution": run["oracle_execution"],
        "authority_identity_sha256": run["oracle_authority_identity_sha256"],
        "corpus_sha256": run["corpus_sha256"],
        "selection_sha256": run["selection_sha256"],
        "selected_case_count": counts["selected_starting_states"],
        "checkpoint_comparisons": counts["checkpoint_comparisons"],
        "hard_mismatch_events": counts["hard_mismatch_events"],
        "numeric_failure_events": counts["numeric_failure_events"],
        "failed_cases": counts["failed_cases"],
        "gate": gate,
        "aggregate_sha256": sha256_file(aggregate_path),
        "run_sha256": sha256_file(run_path),
        "chunk_count": len(chunks),
        "chunk_set_sha256": sha256_bytes(canonical_json_bytes(chunks)),
        "claim_boundary": "complete frozen v0.2.2 static-world acceptance result",
    }


def _verify_trace_cache(
    trace_dir: Path,
    authority_identity: dict[str, Any],
) -> dict[str, Any]:
    identity_path = trace_dir / "identity.json"
    manifest_path = trace_dir / "manifest.json"
    complete_path = trace_dir / "complete.json"
    for path in (identity_path, manifest_path, complete_path):
        if not path.is_file():
            raise FileNotFoundError(f"deep trace cache is incomplete: {path}")
    trace_identity = json.loads(identity_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if complete.get("manifest_sha256") != sha256_file(manifest_path):
        raise RuntimeError("deep trace completion marker does not match manifest")
    if (
        manifest.get("status") != "complete"
        or manifest.get("trace_identity_sha256") != trace_identity.get("trace_identity_sha256")
        or manifest.get("authority_identity_sha256")
        != authority_identity["authority_identity_sha256"]
    ):
        raise RuntimeError("deep trace identity or authority binding mismatch")
    artifact_count = 0
    compressed_size = 0
    for case in manifest["cases"]:
        case_manifest_path = trace_dir / case["manifest"]
        if sha256_file(case_manifest_path) != case["manifest_sha256"]:
            raise RuntimeError(f"deep trace case manifest hash mismatch: {case_manifest_path}")
        case_manifest = json.loads(case_manifest_path.read_text(encoding="utf-8"))
        for artifact in case_manifest["artifacts"]:
            path = trace_dir / artifact["path"]
            if sha256_file(path) != artifact["sha256"]:
                raise RuntimeError(f"deep trace artifact hash mismatch: {path}")
            artifact_count += 1
            compressed_size += int(artifact["size_bytes"])
    return {
        "status": manifest["status"],
        "trace_identity_sha256": manifest["trace_identity_sha256"],
        "trace_identity_inputs_sha256": sha256_bytes(
            canonical_json_bytes(trace_identity["identity_inputs"])
        ),
        "pilot_selection_sha256": trace_identity["identity_inputs"]["pilot_failure_selection"][
            "selection_sha256"
        ],
        "case_count": manifest["case_count"],
        "case_ids": manifest["case_ids"],
        "candidate_pair_count": manifest["candidate_pair_count"],
        "gjk_iteration_stream_count": manifest["gjk_iteration_stream_count"],
        "epa_solver_trace_count": manifest["epa_solver_trace_count"],
        "artifact_count": artifact_count,
        "compressed_size_bytes": compressed_size,
        "manifest_sha256": sha256_file(manifest_path),
        "complete_marker_sha256": sha256_file(complete_path),
    }


def _goal_evidence(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tick = payload["trace"][0]
    faces = [int(contact["face"]) for contact in tick["rivalsim"]["contacts"]]
    if faces != EXPECTED_RETAINED_GOAL_FACES:
        raise RuntimeError(f"diagnostic goal retained faces changed: {faces}")
    return {
        "case_id": payload["case"]["case_id"],
        "tick": tick["tick"],
        "retained_faces": faces,
        "expected_retained_faces": EXPECTED_RETAINED_GOAL_FACES,
        "retained_face_gate_pass": True,
        "numeric_deltas": tick["deltas"],
        "source_artifact_sha256": sha256_file(path),
        "claim_boundary": (
            "diagnostic retained-witness and reported-delta evidence only; not a complete "
            "breadth or acceptance result"
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())
