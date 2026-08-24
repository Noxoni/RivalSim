"""Build compact, validated release evidence for the v0.2.2 acceptance gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-run-dir", type=Path, required=True)
    parser.add_argument("--pilot-run-dir", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--v01-regression", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, action="append", default=[])
    parser.add_argument("--focused-diagnostic", type=Path, action="append", default=[])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "v0.2.2",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def _chunk_custody(run_dir: Path) -> dict[str, Any]:
    chunks = sorted(run_dir.glob("chunk-*.json.gz"))
    if not chunks:
        raise RuntimeError(f"no completed chunks found in {run_dir}")
    entries = [
        {
            "file": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in chunks
    ]
    return {
        "chunk_count": len(entries),
        "compressed_size_bytes": sum(int(item["size_bytes"]) for item in entries),
        "chunk_set_sha256": _canonical_sha256(entries),
    }


def _validate_full(aggregate: dict[str, Any]) -> None:
    run = aggregate.get("run", {})
    counts = aggregate.get("counts", {})
    gate = aggregate.get("gate", {})
    if aggregate.get("milestone") != "v0.2.2":
        raise RuntimeError("full run is not a v0.2.2 artifact")
    if run.get("selection_kind") != "complete":
        raise RuntimeError("acceptance evidence requires the complete corpus")
    if run.get("oracle_execution") != "cached_native_authority_only":
        raise RuntimeError("acceptance run did not use only cached native authority")
    if int(run.get("selected_case_count", -1)) != 39236:
        raise RuntimeError("acceptance run did not contain all 39,236 cases")
    if int(counts.get("checkpoint_comparisons", -1)) != 156944:
        raise RuntimeError("acceptance run did not contain all checkpoint comparisons")
    if any(
        int(counts.get(field, -1)) != 0
        for field in ("hard_mismatch_events", "numeric_failure_events", "failed_cases")
    ):
        raise RuntimeError("acceptance run contains blocking failures")
    if not gate.get("selection_complete") or not gate.get("complete_v022_gate_pass"):
        raise RuntimeError("complete v0.2.2 gate did not pass")
    if gate.get("classification") != "PASS_GREEN":
        raise RuntimeError("complete v0.2.2 gate is not PASS_GREEN")


def _validate_pilot(aggregate: dict[str, Any], full: dict[str, Any]) -> None:
    run = aggregate.get("run", {})
    counts = aggregate.get("counts", {})
    gate = aggregate.get("gate", {})
    if run.get("oracle_execution") != "cached_native_authority_only":
        raise RuntimeError("representative run did not use only cached native authority")
    if int(run.get("selected_case_count", -1)) != 1043:
        raise RuntimeError("representative run did not contain all 1,043 selected cases")
    for field in ("oracle_authority_identity_sha256", "corpus_sha256"):
        if run.get(field) != full["run"].get(field):
            raise RuntimeError(f"representative/full identity mismatch: {field}")
    if any(
        int(counts.get(field, -1)) != 0
        for field in ("hard_mismatch_events", "numeric_failure_events", "failed_cases")
    ):
        raise RuntimeError("representative run contains blocking failures")
    if not gate.get("selected_run_pass") or gate.get("classification") != "PILOT_PASS":
        raise RuntimeError("representative gate did not pass")


def _compact_run(run_dir: Path, aggregate: dict[str, Any]) -> dict[str, Any]:
    aggregate_path = run_dir / "aggregate.json"
    run_path = run_dir / "run.json"
    if not aggregate_path.is_file() or not run_path.is_file():
        raise RuntimeError(f"run is incomplete: {run_dir}")
    return {
        "aggregate_sha256": _sha256(aggregate_path),
        "run_sha256": _sha256(run_path),
        **_chunk_custody(run_dir),
        "run": aggregate["run"],
        "counts": aggregate["counts"],
        "pass_fail_by_horizon": aggregate["pass_fail_by_horizon"],
        "gate": aggregate["gate"],
        "run_seconds": aggregate["run_seconds"],
    }


def _compact_benchmark(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    summary = value.get("summary", {})
    stress = value.get("stress_gate", {})
    if value.get("milestone") != "v0.2.2":
        raise RuntimeError("benchmark is not a v0.2.2 artifact")
    if summary.get("verdict") != "PASS_GREEN" or not summary.get("parity_gate_pass"):
        raise RuntimeError("benchmark verdict is not PASS_GREEN with parity")
    if not summary.get("hot_loop_gpu_resident") or not stress.get("stress_gate_pass"):
        raise RuntimeError("benchmark residency or stress gate failed")
    return {
        "schema_version": 1,
        "milestone": "v0.2.2",
        "evidence_kind": "performance",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_artifact_sha256": _sha256(path),
        "configuration": value["configuration"],
        "environment": value["environment"],
        "geometry_query_gate": value["geometry_query_gate"],
        "stress_gate": stress,
        "b3_points": value["variants"]["B3"],
        "comparison_to_v0_2_1": value["frozen_v0_2_b3_comparison"],
        "summary": summary,
    }


def _compact_regression(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    summary = value.get("summary", {})
    required = (
        "same_equation_pass",
        "rocketsim_pass",
        "axis_sign_pass",
        "basic_parity_pass",
    )
    if int(summary.get("scenario_count", -1)) != 27:
        raise RuntimeError("v0.1 regression does not contain 27 scenarios")
    if not all(bool(summary.get(field)) for field in required):
        raise RuntimeError("v0.1 regression failed")
    return {
        "schema_version": 1,
        "milestone": "v0.2.2",
        "evidence_kind": "v0.1_live_regression",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_artifact_sha256": _sha256(path),
        "source_milestone": value.get("milestone"),
        "environment": value.get("environment"),
        "rocketsim": value.get("rocketsim"),
        "horizons_ticks": value.get("horizons_ticks"),
        "tolerances": value.get("tolerances"),
        "summary": summary,
    }


def _trace_reference(path: Path, authority_sha256: str) -> dict[str, Any]:
    identity_path = path / "identity.json"
    manifest_path = path / "manifest.json"
    complete_path = path / "complete.json"
    identity = _json(identity_path)
    manifest = _json(manifest_path)
    complete = _json(complete_path)
    if complete.get("manifest_sha256") != _sha256(manifest_path):
        raise RuntimeError(f"trace completion marker mismatch: {path}")
    if (
        manifest.get("status") != "complete"
        or manifest.get("trace_identity_sha256") != identity.get("trace_identity_sha256")
        or manifest.get("authority_identity_sha256") != authority_sha256
    ):
        raise RuntimeError(f"trace identity or authority mismatch: {path}")
    inputs = identity["identity_inputs"]
    return {
        "trace_identity_sha256": manifest["trace_identity_sha256"],
        "authority_identity_sha256": manifest["authority_identity_sha256"],
        "case_count": manifest["case_count"],
        "case_ids": manifest["case_ids"],
        "candidate_pair_count": manifest["candidate_pair_count"],
        "gjk_iteration_stream_count": manifest["gjk_iteration_stream_count"],
        "epa_solver_trace_count": manifest["epa_solver_trace_count"],
        "trace_schema_version": inputs["trace_schema_version"],
        "diagnostic_source_sha256": inputs["diagnostic"]["source_sha256"],
        "diagnostic_executable_sha256": inputs["diagnostic"]["executable_sha256"],
        "manifest_sha256": _sha256(manifest_path),
        "complete_marker_sha256": _sha256(complete_path),
    }


def _diagnostic_reference(path: Path) -> dict[str, Any]:
    value = _json(path)
    ticks = value["trace"]
    maximums = {
        metric: max(float(tick["deltas"][metric]) for tick in ticks)
        for metric in (
            "position_uu",
            "linear_velocity_uu_per_s",
            "angular_velocity_rad_per_s",
        )
    }
    bit_exact_state_ticks = [
        int(tick["tick"])
        for tick in ticks
        if all(float(tick["deltas"][metric]) == 0.0 for metric in maximums)
    ]
    return {
        "case_id": value["case"]["case_id"],
        "captured_ticks": [int(tick["tick"]) for tick in ticks],
        "bit_exact_rigid_state_ticks": bit_exact_state_ticks,
        "maximum_deltas": maximums,
        "source_artifact_sha256": _sha256(path),
    }


def main() -> int:
    args = parse_args()
    full_dir = args.full_run_dir.resolve()
    pilot_dir = args.pilot_run_dir.resolve()
    full = _json(full_dir / "aggregate.json")
    pilot = _json(pilot_dir / "aggregate.json")
    benchmark = _json(args.benchmark.resolve())
    regression = _json(args.v01_regression.resolve())
    _validate_full(full)
    _validate_pilot(pilot, full)

    coverage = full["coverage"]
    parity = {
        "schema_version": 1,
        "milestone": "v0.2.2",
        "evidence_kind": "cached_complete_static_world_acceptance",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "classification": "PASS_GREEN",
        "implementation": full["implementation"],
        "frozen_protocol": full["frozen_protocol"],
        "authority": {
            "identity_sha256": full["run"]["oracle_authority_identity_sha256"],
            "corpus_sha256": full["run"]["corpus_sha256"],
            "oracle_execution": full["run"]["oracle_execution"],
        },
        "complete_acceptance": _compact_run(full_dir, full),
        "representative_gate": _compact_run(pilot_dir, pilot),
        "numeric_error_distributions": full["numeric_error_distributions"],
        "worst_local_errors": full["worst_local_errors"],
        "failure_clusters": full["failure_clusters"],
        "coverage": {
            "definitions": coverage["definitions"],
            "topology_audited": coverage["topology_audited"],
            "generated": coverage["generated"],
            "actual_paired_target_contact": coverage["actual_paired_target_contact"],
            "analytic_planes": coverage["analytic_planes"],
            "important_regions": coverage["important_regions"],
        },
        "claim_boundary": (
            "complete frozen v0.2.2 Octane/Soccar static-world corpus only; "
            "dynamic bodies, ball physics, game rules, and v0.3 are excluded"
        ),
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "parity.json": parity,
        "benchmark.json": _compact_benchmark(args.benchmark.resolve(), benchmark),
        "regression.json": _compact_regression(args.v01_regression.resolve(), regression),
    }
    if args.trace_dir or args.focused_diagnostic:
        outputs["source_port.json"] = {
            "schema_version": 1,
            "milestone": "v0.2.2",
            "evidence_kind": "source_port_causal_trace_index",
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "authority_identity_sha256": parity["authority"]["identity_sha256"],
            "deep_native_trace_references": [
                _trace_reference(path.resolve(), parity["authority"]["identity_sha256"])
                for path in args.trace_dir
            ],
            "focused_final_diagnostics": [
                _diagnostic_reference(path.resolve()) for path in args.focused_diagnostic
            ],
            "source_backed_corrections": [
                "Bullet box-vs-static-triangle GJK/Voronoi/EPA witness operation order",
                "persistent-manifold refresh, four-point reduction, and internal-edge adjustment",
                "wheel ray, suspension, friction, solver-row, split-impulse, and integration order",
                "btGjkPairDetector internal-valid versus callback-report control flow",
                "RocketSim brake-force float32 multiplication order",
            ],
            "prohibited_fitting_added": False,
            "claim_boundary": (
                "causal source-port evidence only; the complete acceptance verdict is in "
                "parity.json"
            ),
        }
    for name, value in outputs.items():
        path = output_dir / name
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        print(f"{path} sha256={_sha256(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
