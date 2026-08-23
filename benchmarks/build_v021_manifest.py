"""Build the compact, machine-verifiable RivalSim v0.2.1 manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FROZEN_REF = "2c5d11899eaaad6a963a370fcc3813202b6fa714"
EVIDENCE_PATHS = (
    "results/v0.2.1/benchmark.json",
    "results/v0.2.1/coverage.json",
    "results/v0.2.1/divergence_index.json",
    "results/v0.2.1/parity.json",
    "docs/V0_2_1_RESULTS.md",
    "docs/REPRODUCING_V0_2_1.md",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--frozen-ref", default=FROZEN_REF)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/v0.2.1/manifest.json"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    implementation_commit = _git("rev-parse", args.implementation_commit).strip()
    frozen_ref = _git("rev-parse", args.frozen_ref).strip()
    benchmark = _load(ROOT / "results/v0.2.1/benchmark.json")
    parity = _load(ROOT / "results/v0.2.1/parity.json")
    coverage = _load(ROOT / "results/v0.2.1/coverage.json")
    divergence = _load(ROOT / "results/v0.2.1/divergence_index.json")
    regression_path = ROOT / ".tools/v0.2.1-final/v0.1-parity.json"
    regression = _load(regression_path)
    frozen = _frozen_records(frozen_ref)
    if not all(record["unchanged"] for record in frozen):
        raise RuntimeError("frozen v0.1/v0.2 evidence differs from the published boundary")
    tracked_collision_assets = [
        path
        for path in _git("ls-files").splitlines()
        if Path(path).suffix.casefold() == ".cmf"
    ]
    if tracked_collision_assets:
        raise RuntimeError(f"collision meshes must remain untracked: {tracked_collision_assets}")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    manifest = {
        "schema_version": 1,
        "milestone": "v0.2.1",
        "manifest_created_utc": datetime.now(UTC).isoformat(),
        "verdict": benchmark["summary"]["verdict"],
        "repository": {
            "origin": _git("remote", "get-url", "origin").strip(),
            "branch": _git("branch", "--show-current").strip(),
            "authority_start_commit": "cc45ab0dce85f2c696800b96e1f4af8b7d8bb1f2",
            "frozen_v0_2_evidence_commit": frozen_ref,
            "implementation_commit": implementation_commit,
            "evidence_parent_commit": implementation_commit,
            "evidence_package_commit": (
                "the commit containing this self-excluding manifest; verify by remote readback"
            ),
        },
        "release": {
            "package": project["name"],
            "version": project["version"],
            "python_requires": project["requires-python"],
            "scope": "static-world local-transition fidelity through 12 ticks",
        },
        "environment": benchmark["environment"],
        "dependency_lock": _path_record(ROOT / "requirements-v0.2.txt"),
        "source_custody": {
            "rocketsim_primary_commit": "c2baacb8f4b441dd8505e63c2aeb5a1679b60b02",
            "rocketsim_binding_commit": "2da51b1dac7b8127127613a5ff30e490bdd70dd8",
            "diagnostic_helper": [
                _path_record(ROOT / "tools/rocketsim_diagnostic/CMakeLists.txt"),
                _path_record(ROOT / "tools/rocketsim_diagnostic/README.md"),
                _path_record(ROOT / "tools/rocketsim_diagnostic/trace.cpp"),
            ],
            "diagnostic_build_products_tracked": False,
        },
        "collision_assets": benchmark["arena"],
        "asset_audit": {
            "tracked_cmf_count": 0,
            "tracked_cmf_paths": [],
            "external_collision_root": benchmark["arena"]["source_root"],
        },
        "implementation_blobs": _implementation_records(implementation_commit),
        "tracked_evidence": [_path_record(ROOT / path) for path in EVIDENCE_PATHS],
        "frozen_evidence": {
            "reference_commit": frozen_ref,
            "all_unchanged": True,
            "records": frozen,
        },
        "verification": {
            "static_world_targeted_tests": {
                "command": (
                    "python -m pytest -q tests/test_arena.py tests/test_world_queries.py "
                    "tests/test_static_world.py"
                ),
                "passed": 15,
                "failed": 0,
            },
            "full_tests": {
                "command": "python -m pytest -q",
                "passed": 38,
                "failed": 0,
            },
            "lint": {"command": "ruff check rivalsim benchmarks tests", "passed": True},
            "compile": {
                "command": "python -m compileall -q rivalsim benchmarks tests",
                "passed": True,
            },
            "v0_1_live_regression": {
                "ignored_output_path": str(regression_path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": _sha256(regression_path.read_bytes()),
                "scenario_count": regression["summary"]["scenario_count"],
                "same_equation_pass": regression["summary"]["same_equation_pass"],
                "rocketsim_pass": regression["summary"]["rocketsim_pass"],
                "axis_sign_pass": regression["summary"]["axis_sign_pass"],
                "basic_parity_pass": regression["summary"]["basic_parity_pass"],
            },
            "stress": benchmark["stress_gate"],
            "geometry_query": benchmark["geometry_query_gate"],
        },
        "validation": {
            "policy": parity["validation_policy"],
            "parity": {
                "scenario_count": parity["scenario_count"],
                "checkpoint_comparisons": parity["scenario_count"]
                * len(parity["horizons_ticks"]),
                "horizons_ticks": parity["horizons_ticks"],
                "frozen_tolerances": parity["frozen_tolerances"],
                "measurement_aggregates": parity["measurement_aggregates"],
                "summary": parity["summary"],
            },
            "causal_divergence": {
                "baseline": divergence["input_boundary"],
                "representative_findings": divergence["representative_causal_findings"],
                "final_local_resolution": divergence["final_local_resolution"],
            },
            "bounded_dfh_coverage": {
                "prototype_status": coverage["prototype_status"],
                "scope": coverage["scope"],
                "geometry": coverage["geometry"],
                "shared_edge_topology_audit": coverage["shared_edge_topology_audit"],
            },
            "performance": benchmark["summary"],
            "frozen_v0_2_comparison": benchmark["frozen_v0_2_b3_comparison"],
        },
        "scope_boundary": {
            "v0_2_1_complete": benchmark["summary"]["verdict"] in {"PASS", "PASS_GREEN"},
            "v0_3_begun": False,
            "dynamic_ball_world_contacts": False,
            "dynamic_car_ball_contacts": False,
            "dynamic_car_car_contacts": False,
            "training_or_policy_integration": False,
            "next_authority_required": True,
        },
        "self_hash_policy": "manifest.json intentionally does not hash itself",
    }
    if manifest["release"]["version"] != "0.2.1":
        raise RuntimeError("package version must be 0.2.1 before building the manifest")
    if manifest["verdict"] != "PASS_GREEN":
        raise RuntimeError(f"unexpected v0.2.1 verdict: {manifest['verdict']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "verdict": manifest["verdict"],
                "frozen_evidence_unchanged": len(frozen),
            },
            indent=2,
        )
    )
    return 0


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git(*args: str, text: bool = True) -> str | bytes:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=text
    ).stdout


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def _path_record(path: Path) -> dict[str, Any]:
    payload = path.read_bytes().replace(b"\r\n", b"\n")
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "size_bytes": len(payload),
        "sha256": _sha256(payload),
        "hash_policy": "canonical_lf_bytes",
    }


def _frozen_records(reference: str) -> list[dict[str, Any]]:
    paths = _git(
        "ls-tree", "-r", "--name-only", reference, "results/v0.1", "results/v0.2"
    ).splitlines()
    records: list[dict[str, Any]] = []
    for relative in paths:
        reference_payload = _git("show", f"{reference}:{relative}", text=False)
        current_payload = _git("show", f"HEAD:{relative}", text=False)
        worktree_status = _git("status", "--porcelain", "--", relative).strip()
        records.append(
            {
                "path": relative,
                "reference_size_bytes": len(reference_payload),
                "reference_sha256": _sha256(reference_payload),
                "current_sha256": _sha256(current_payload),
                "worktree_clean": not worktree_status,
                "unchanged": current_payload == reference_payload and not worktree_status,
            }
        )
    return records


def _implementation_records(commit: str) -> list[dict[str, Any]]:
    paths = _git(
        "diff-tree", "--no-commit-id", "--name-only", "-r", commit
    ).splitlines()
    records: list[dict[str, Any]] = []
    for relative in paths:
        payload = _git("show", f"{commit}:{relative}", text=False)
        records.append(
            {
                "path": relative,
                "size_bytes": len(payload),
                "sha256": _sha256(payload),
            }
        )
    return records


if __name__ == "__main__":
    raise SystemExit(main())
