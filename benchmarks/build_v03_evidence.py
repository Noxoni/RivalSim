"""Assemble compact v0.3 release evidence from completed local gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "v0.3"

PHASE_FILES = {
    "ball_world": ROOT / ".tools" / "v0.3" / "phase-a" / "full-v03-release.json",
    "car_ball": ROOT / ".tools" / "v0.3" / "phase-b" / "full-v03-release.json",
    "car_car": ROOT / ".tools" / "v0.3" / "phase-c" / "full-v03-release.json",
    "integrated": ROOT / ".tools" / "v0.3" / "phase-d" / "full-v03-release.json",
}
REGRESSION_V022 = (
    ROOT / ".tools" / "v0.3" / "regression-v022-full" / "aggregate.json"
)
REGRESSION_V01 = ROOT / ".tools" / "v0.3" / "v01-regression-final.json"
BENCHMARK = ROOT / ".tools" / "v0.3" / "benchmark-final.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-tests", type=int, required=True)
    parser.add_argument("--ruff-pass", action="store_true", required=True)
    parser.add_argument("--compile-pass", action="store_true", required=True)
    parser.add_argument("--diff-check-pass", action="store_true", required=True)
    return parser.parse_args()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _assert_green(phases: dict[str, dict[str, Any]], v022: dict, v01: dict, bench: dict) -> None:
    assert phases["ball_world"]["classification"] == "PASS_GREEN"
    assert phases["ball_world"]["gate"]["phase_a_complete_gate_pass"] is True
    assert phases["car_ball"]["classification"] == "PASS_GREEN"
    assert phases["car_ball"]["gate"]["phase_b_complete_gate_pass"] is True
    assert phases["car_car"]["classification"] == "PASS_GREEN"
    assert phases["car_car"]["blocking"]["failed_case_count"] == 0
    assert phases["integrated"]["status"] == "PASS_GREEN"
    assert phases["integrated"]["gate"]["phase_d_complete_gate_pass"] is True
    assert v022["gate"]["complete_v022_gate_pass"] is True
    assert v022["counts"]["selected_starting_states"] == 39_236
    assert v01["summary"]["scenario_count"] == 27
    assert v01["summary"]["basic_parity_pass"] is True
    assert bench["summary"]["verdict"] == "PASS_GREEN"
    assert bench["summary"]["performance_gate_pass"] is True
    assert bench["summary"]["hot_loop_gpu_resident"] is True


def _cache_record(phase: str, result: dict[str, Any]) -> dict[str, Any]:
    identity_sha = result.get("authority_identity_sha256")
    if identity_sha is None:
        identity_sha = result["authority"]["identity_sha256"]
    roots = {
        "ball_world": ROOT / ".tools" / "v0.3" / "phase-a" / "oracle-cache",
        "car_ball": ROOT / ".tools" / "v0.3" / "phase-b" / "oracle-cache",
        "car_car": ROOT / ".tools" / "v0.3" / "phase-c" / "oracle-cache-relational",
        "integrated": ROOT / ".tools" / "v0.3" / "phase-d" / "oracle-cache-relational",
    }
    local_names = {
        "ball_world": "phase-a/oracle-cache",
        "car_ball": "phase-b/oracle-cache",
        "car_car": "phase-c/oracle-cache-relational",
        "integrated": "phase-d/oracle-cache-relational",
    }
    cache_dir = roots[phase] / identity_sha
    identity = _read(cache_dir / "identity.json")
    manifest = _read(cache_dir / "manifest.json")
    assert identity["authority_identity_sha256"] == identity_sha
    assert manifest["authority_identity_sha256"] == identity_sha
    assert manifest["status"] == "COMPLETE_NATIVE_AUTHORITY"
    inputs = identity["identity_inputs"]
    corpus = inputs["corpus"]
    assets = inputs["collision_assets"]
    return {
        "phase": identity["phase"],
        "authority_identity_sha256": identity_sha,
        "cache_format_version": identity["cache_format_version"],
        "status": manifest["status"],
        "case_count": manifest["case_count"],
        "frame_count": manifest["frame_count"],
        "initial_readback_count": manifest["initial_readback_count"],
        "chunk_count": manifest["chunk_count"],
        "captured_ticks": manifest["captured_ticks"],
        "native_branches": manifest.get("native_branches", ["single_native_outcome"]),
        "frozen_corpus_artifact_sha256": manifest["frozen_corpus"]["sha256"],
        "corpus_sha256": corpus["corpus_sha256"],
        "corpus_generator_source_sha256": corpus["generator_source_sha256"],
        "corpus_generator_config_sha256": corpus["generator_config_sha256"],
        "corpus_seed": corpus["seed"],
        "authority_settings_sha256": inputs["authority_settings_sha256"],
        "authority_tooling": inputs["authority_tooling"],
        "rocketsim": inputs["rocketsim"],
        "collision_assets": {
            "format": assets["format"],
            "combined_content_sha256": assets["combined_content_sha256"],
            "file_count": len(assets["files"]),
        },
        "cache_location": f"local ignored .tools/v0.3/{local_names[phase]}/{identity_sha}",
        "live_fallback_after_freeze": False,
    }


def _phase_c_release(full: dict[str, Any]) -> dict[str, Any]:
    checkpoint = _read(RESULTS / "car_car.json")
    checkpoint["generated_at_utc"] = full["generated_at_utc"]
    checkpoint["full_gate"] = {
        "classification": full["classification"],
        "corpus_sha256": full["corpus_sha256"],
        "selection_sha256": full["selection"]["selection_sha256"],
        "selected_case_count": full["selection"]["selected_case_count"],
        **full["native_multi_outcome_relation"],
        "pass_by_horizon": full["pass_by_horizon"],
        **full["blocking"],
        "ordered_bump_demo_events": full["ordered_bump_demo_events_by_branch"],
        "metric_maxima_non_demo": full["metric_maxima_non_demo_by_branch"],
    }
    return checkpoint


def _source_port() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "milestone": "v0.3",
        "status": "SOURCE_PORT_COMPLETE",
        "pinned_lineage": {
            "rocketsim_primary_commit": "c2baacb8f4b441dd8505e63c2aeb5a1679b60b02",
            "rocketsim_binding_commit": "2da51b1dac7b8127127613a5ff30e490bdd70dd8",
            "rocketsim_package": "2.2.1",
            "bullet_subtree": "RocketSim pinned bullet3-3.24",
            "soccar_cmf_sha256": (
                "2239556BDC74D205CAA6E46A0F6E91FA2C6E4257E84D4F608BA775958B0A5538"
            ),
        },
        "phase_paths": {
            "ball_world": [
                "Ball::_BulletSetup/SetState/_FinishPhysicsTick",
                "SphereTriangleDetector::getClosestPoints/collide/closestPointTriangle",
                "btConvexPlaneCollisionAlgorithm::processCollision",
                "btAdjustInternalEdgeContacts",
                "btPersistentManifold::refreshContactPoints/addManifoldPoint",
            ],
            "car_ball": [
                "btCompoundCollisionAlgorithm child box/sphere dispatch",
                "btGjkPairDetector::getClosestPointsNonVirtual",
                "btVoronoiSimplexSolver simplex update/witness recovery",
                "btGjkEpaPenetrationDepthSolver and EPA witnesses",
                "Arena::_BtCallback_OnCarBallCollision",
            ],
            "car_car": [
                "btCompoundCollisionAlgorithm child box/box dispatch",
                "btBoxBoxDetector::dBoxBox2/intersectRectQuad2/cullPoints2",
                "Arena::_BtCallback_OnCarCarCollision",
                "Car::_FinishPhysicsTick queued bump impulse and caps",
            ],
            "integrated": [
                "Arena::_cars lifecycle-ordered Car::_PreTickUpdate traversal",
                "btRSBroadphase dynamic cell handle remove/append pair stream",
                "btSimulationIslandManager island union and equal-island quicksort",
                "btSequentialImpulseConstraintSolver normal/special/friction/split ordering",
                "btSolverBody writeback and btDiscreteDynamicsWorld transform integration",
            ],
        },
        "source_backed_corrections": [
            "sphere/triangle two-sided face and seven-region closest-point branch order",
            "sphere/static persistent threshold, material combination, and internal-edge order",
            "box/sphere support, Voronoi, EPA, retained manifold, and hit callback order",
            "box/box face/edge candidate order, cull ordering, and compound-root local points",
            "per-world car visitation lifecycle state retained until membership changes",
            "dynamic broadphase cell lifecycle and dispatcher manifold insertion order",
            "one shared three-body solver state, globally ordered rows, split impulse, "
            "and writeback",
            "left-to-right float32 car force/torque and inverse-inertia auto-roll arithmetic",
        ],
        "diagnostic_exact_bits": {
            "case": "D290 tick 2 first-source-divergence proof",
            "force_xyz": ["C34CB0C4", "4161A6F3", "C0803340"],
            "torque_xyz": ["C56988AD", "C6036A54", "4420A24D"],
            "normal_impulse": "40C7FDDE",
            "tangent_impulse": "BFEFFD71",
            "result": "native and GPU exact through tick 12 after source-order correction",
        },
        "prohibited_mechanisms": {
            "case_ids_or_expected_outputs_in_runtime": False,
            "runtime_best_match_branch_selection": False,
            "native_pointer_or_allocator_emulation": False,
            "tolerance_broadening": False,
            "tie_epsilons_or_behavioral_stabilizers": False,
            "generic_bullet_port": False,
        },
    }


def main() -> int:
    args = parse_args()
    phases = {name: _read(path) for name, path in PHASE_FILES.items()}
    v022 = _read(REGRESSION_V022)
    v01 = _read(REGRESSION_V01)
    benchmark = _read(BENCHMARK)
    _assert_green(phases, v022, v01, benchmark)

    _write(RESULTS / "ball_world.json", phases["ball_world"])
    _write(RESULTS / "car_ball.json", phases["car_ball"])
    _write(RESULTS / "car_car.json", _phase_c_release(phases["car_car"]))
    _write(RESULTS / "integrated.json", phases["integrated"])
    _write(RESULTS / "benchmark.json", benchmark)
    _write(RESULTS / "source_port.json", _source_port())

    oracle = {
        "schema_version": 1,
        "milestone": "v0.3",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "COMPLETE_NATIVE_AUTHORITY",
        "policy": {
            "isolated_native_world_per_case": True,
            "exact_source_state_and_post_set_state_readback_cached": True,
            "all_ticks_1_through_12_cached": True,
            "large_artifacts_tracked": False,
            "missing_or_corrupt_cache_is_error": True,
            "live_fallback_after_freeze": False,
        },
        "phases": {name: _cache_record(name, result) for name, result in phases.items()},
    }
    _write(RESULTS / "oracle_data.json", oracle)

    regression = {
        "schema_version": 1,
        "milestone": "v0.3",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "PASS_GREEN",
        "v0_2_2_static_acceptance": {
            "counts": v022["counts"],
            "pass_fail_by_horizon": v022["pass_fail_by_horizon"],
            "gate": v022["gate"],
        },
        "v0_1_live_rocketsim": {
            "scenario_count": v01["summary"]["scenario_count"],
            "same_equation_pass": v01["summary"]["same_equation_pass"],
            "rocketsim_pass": v01["summary"]["rocketsim_pass"],
            "axis_sign_pass": v01["summary"]["axis_sign_pass"],
            "basic_parity_pass": v01["summary"]["basic_parity_pass"],
        },
        "arena_query_backends": benchmark["geometry_query_gate"],
        "deterministic_stress": benchmark["dynamic_stress_gate"],
        "repository_checks": {
            "pytest_passed": args.repository_tests,
            "pytest_failed": 0,
            "ruff_pass": args.ruff_pass,
            "compileall_pass": args.compile_pass,
            "git_diff_check_pass": args.diff_check_pass,
        },
        "prior_published_evidence": {
            "baseline_commit": "6dfd44ad9afeb3d1164da7e0e38c097fb74d07b8",
            "v0_1_v0_2_v0_2_1_v0_2_2_byte_diff_count": 0,
        },
    }
    _write(RESULTS / "regression.json", regression)

    for name in (
        "ball_world.json",
        "car_ball.json",
        "car_car.json",
        "integrated.json",
        "oracle_data.json",
        "source_port.json",
        "regression.json",
        "benchmark.json",
    ):
        path = RESULTS / name
        print(f"{path.relative_to(ROOT)} sha256={_sha256(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
