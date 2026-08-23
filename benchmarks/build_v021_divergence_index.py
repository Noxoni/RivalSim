"""Build the machine-readable v0.2.1 causal divergence index."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

REPRESENTATIVE_FINDINGS = (
    {
        "representative": "steer_medium_full_left",
        "families": ["steering", "powerslide", "longitudinal", "boost"],
        "first_causal_stage": "wheel friction preparation and two-phase application order",
        "baseline_observation": (
            "v0.2 applied wheel forces while iterating wheels from a changing rigid-body state. "
            "The native trace showed updateVehicleFirst caches every ray, suspension value and "
            "friction impulse from one pre-solve state; updateVehicleSecond applies them later."
        ),
        "correction": (
            "Cache all four wheel transforms, rays, suspension forces, bilateral side impulses "
            "and rolling impulses before applying suspension/friction in RocketSim order; mirror "
            "the handbrake and ground-boost branches."
        ),
        "rivalsim_sources": [
            "rivalsim/kernels/vehicle.py:wheel_pre_tick",
            "rivalsim/static_world.py:StaticWorldSim._step_impl",
            "rivalsim/kernels/boost_pad.py",
        ],
        "oracle_sources": [
            "src/Sim/Car/Car.cpp:_PreTickUpdate/_PostTickUpdate",
            "src/Sim/btVehicleRL/btVehicleRL.cpp:updateVehicleFirst/updateVehicleSecond",
            "BulletDynamics/ConstraintSolver/btContactConstraint.cpp:resolveSingleBilateral",
        ],
    },
    {
        "representative": "powerslide_initiation",
        "families": ["powerslide", "steering"],
        "first_causal_stage": "wheel axle/forward friction basis and cached impulse scaling",
        "baseline_observation": (
            "The diagnostic wheel rows diverged before the car trajectory: lateral and rolling "
            "impulses used approximate ordering/scaling, so the v0.2 state exceeded velocity, "
            "orientation and angular-velocity tolerances by tick 4."
        ),
        "correction": (
            "Reproduce RocketSim wheel basis construction, clipped suspension velocity, friction "
            "slip limits, extra pushback and powerslide friction scaling in source order."
        ),
        "rivalsim_sources": ["rivalsim/kernels/vehicle.py:wheel_pre_tick"],
        "oracle_sources": ["src/Sim/btVehicleRL/btVehicleRL.cpp"],
    },
    {
        "representative": "ramp_transition",
        "families": ["arena_surfaces", "landings"],
        "first_causal_stage": (
            "triangle feature selection, BVH visit order and internal-edge normal adjustment"
        ),
        "baseline_observation": (
            "The native manifold selected the RocketSim CMF-local triangle feature and adjusted "
            "its normal with Bullet adjacency data. v0.2 used the combined Warp query order and "
            "a generic SAT normal, producing a local angular-velocity failure by tick 4."
        ),
        "correction": (
            "Reproduce per-CMF Bullet quantized-BVH face rank, shared-edge flags/angles, box/"
            "triangle SAT plus GJK closest features, exact contact thresholds and callback "
            "normal order."
        ),
        "rivalsim_sources": [
            "rivalsim/arena.py:build_bullet_bvh_rank",
            "rivalsim/arena.py:build_internal_edge_data",
            "rivalsim/kernels/vehicle.py:chassis_contacts_v021",
        ],
        "oracle_sources": [
            "src/RocketSim.cpp:btGenerateInternalEdgeInfo",
            "src/Sim/Arena/Arena.cpp:btAdjustInternalEdgeContacts",
            "BulletCollision/BroadphaseCollision/btQuantizedBvh.cpp",
            "BulletCollision/NarrowPhaseCollision/btGjkPairDetector.cpp",
        ],
    },
    {
        "representative": "off_center_impact",
        "families": ["body_contacts", "settle_and_rest"],
        "first_causal_stage": "manifold constraint setup and Bullet split/velocity solve",
        "baseline_observation": (
            "v0.2's one-pass bounded contact response differed on the first impact tick. The "
            "native trace exposed box margin/inertia, contact RHS, tangent basis, accumulated "
            "impulses and split push/turn velocity as the first causal solver state."
        ),
        "correction": (
            "Use the native box half extents/margin and inverse inertia; construct every contact "
            "row from one unchanged pre-solve state; run ten split-impulse and velocity PGS "
            "iterations; then write back transform/velocity and apply deferred caps."
        ),
        "rivalsim_sources": [
            "rivalsim/kernels/vehicle.py:chassis_contacts_v021",
            "rivalsim/vehicle_state.py",
            "rivalsim/kernels/integrate.py",
        ],
        "oracle_sources": [
            "src/Sim/Arena/Arena.cpp:solver configuration",
            "BulletDynamics/ConstraintSolver/btSequentialImpulseConstraintSolver.cpp",
            "BulletCollision/CollisionDispatch/btBoxBoxDetector.cpp",
        ],
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline", type=Path, default=ROOT / "results/v0.2/parity.json"
    )
    parser.add_argument(
        "--final", type=Path, default=ROOT / "results/v0.2.1/parity.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/v0.2.1/divergence_index.json"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline = _load(args.baseline)
    final = _load(args.final)
    ranked = [_scenario_index(scenario) for scenario in baseline["scenarios"]]
    ranked.sort(key=_rank_key)
    hard_fields = sum(
        len(checkpoint["hard_mismatches"])
        for scenario in baseline["scenarios"]
        for checkpoint in scenario["horizons"]
    )
    result = {
        "schema_version": 1,
        "milestone": "v0.2.1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "input_boundary": {
            "frozen_evidence": str(args.baseline.relative_to(ROOT)).replace("\\", "/"),
            "horizons_ticks": baseline["horizons_ticks"],
            "scenario_count": baseline["scenario_count"],
            "hard_mismatch_records": baseline["summary"]["hard_mismatch_count"],
            "hard_mismatch_fields": hard_fields,
            "numeric_tolerance_failures": baseline["summary"]["numeric_failure_count"],
            "frozen_tolerances": baseline["frozen_tolerances"],
        },
        "ranking_policy": (
            "Earliest numeric or hard failure horizon, then hard-before-numeric at that horizon, "
            "then descending failure count and scenario name."
        ),
        "scenario_ranking": ranked,
        "representative_causal_findings": list(REPRESENTATIVE_FINDINGS),
        "internal_oracle": {
            "rocketsim_primary_commit": "c2baacb8f4b441dd8505e63c2aeb5a1679b60b02",
            "rocketsim_binding_commit": "2da51b1dac7b8127127613a5ff30e490bdd70dd8",
            "source_helper": "tools/rocketsim_diagnostic/trace.cpp",
            "build_recipe": "tools/rocketsim_diagnostic/CMakeLists.txt",
            "comparison_wrapper": ".tools/compare_native_v021.py (diagnostic, ignored)",
            "semantic_isolation": (
                "The helper links the unmodified pinned RocketSimPython and vendored Bullet "
                "sources into a separate executable. It reads vehicle, rigid-body and dispatcher "
                "state and may manually stage the existing pre/solve/post calls; no upstream "
                "source file is patched and no diagnostic code enters RivalSim timing."
            ),
            "trace_fields": [
                "pre/post car transform and linear/angular velocity",
                "wheel ray hit, suspension, axle/forward basis and friction impulses",
                "Bullet manifolds, triangle identity, point, normal, distance and lifetime",
                "normal/lateral/push impulses and solver prestate",
                "box-vs-triangle GJK simplex/support diagnostics",
            ],
        },
        "validation_policy_change": {
            "authority": "2026-08-23 immediate user steering adjustment",
            "hard_local_horizons_ticks": final["horizons_ticks"],
            "long_open_loop_horizons_status": "diagnostic_only_non_blocking",
            "reason": (
                "Local transition accuracy through 100 ms is the milestone requirement; "
                "multi-second synchronized open-loop identity is dominated by chaotic floating-"
                "point and contact-branch divergence and is not the transfer criterion."
            ),
        },
        "final_local_resolution": {
            "evidence": str(args.final.relative_to(ROOT)).replace("\\", "/"),
            "scenario_count": final["scenario_count"],
            "checkpoint_comparisons": final["scenario_count"]
            * len(final["horizons_ticks"]),
            "hard_mismatch_records": final["summary"]["hard_mismatch_count"],
            "numeric_tolerance_failures": final["summary"]["numeric_failure_count"],
            "parity_gate_pass": final["summary"]["parity_gate_pass"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                "baseline_records": result["input_boundary"]["hard_mismatch_records"],
                "baseline_numeric": result["input_boundary"]["numeric_tolerance_failures"],
                "final_pass": result["final_local_resolution"]["parity_gate_pass"],
            },
            indent=2,
        )
    )
    return 0


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.resolve().read_text(encoding="utf-8"))


def _scenario_index(scenario: dict[str, Any]) -> dict[str, Any]:
    hard = [record for record in scenario["horizons"] if record["hard_mismatches"]]
    numeric = [record for record in scenario["horizons"] if record["numeric_failures"]]
    first_hard = hard[0] if hard else None
    first_numeric = numeric[0] if numeric else None
    return {
        "scenario": scenario["name"],
        "family": scenario["family"],
        "first_hard_failure_horizon_ticks": (
            first_hard["horizon_ticks"] if first_hard else None
        ),
        "first_hard_mismatch_fields": (
            first_hard["hard_mismatches"] if first_hard else []
        ),
        "first_numeric_failure_horizon_ticks": (
            first_numeric["horizon_ticks"] if first_numeric else None
        ),
        "first_numeric_failure_metrics": (
            first_numeric["numeric_failures"] if first_numeric else []
        ),
        "hard_failure_records": len(hard),
        "hard_mismatch_fields": sum(len(record["hard_mismatches"]) for record in hard),
        "numeric_tolerance_failures": sum(
            len(record["numeric_failures"]) for record in numeric
        ),
    }


def _rank_key(item: dict[str, Any]) -> tuple[int, int, int, str]:
    hard = item["first_hard_failure_horizon_ticks"]
    numeric = item["first_numeric_failure_horizon_ticks"]
    earliest = min(value for value in (hard, numeric) if value is not None)
    hard_first = 0 if hard is not None and hard == earliest else 1
    failures = item["hard_mismatch_fields"] + item["numeric_tolerance_failures"]
    return earliest, hard_first, -failures, item["scenario"]


if __name__ == "__main__":
    raise SystemExit(main())
