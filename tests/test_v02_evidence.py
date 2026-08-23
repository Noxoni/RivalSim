from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "v0.2"


def _load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    committed_bytes = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(committed_bytes).hexdigest().upper()


def test_published_v02_benchmark_is_complete_stable_and_honestly_red() -> None:
    payload = _load("benchmark.json")

    assert payload["milestone"] == "v0.2"
    assert payload["configuration"]["mandatory_batches"] == [
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        65536,
        131072,
    ]
    assert payload["configuration"]["repeats"] == 5
    assert set(payload["variants"]) == {"B0", "B1_default", "B1_cubql", "B2", "B3"}

    points = [point for variant in payload["variants"].values() for point in variant]
    assert len(points) == 44
    assert max(point["coefficient_of_variation"] for point in points) <= 0.05
    assert all(point["hot_loop_host_to_device_bytes"] == 0 for point in points)
    assert all(point["hot_loop_device_to_host_bytes"] == 0 for point in points)
    assert all(point["nan_or_error_count"] == 0 for point in points)

    query = payload["geometry_query_gate"]
    assert query["ray_count"] == 4608
    assert query["selected_ray_backend"] == "cubql"
    assert all(
        result["exact_hit_distance_normal_pass"] for result in query["backends"].values()
    )

    stress = payload["stress_gate"]
    assert stress["measured_ticks"] == 2400
    assert stress["finite"] and stress["deterministic_equal"]
    assert stress["bounded_velocity"] and stress["bounded_penetration"]
    assert len(set(stress["state_sha256_runs"])) == 1

    summary = payload["summary"]
    assert summary["performance_band"] == "green_threshold"
    assert summary["best_b3_worlds"] == 262144
    assert summary["best_b3_aggregate_simulated_game_seconds_per_s"] >= 100_000
    assert not summary["parity_gate_pass"]
    assert summary["verdict"] == "PAUSE_RED"


def test_published_v02_parity_uses_frozen_limits_and_preserves_failures() -> None:
    payload = _load("parity.json")

    assert payload["milestone"] == "v0.2"
    assert payload["mode"] == "gate"
    assert payload["scenario_count"] == 35
    assert payload["horizons_ticks"] == [1, 4, 8, 30, 60, 120, 300, 600]
    assert len(payload["families"]) == 8
    assert payload["frozen_tolerances"] == {
        "position_uu": 10.0,
        "linear_velocity_uu_per_s": 25.0,
        "orientation_rad": 0.025,
        "angular_velocity_rad_per_s": 0.1,
        "boost": 0.01,
        "handbrake_value": 0.0001,
        "world_contact_normal_rad": 0.05,
    }

    records = [horizon for scenario in payload["scenarios"] for horizon in scenario["horizons"]]
    assert len(records) == 35 * 8
    assert sum(bool(record["hard_mismatches"]) for record in records) == 85
    assert sum(len(record["numeric_failures"]) for record in records) == 617
    assert payload["summary"]["hard_mismatch_count"] == 85
    assert payload["summary"]["numeric_failure_count"] == 617
    assert not payload["summary"]["parity_gate_pass"]


def test_v02_manifest_binds_evidence_assets_and_scope() -> None:
    manifest = _load("manifest.json")

    assert manifest["milestone"] == "v0.2"
    assert manifest["verdict"] == "PAUSE_RED"
    assert manifest["repository"]["implementation_commit"] == (
        "f2363104a56a358276682e16110d16f37e8d0539"
    )
    evidence = {entry["path"]: entry for entry in manifest["tracked_evidence"]}
    for name in ("benchmark.json", "parity.json"):
        relative = f"results/v0.2/{name}"
        path = RESULTS / name
        assert evidence[relative]["size_bytes"] == len(path.read_bytes().replace(b"\r\n", b"\n"))
        assert evidence[relative]["sha256"] == _sha256(path)

    assets = manifest["collision_assets"]
    assert assets["file_count"] == 16
    assert assets["vertices"] == 4468
    assert assets["triangles"] == 8020
    assert assets["combined_content_sha256"] == (
        "2239556BDC74D205CAA6E46A0F6E91FA2C6E4257E84D4F608BA775958B0A5538"
    )
    assert manifest["asset_audit"]["tracked_extracted_asset_count"] == 0
    assert manifest["frozen_v0_1_evidence"]["unchanged_at_implementation_commit"]
    assert manifest["scope_boundary"]["v0_2_complete"]
    assert not manifest["scope_boundary"]["v0_3_begun"]
