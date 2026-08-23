from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

RESULTS = Path(__file__).parents[1] / "results" / "v0.1"


def test_published_parity_gate_is_complete_and_passing() -> None:
    parity = json.loads((RESULTS / "parity.json").read_text(encoding="utf-8"))
    assert parity["milestone"] == "v0.1"
    assert parity["mode"] == "gate_evaluation"
    assert parity["horizons_ticks"] == [1, 4, 8, 30, 60, 120]
    assert parity["summary"]["scenario_count"] == 27
    assert parity["summary"]["same_equation_pass"] is True
    assert parity["summary"]["rocketsim_pass"] is True
    assert parity["summary"]["axis_sign_pass"] is True
    assert parity["summary"]["basic_parity_pass"] is True


def test_published_performance_gate_is_complete_and_passing() -> None:
    benchmark = json.loads((RESULTS / "benchmark.json").read_text(encoding="utf-8"))
    assert benchmark["milestone"] == "v0.1"
    mandatory = benchmark["configuration"]["mandatory_batches"]
    stability_cv_max = benchmark["configuration"]["stability_cv_max"]
    assert mandatory == [256, 512, 1024, 2048, 4096, 8192, 16384]
    assert stability_cv_max == 0.05
    assert {point["worlds"] for point in benchmark["gpu"]}.issuperset(mandatory)
    assert {point["worlds"] for point in benchmark["cpu_same_equation"]} == set(mandatory)
    assert all(not point["nan_or_error"] for point in benchmark["gpu"])
    assert all(not point["nan_or_error"] for point in benchmark["cpu_same_equation"])
    assert all(
        point["coefficient_of_variation"] <= stability_cv_max for point in benchmark["gpu"]
    )
    assert all(
        point["coefficient_of_variation"] <= stability_cv_max
        for point in benchmark["cpu_same_equation"]
    )
    scaling_points = [point for point in benchmark["gpu"] if point["worlds"] in mandatory[3:]]
    assert all(
        current["world_ticks_per_s_median"] > previous["world_ticks_per_s_median"]
        for previous, current in pairwise(scaling_points)
    )
    assert all(
        point["hot_loop_host_to_device_bytes"] == 0
        and point["hot_loop_device_to_host_bytes"] == 0
        for point in benchmark["gpu"]
    )
    assert all(benchmark["summary"]["performance_conditions"].values())
    assert benchmark["summary"]["performance_gate_pass"] is True
