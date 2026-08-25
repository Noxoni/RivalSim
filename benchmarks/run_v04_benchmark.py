"""Benchmark the complete RivalSim v0.4 lifecycle path and both ray backends."""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import statistics
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pynvml
import warp as wp
from run_v02_benchmark import (
    STABILITY_CV_MAX,
    TICK_RATE,
    TelemetrySampler,
    environment,
    geometry_query_gate,
)

from rivalsim.arena import ArenaGeometry, WarpArenaMeshes
from rivalsim.state import StateSnapshot
from rivalsim.static_world import ActionTape, CompleteWorldSim
from rivalsim.v03_phase_d_corpus import generate_phase_d_cases, phase_d_cases_to_state

DEFAULT_BATCHES = (1024, 2048, 4096, 8192, 16384, 32768, 65536)
PERFORMANCE_FLOOR_SIM_S_PER_S = 100_000.0
REFERENCE_V03_SIM_S_PER_S = 196_614.39000488707
RESET_HEAVY_INTERVAL_TICKS = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collision-dir", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lifecycle-gate", type=Path, required=True)
    parser.add_argument("--v03-phase-a", type=Path, required=True)
    parser.add_argument("--v03-phase-b", type=Path, required=True)
    parser.add_argument("--v03-phase-c", type=Path, required=True)
    parser.add_argument("--v03-phase-d", type=Path, required=True)
    parser.add_argument("--v022-regression", type=Path, required=True)
    parser.add_argument("--v01-regression", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--ticks", type=int, default=32)
    parser.add_argument("--warmup-ticks", type=int, default=8)
    parser.add_argument("--graph-block-ticks", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--max-worlds", type=int, default=131072)
    parser.add_argument("--batches", type=int, nargs="+", default=DEFAULT_BATCHES)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repeats < 3:
        raise ValueError("at least three benchmark repeats are required")
    batches = tuple(sorted(set(int(value) for value in args.batches)))
    if not batches or batches[0] <= 0 or batches[-1] > args.max_worlds:
        raise ValueError("benchmark batches must be positive and within --max-worlds")

    wp.init()
    with contextlib.suppress(pynvml.NVMLError):
        pynvml.nvmlInit()
    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    meshes = WarpArenaMeshes(geometry, args.device)
    query_gate = geometry_query_gate(geometry, meshes, args.device)
    action_tape = ActionTape.deterministic(length=64, hold_ticks=4)
    phase_d_cases = generate_phase_d_cases()

    complete_points: list[dict[str, Any]] = []
    for worlds in batches:
        point = benchmark_point(
            worlds=worlds,
            collision_root=args.collision_dir,
            geometry=geometry,
            meshes=meshes,
            action_tape=action_tape,
            initial=_tiled_phase_d_state(phase_d_cases, worlds),
            device=args.device,
            ticks=args.ticks,
            repeats=args.repeats,
            warmup_ticks=args.warmup_ticks,
            graph_block_ticks=args.graph_block_ticks,
            seed=args.seed,
            full_reset_interval_ticks=0,
            workload="complete_mixed_lifecycle",
        )
        complete_points.append(point)
        _print_point(point)

    if (
        batches[-1] < args.max_worlds
        and len(complete_points) >= 2
        and complete_points[-1]["world_ticks_per_s_median"]
        >= complete_points[-2]["world_ticks_per_s_median"] * 1.05
    ):
        point = benchmark_point(
            worlds=args.max_worlds,
            collision_root=args.collision_dir,
            geometry=geometry,
            meshes=meshes,
            action_tape=action_tape,
            initial=_tiled_phase_d_state(phase_d_cases, args.max_worlds),
            device=args.device,
            ticks=args.ticks,
            repeats=args.repeats,
            warmup_ticks=args.warmup_ticks,
            graph_block_ticks=args.graph_block_ticks,
            seed=args.seed,
            full_reset_interval_ticks=0,
            workload="complete_mixed_lifecycle",
        )
        complete_points.append(point)
        _print_point(point)

    best = _best_stable(complete_points)
    reset_heavy = benchmark_point(
        worlds=int(best["worlds"]),
        collision_root=args.collision_dir,
        geometry=geometry,
        meshes=meshes,
        action_tape=action_tape,
        initial=None,
        device=args.device,
        ticks=args.ticks,
        repeats=args.repeats,
        warmup_ticks=args.warmup_ticks,
        graph_block_ticks=args.graph_block_ticks,
        seed=args.seed,
        full_reset_interval_ticks=RESET_HEAVY_INTERVAL_TICKS,
        workload="reset_heavy",
    )
    reset_heavy["reset_transitions_per_s_median"] = (
        reset_heavy["world_ticks_per_s_median"] / RESET_HEAVY_INTERVAL_TICKS
    )
    _print_point(reset_heavy)

    gate_inputs = {
        "lifecycle": _read_required_json(args.lifecycle_gate),
        "v03_phase_a": _read_required_json(args.v03_phase_a),
        "v03_phase_b": _read_required_json(args.v03_phase_b),
        "v03_phase_c": _read_required_json(args.v03_phase_c),
        "v03_phase_d": _read_required_json(args.v03_phase_d),
        "v022": _read_required_json(args.v022_regression),
        "v01": _read_required_json(args.v01_regression),
    }
    input_pass = {
        "lifecycle": gate_inputs["lifecycle"].get("verdict") == "PASS_GREEN",
        "v03_phase_a": _v03_phase_pass(gate_inputs["v03_phase_a"], "a"),
        "v03_phase_b": _v03_phase_pass(gate_inputs["v03_phase_b"], "b"),
        "v03_phase_c": _v03_phase_pass(gate_inputs["v03_phase_c"], "c"),
        "v03_phase_d": _v03_phase_pass(gate_inputs["v03_phase_d"], "d"),
        "v022": _v022_pass(gate_inputs["v022"]),
        "v01": _v01_pass(gate_inputs["v01"]),
    }
    ray_pass = all(
        bool(value["exact_hit_distance_normal_pass"]) for value in query_gate["backends"].values()
    )
    hot_loop_resident = all(
        int(point["hot_loop_host_to_device_bytes"]) == 0
        and int(point["hot_loop_device_to_host_bytes"]) == 0
        for point in [*complete_points, reset_heavy]
    )
    performance_pass = bool(
        best["aggregate_simulated_game_seconds_per_s_median"] >= PERFORMANCE_FLOOR_SIM_S_PER_S
        and best["stable"]
        and reset_heavy["stable"]
        and hot_loop_resident
    )
    complete_pass = bool(all(input_pass.values()) and ray_pass and performance_pass)

    output = {
        "schema_version": 1,
        "milestone": "v0.4",
        "created_utc": datetime.now(UTC).isoformat(),
        "configuration": {
            "batches": list(batches),
            "maximum_adaptive_batch": args.max_worlds,
            "ticks_requested_per_repeat": args.ticks,
            "repeats": args.repeats,
            "warmup_ticks": args.warmup_ticks,
            "cuda_graph_block_ticks": args.graph_block_ticks,
            "stability_cv_max": STABILITY_CV_MAX,
            "performance_floor_sim_s_per_s": PERFORMANCE_FLOOR_SIM_S_PER_S,
            "reset_heavy_interval_ticks": RESET_HEAVY_INTERVAL_TICKS,
            "seed": args.seed,
            "action_tape": {
                "length": action_tape.length,
                "hold_ticks": action_tape.hold_ticks,
                "device_resident": True,
                "changes_controls": True,
            },
            "world": "exactly two Octanes, one Soccar ball, static Soccar arena",
            "state_distribution": "all eight frozen v0.3 Phase D families tiled",
        },
        "environment": environment(args.device),
        "arena": {
            **geometry.metadata(),
            "source_root": args.collision_dir,
            "raw_geometry_bytes": int(geometry.vertices_uu.nbytes + geometry.triangles.nbytes),
        },
        "geometry_query_gate": query_gate,
        "complete_path_sweep": complete_points,
        "reset_heavy": reset_heavy,
        "v0_3_comparison": {
            "reference_aggregate_simulated_game_seconds_per_s": (REFERENCE_V03_SIM_S_PER_S),
            "current_aggregate_simulated_game_seconds_per_s": best[
                "aggregate_simulated_game_seconds_per_s_median"
            ],
            "retained_fraction": best["aggregate_simulated_game_seconds_per_s_median"]
            / REFERENCE_V03_SIM_S_PER_S,
            "slowdown_factor": REFERENCE_V03_SIM_S_PER_S
            / best["aggregate_simulated_game_seconds_per_s_median"],
            "scope_note": (
                "v0.4 adds resident pads, scoring, kickoff resets, demolition, "
                "respawn, clocks, raw events, and full-world resets"
            ),
        },
        "gate_inputs": {
            "lifecycle_authority_identity": gate_inputs["lifecycle"].get("authority_identity"),
            "paths": {
                "lifecycle": str(args.lifecycle_gate),
                "v03_phase_a": str(args.v03_phase_a),
                "v03_phase_b": str(args.v03_phase_b),
                "v03_phase_c": str(args.v03_phase_c),
                "v03_phase_d": str(args.v03_phase_d),
                "v022": str(args.v022_regression),
                "v01": str(args.v01_regression),
            },
            "pass": input_pass,
        },
        "summary": {
            "best_worlds": int(best["worlds"]),
            "best_world_ticks_per_s": best["world_ticks_per_s_median"],
            "best_aggregate_simulated_game_seconds_per_s": best[
                "aggregate_simulated_game_seconds_per_s_median"
            ],
            "best_coefficient_of_variation": best["coefficient_of_variation"],
            "reset_heavy_sim_s_per_s": reset_heavy["aggregate_simulated_game_seconds_per_s_median"],
            "reset_transitions_per_s": reset_heavy["reset_transitions_per_s_median"],
            "authority_and_regression_pass": all(input_pass.values()),
            "arena_query_pass": ray_pass,
            "hot_loop_gpu_resident": hot_loop_resident,
            "performance_gate_pass": performance_pass,
            "verdict": "PASS_GREEN" if complete_pass else "PAUSE_PERF",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["summary"], indent=2), flush=True)
    return 0 if complete_pass else 1


def benchmark_point(
    *,
    worlds: int,
    collision_root: str,
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    action_tape: ActionTape,
    initial: StateSnapshot | None,
    device: str,
    ticks: int,
    repeats: int,
    warmup_ticks: int,
    graph_block_ticks: int,
    seed: int,
    full_reset_interval_ticks: int,
    workload: str,
) -> dict[str, Any]:
    kwargs = {
        "device": device,
        "initial": initial,
        "geometry": geometry,
        "meshes": meshes,
        "action_tape": action_tape,
        "car_lifecycle_seed": seed,
        "kickoff_selector": np.arange(worlds, dtype=np.int32) % 5,
        "respawn_selector": np.arange(worlds, dtype=np.int32) % 4,
        "full_reset_interval_ticks": full_reset_interval_ticks,
    }
    calibration = CompleteWorldSim(worlds, collision_root, **kwargs)
    calibration.step(warmup_ticks, synchronize=True)
    calibration.capture_graph(graph_block_ticks)
    calibration.step_graph(graph_block_ticks, synchronize=True)
    started = time.perf_counter()
    calibration.step_graph(ticks, synchronize=True)
    calibration_seconds = time.perf_counter() - started
    multiplier = max(1, int(np.ceil(0.12 / calibration_seconds)))
    effective_ticks = min(4096, ticks * multiplier)
    effective_ticks = max(
        graph_block_ticks,
        (effective_ticks // graph_block_ticks) * graph_block_ticks,
    )
    del calibration
    gc.collect()

    durations: list[float] = []
    hot_h2d: list[int] = []
    hot_d2h: list[int] = []
    final_sim: CompleteWorldSim | None = None
    sampler = TelemetrySampler()
    sampler.start()
    for repeat in range(repeats):
        sim = CompleteWorldSim(worlds, collision_root, **kwargs)
        sim.step(warmup_ticks, synchronize=True)
        sim.capture_graph(graph_block_ticks)
        sim.step_graph(graph_block_ticks, synchronize=True)
        sim.reset_transfer_counters()
        started = time.perf_counter()
        sim.step_graph(effective_ticks, synchronize=True)
        durations.append(time.perf_counter() - started)
        hot_h2d.append(sim.host_to_device_bytes)
        hot_d2h.append(sim.device_to_host_bytes)
        if repeat + 1 < repeats:
            del sim
            gc.collect()
        else:
            final_sim = sim
    telemetry = sampler.stop()
    if final_sim is None:
        raise RuntimeError("benchmark did not produce a final simulator")
    state = final_sim.snapshot()
    lifecycle = final_sim.lifecycle_snapshot()
    rates = [worlds * effective_ticks / duration for duration in durations]
    median_rate = statistics.median(rates)
    mean_rate = statistics.fmean(rates)
    cv = statistics.stdev(rates) / mean_rate
    result = {
        "workload": workload,
        "worlds": worlds,
        "ticks_per_repeat": effective_ticks,
        "full_reset_interval_ticks": full_reset_interval_ticks,
        "calibration_seconds": calibration_seconds,
        "repeat_seconds": durations,
        "world_ticks_per_s_repeats": rates,
        "world_ticks_per_s_median": median_rate,
        "world_ticks_per_s_mean": mean_rate,
        "aggregate_simulated_game_seconds_per_s_median": median_rate / TICK_RATE,
        "coefficient_of_variation": cv,
        "stable": cv <= STABILITY_CV_MAX,
        "telemetry": asdict(telemetry),
        "logical_state_bytes": final_sim.logical_state_bytes,
        "logical_state_bytes_per_world": final_sim.logical_state_bytes / worlds,
        "hot_loop_host_to_device_bytes": max(hot_h2d),
        "hot_loop_device_to_host_bytes": max(hot_d2h),
        "verification_readback_outside_timing": True,
        "nan_or_error_count": int(
            np.size(state.car_pos)
            - np.count_nonzero(np.isfinite(state.car_pos))
            + np.size(state.car_vel)
            - np.count_nonzero(np.isfinite(state.car_vel))
            + np.size(state.ball_pos)
            - np.count_nonzero(np.isfinite(state.ball_pos))
            + np.size(state.ball_vel)
            - np.count_nonzero(np.isfinite(state.ball_vel))
        ),
        "world_tick_min": int(lifecycle.world_tick.min()),
        "world_tick_max": int(lifecycle.world_tick.max()),
    }
    del final_sim, state, lifecycle
    gc.collect()
    return result


def _tiled_phase_d_state(cases: tuple, worlds: int) -> StateSnapshot:
    return phase_d_cases_to_state([cases[index % len(cases)] for index in range(worlds)])


def _best_stable(points: list[dict[str, Any]]) -> dict[str, Any]:
    stable = [
        point for point in points if point["stable"] and int(point["nan_or_error_count"]) == 0
    ]
    return max(stable or points, key=lambda point: point["world_ticks_per_s_median"])


def _read_required_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required gate evidence is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _v022_pass(value: dict[str, Any]) -> bool:
    return bool(
        value.get("verdict") == "PASS_GREEN"
        or value.get("gate", {}).get("complete_v022_gate_pass", False)
        or value.get("summary", {}).get("verdict") == "PASS_GREEN"
    )


def _v03_phase_pass(value: dict[str, Any], phase: str) -> bool:
    key = f"phase_{phase}_complete_gate_pass"
    return bool(
        value.get("classification") == "PASS_GREEN"
        or value.get("summary", {}).get(key, False)
        or value.get("gate", {}).get(key, False)
    )


def _v01_pass(value: dict[str, Any]) -> bool:
    return bool(
        value.get("verdict") == "PASS_GREEN"
        or value.get("summary", {}).get("basic_parity_pass", False)
        or value.get("summary", {}).get("verdict") == "PASS_GREEN"
    )


def _print_point(point: dict[str, Any]) -> None:
    print(
        point["workload"],
        point["worlds"],
        f"{point['aggregate_simulated_game_seconds_per_s_median']:.3f} sim-s/s",
        f"CV={point['coefficient_of_variation']:.5f}",
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
