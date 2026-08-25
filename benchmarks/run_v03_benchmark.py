"""Benchmark and stress the complete RivalSim v0.3 dynamic-contact path."""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import statistics
import time
from collections.abc import Callable
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
from rivalsim.static_world import (
    ActionTape,
    CarCarWorldSim,
    DynamicWorldSim,
    IntegratedWorldSim,
    StaticWorldSim,
)
from rivalsim.v03_phase_d_corpus import (
    PHASE_D_FAMILIES,
    generate_phase_d_cases,
    phase_d_cases_to_state,
)

DEFAULT_BATCHES = (1024, 2048, 4096, 8192, 16384, 32768, 65536)
PERFORMANCE_FLOOR_SIM_S_PER_S = 100_000.0
REFERENCE_V022_SIM_S_PER_S = 511_886.15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collision-dir", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase-d-parity", type=Path, required=True)
    parser.add_argument("--v022-regression", type=Path, required=True)
    parser.add_argument("--v01-regression", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--ticks", type=int, default=32)
    parser.add_argument("--warmup-ticks", type=int, default=8)
    parser.add_argument("--graph-block-ticks", type=int, default=8)
    parser.add_argument("--stress-worlds", type=int, default=64)
    parser.add_argument("--stress-ticks", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--max-worlds", type=int, default=131072)
    parser.add_argument("--batches", type=int, nargs="+", default=DEFAULT_BATCHES)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repeats < 3:
        raise ValueError("at least three benchmark repeats are required")
    if args.graph_block_ticks <= 0 or args.ticks <= 0:
        raise ValueError("tick and CUDA graph sizes must be positive")
    batches = tuple(sorted(set(int(value) for value in args.batches)))
    if not batches or batches[0] <= 0 or batches[-1] > args.max_worlds:
        raise ValueError("benchmark batches must be positive and no larger than --max-worlds")

    wp.init()
    with contextlib.suppress(pynvml.NVMLError):
        pynvml.nvmlInit()
    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    meshes = WarpArenaMeshes(geometry, args.device)
    query_gate = geometry_query_gate(geometry, meshes, args.device)
    action_tape = ActionTape.deterministic(length=64, hold_ticks=4)
    phase_d_cases = generate_phase_d_cases()

    stress = run_dynamic_stress_gate(
        collision_root=args.collision_dir,
        geometry=geometry,
        meshes=meshes,
        action_tape=action_tape,
        phase_d_cases=phase_d_cases,
        worlds=args.stress_worlds,
        ticks=args.stress_ticks,
        graph_block_ticks=args.graph_block_ticks,
        device=args.device,
        seed=args.seed,
    )
    if not stress["stress_gate_pass"]:
        raise RuntimeError(
            "dynamic stress failed before performance timing: "
            + json.dumps(stress, sort_keys=True)
        )

    integrated_points: list[dict[str, Any]] = []
    for worlds in batches:
        point = benchmark_point(
            component="integrated",
            worlds=worlds,
            collision_root=args.collision_dir,
            geometry=geometry,
            meshes=meshes,
            action_tape=action_tape,
            phase_d_cases=phase_d_cases,
            device=args.device,
            ticks=args.ticks,
            repeats=args.repeats,
            warmup_ticks=args.warmup_ticks,
            graph_block_ticks=args.graph_block_ticks,
            seed=args.seed,
        )
        integrated_points.append(point)
        _print_point(point)

    if (
        batches[-1] < args.max_worlds
        and integrated_points[-1]["world_ticks_per_s_median"]
        >= integrated_points[-2]["world_ticks_per_s_median"] * 1.05
    ):
        point = benchmark_point(
            component="integrated",
            worlds=args.max_worlds,
            collision_root=args.collision_dir,
            geometry=geometry,
            meshes=meshes,
            action_tape=action_tape,
            phase_d_cases=phase_d_cases,
            device=args.device,
            ticks=args.ticks,
            repeats=args.repeats,
            warmup_ticks=args.warmup_ticks,
            graph_block_ticks=args.graph_block_ticks,
            seed=args.seed,
        )
        integrated_points.append(point)
        _print_point(point)

    best = _best_stable(integrated_points)
    component_points: dict[str, dict[str, Any]] = {"integrated": best}
    for component in ("static", "ball_world", "car_ball", "car_car"):
        point = benchmark_point(
            component=component,
            worlds=int(best["worlds"]),
            collision_root=args.collision_dir,
            geometry=geometry,
            meshes=meshes,
            action_tape=action_tape,
            phase_d_cases=phase_d_cases,
            device=args.device,
            ticks=args.ticks,
            repeats=args.repeats,
            warmup_ticks=args.warmup_ticks,
            graph_block_ticks=args.graph_block_ticks,
            seed=args.seed,
        )
        component_points[component] = point
        _print_point(point)

    parity = _read_required_json(args.phase_d_parity)
    v022 = _read_required_json(args.v022_regression)
    v01 = _read_required_json(args.v01_regression)
    fidelity_pass = bool(
        parity.get("gate", {}).get("phase_d_complete_gate_pass", False)
        and v022.get("gate", {}).get("complete_v022_gate_pass", False)
        and v01.get("summary", {}).get("basic_parity_pass", False)
    )
    ray_pass = all(
        bool(value["exact_hit_distance_normal_pass"])
        for value in query_gate["backends"].values()
    )
    hot_loop_resident = all(
        int(point["hot_loop_host_to_device_bytes"]) == 0
        and int(point["hot_loop_device_to_host_bytes"]) == 0
        for point in [*integrated_points, *component_points.values()]
    )
    performance_pass = bool(
        best["aggregate_simulated_game_seconds_per_s_median"]
        >= PERFORMANCE_FLOOR_SIM_S_PER_S
        and best["stable"]
        and hot_loop_resident
    )
    complete_pass = bool(
        fidelity_pass
        and ray_pass
        and stress["stress_gate_pass"]
        and performance_pass
    )

    output = {
        "schema_version": 1,
        "milestone": "v0.3",
        "created_utc": datetime.now(UTC).isoformat(),
        "configuration": {
            "batches": [int(value) for value in batches],
            "maximum_adaptive_batch": args.max_worlds,
            "ticks_requested_per_repeat": args.ticks,
            "repeats": args.repeats,
            "warmup_ticks": args.warmup_ticks,
            "cuda_graph_block_ticks": args.graph_block_ticks,
            "stability_cv_max": STABILITY_CV_MAX,
            "performance_floor_sim_s_per_s": PERFORMANCE_FLOOR_SIM_S_PER_S,
            "seed": args.seed,
            "action_tape": {
                "length": action_tape.length,
                "hold_ticks": action_tape.hold_ticks,
                "device_resident": True,
                "changes_controls": True,
            },
            "world": "exactly two Octanes, one Soccar ball, static Soccar arena",
            "state_distribution": "all eight frozen Phase D families tiled across worlds",
        },
        "environment": environment(args.device),
        "arena": {
            **geometry.metadata(),
            "source_root": args.collision_dir,
            "raw_geometry_bytes": int(
                geometry.vertices_uu.nbytes + geometry.triangles.nbytes
            ),
        },
        "geometry_query_gate": query_gate,
        "dynamic_stress_gate": stress,
        "integrated_sweep": integrated_points,
        "component_points_at_selected_batch": component_points,
        "component_cost": _component_costs(component_points),
        "v0_2_2_comparison": {
            "reference_aggregate_simulated_game_seconds_per_s": (
                REFERENCE_V022_SIM_S_PER_S
            ),
            "current_aggregate_simulated_game_seconds_per_s": best[
                "aggregate_simulated_game_seconds_per_s_median"
            ],
            "retained_fraction": best[
                "aggregate_simulated_game_seconds_per_s_median"
            ]
            / REFERENCE_V022_SIM_S_PER_S,
            "slowdown_factor": REFERENCE_V022_SIM_S_PER_S
            / best["aggregate_simulated_game_seconds_per_s_median"],
            "scope_note": (
                "v0.2.2 B3 is static-world only; v0.3 includes ball-world, "
                "both car-ball pairs, car-car, and the shared island"
            ),
        },
        "gate_inputs": {
            "phase_d_parity": str(args.phase_d_parity),
            "phase_d_pass": bool(
                parity.get("gate", {}).get("phase_d_complete_gate_pass", False)
            ),
            "v0_2_2_regression": str(args.v022_regression),
            "v0_2_2_pass": bool(
                v022.get("gate", {}).get("complete_v022_gate_pass", False)
            ),
            "v0_1_regression": str(args.v01_regression),
            "v0_1_pass": bool(
                v01.get("summary", {}).get("basic_parity_pass", False)
            ),
        },
        "summary": {
            "best_worlds": int(best["worlds"]),
            "best_world_ticks_per_s": best["world_ticks_per_s_median"],
            "best_aggregate_simulated_game_seconds_per_s": best[
                "aggregate_simulated_game_seconds_per_s_median"
            ],
            "best_coefficient_of_variation": best["coefficient_of_variation"],
            "fidelity_and_regression_pass": fidelity_pass,
            "arena_query_pass": ray_pass,
            "deterministic_stress_pass": stress["stress_gate_pass"],
            "hot_loop_gpu_resident": hot_loop_resident,
            "performance_gate_pass": performance_pass,
            "verdict": "PASS_GREEN" if complete_pass else "PAUSE_PERF",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["summary"], indent=2), flush=True)
    return 0 if complete_pass else 1


def run_dynamic_stress_gate(
    *,
    collision_root: str,
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    action_tape: ActionTape,
    phase_d_cases: tuple,
    worlds: int,
    ticks: int,
    graph_block_ticks: int,
    device: str,
    seed: int,
) -> dict[str, Any]:
    if worlds < len(PHASE_D_FAMILIES):
        raise ValueError("dynamic stress must contain every Phase D family")
    initial = _tiled_phase_d_state(phase_d_cases, worlds, stress_stratified=True)
    runs: list[dict[str, Any]] = []
    digests: list[str] = []
    for _run_index in range(2):
        sim = IntegratedWorldSim(
            worlds,
            collision_root,
            variant="B3",
            device=device,
            initial=initial,
            geometry=geometry,
            meshes=meshes,
            action_tape=action_tape,
            car_lifecycle_seed=seed,
        )
        sim.capture_graph(graph_block_ticks)
        sim.reset_transfer_counters()
        sim.step_graph(ticks, synchronize=True)
        hot_h2d = sim.host_to_device_bytes
        hot_d2h = sim.device_to_host_bytes
        digest, hashed_bytes, array_count = _full_state_digest(sim)
        state = sim.snapshot()
        vehicle = sim.vehicle_snapshot()
        penetration = _maximum_penetration_uu(sim, vehicle)
        run = {
            "full_state_sha256": digest,
            "hashed_state_bytes": hashed_bytes,
            "hashed_array_count": array_count,
            "finite": bool(
                np.isfinite(state.car_pos).all()
                and np.isfinite(state.car_vel).all()
                and np.isfinite(state.car_quat).all()
                and np.isfinite(state.car_ang_vel).all()
                and np.isfinite(state.ball_pos).all()
                and np.isfinite(state.ball_vel).all()
                and np.isfinite(state.ball_quat).all()
                and np.isfinite(state.ball_ang_vel).all()
            ),
            "max_car_linear_speed_uu_per_s": float(
                np.max(np.linalg.norm(state.car_vel, axis=-1))
            ),
            "max_car_angular_speed_rad_per_s": float(
                np.max(np.linalg.norm(state.car_ang_vel, axis=-1))
            ),
            "max_ball_linear_speed_uu_per_s": float(
                np.max(np.linalg.norm(state.ball_vel, axis=-1))
            ),
            "max_ball_angular_speed_rad_per_s": float(
                np.max(np.linalg.norm(state.ball_ang_vel, axis=-1))
            ),
            "max_final_penetration_uu": penetration,
            "hot_loop_host_to_device_bytes": hot_h2d,
            "hot_loop_device_to_host_bytes": hot_d2h,
        }
        runs.append(run)
        digests.append(digest)
        del sim, state, vehicle
        gc.collect()

    result = {
        "worlds": worlds,
        "ticks": ticks,
        "seed": seed,
        "independent_runs": 2,
        "initial_family_counts": _stress_family_counts(worlds),
        "coverage_intent": [
            "ball bounces and wall/corner contacts",
            "car-ball impacts",
            "car-car impacts and bump/demo neighborhoods",
            "wheel/car/ball interactions",
            "mixed static and dynamic manifolds",
        ],
        "runs": runs,
        "full_state_sha256_runs": digests,
        "deterministic_equal": digests[0] == digests[1],
        "finite": all(bool(run["finite"]) for run in runs),
        "bounded_authoritative_speeds": all(
            float(run["max_car_linear_speed_uu_per_s"]) <= 2300.001
            and float(run["max_car_angular_speed_rad_per_s"]) <= 5.5001
            and float(run["max_ball_linear_speed_uu_per_s"]) <= 6000.001
            and float(run["max_ball_angular_speed_rad_per_s"]) <= 6.0001
            for run in runs
        ),
        "bounded_final_penetration": all(
            float(run["max_final_penetration_uu"]) < 200.0 for run in runs
        ),
        "hot_loop_gpu_resident": all(
            int(run["hot_loop_host_to_device_bytes"]) == 0
            and int(run["hot_loop_device_to_host_bytes"]) == 0
            for run in runs
        ),
    }
    result["stress_gate_pass"] = bool(
        result["deterministic_equal"]
        and result["finite"]
        and result["bounded_authoritative_speeds"]
        and result["bounded_final_penetration"]
        and result["hot_loop_gpu_resident"]
    )
    return result


def benchmark_point(
    *,
    component: str,
    worlds: int,
    collision_root: str,
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    action_tape: ActionTape,
    phase_d_cases: tuple,
    device: str,
    ticks: int,
    repeats: int,
    warmup_ticks: int,
    graph_block_ticks: int,
    seed: int,
) -> dict[str, Any]:
    initial = _tiled_phase_d_state(phase_d_cases, worlds)
    factory = _sim_factory(component)

    calibration = factory(
        worlds,
        collision_root,
        device=device,
        initial=initial,
        geometry=geometry,
        meshes=meshes,
        action_tape=action_tape,
        car_lifecycle_seed=seed,
    )
    calibration.step(warmup_ticks, synchronize=True)
    calibration.capture_graph(graph_block_ticks)
    calibration.step_graph(graph_block_ticks, synchronize=True)
    started = time.perf_counter()
    calibration.step_graph(ticks, synchronize=True)
    calibration_seconds = time.perf_counter() - started
    target_repeat_seconds = 0.12
    multiplier = max(1, int(np.ceil(target_repeat_seconds / calibration_seconds)))
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
    final_sim: StaticWorldSim | None = None
    sampler = TelemetrySampler()
    sampler.start()
    for repeat in range(repeats):
        sim = factory(
            worlds,
            collision_root,
            device=device,
            initial=initial,
            geometry=geometry,
            meshes=meshes,
            action_tape=action_tape,
            car_lifecycle_seed=seed,
        )
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
    rates = [worlds * effective_ticks / duration for duration in durations]
    median_rate = statistics.median(rates)
    mean_rate = statistics.fmean(rates)
    cv = statistics.stdev(rates) / mean_rate
    result = {
        "component": component,
        "worlds": worlds,
        "ticks_per_repeat": effective_ticks,
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
    }
    del final_sim, state, initial
    gc.collect()
    return result


def _sim_factory(component: str) -> Callable[..., StaticWorldSim]:
    if component == "static":

        def static_factory(*args, car_lifecycle_seed: int, **kwargs):
            del car_lifecycle_seed
            return StaticWorldSim(*args, variant="B3", **kwargs)

        return static_factory
    if component == "ball_world":

        def ball_factory(*args, car_lifecycle_seed: int, **kwargs):
            del car_lifecycle_seed
            return DynamicWorldSim(
                *args,
                variant="B3",
                enable_car_ball_contacts=False,
                **kwargs,
            )

        return ball_factory
    if component == "car_ball":

        def car_ball_factory(*args, car_lifecycle_seed: int, **kwargs):
            del car_lifecycle_seed
            return DynamicWorldSim(*args, variant="B3", **kwargs)

        return car_ball_factory
    if component == "car_car":

        def car_car_factory(*args, car_lifecycle_seed: int, **kwargs):
            return CarCarWorldSim(
                *args,
                variant="B3",
                car_lifecycle_seed=car_lifecycle_seed,
                **kwargs,
            )

        return car_car_factory
    if component == "integrated":

        def integrated_factory(*args, car_lifecycle_seed: int, **kwargs):
            return IntegratedWorldSim(
                *args,
                variant="B3",
                car_lifecycle_seed=car_lifecycle_seed,
                **kwargs,
            )

        return integrated_factory
    raise ValueError(f"unknown v0.3 benchmark component: {component}")


def _tiled_phase_d_state(
    cases: tuple,
    worlds: int,
    *,
    stress_stratified: bool = False,
) -> StateSnapshot:
    if stress_stratified:
        per_family = len(cases) // len(PHASE_D_FAMILIES)
        indices = []
        for ordinal in range(worlds):
            family = ordinal % len(PHASE_D_FAMILIES)
            within = (ordinal // len(PHASE_D_FAMILIES)) % per_family
            indices.append(family * per_family + within)
        selected = [cases[index] for index in indices]
    else:
        selected = [cases[index % len(cases)] for index in range(worlds)]
    return phase_d_cases_to_state(selected)


def _stress_family_counts(worlds: int) -> dict[str, int]:
    return {
        family: sum(
            1
            for ordinal in range(worlds)
            if ordinal % len(PHASE_D_FAMILIES) == family_index
        )
        for family_index, family in enumerate(PHASE_D_FAMILIES)
    }


def _full_state_digest(sim: IntegratedWorldSim) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    hashed_bytes = 0
    array_count = 0
    owners = (
        ("sim", sim),
        ("state", sim.state),
        ("controls", sim.controls),
        ("vehicle", sim.vehicle),
        ("ball_world", sim.ball_world),
        ("car_ball_a", sim.car_ball),
        ("car_ball_b", sim.car_ball_b),
        ("car_car", sim.car_car),
    )
    seen: set[int] = set()
    for owner_name, owner in owners:
        for name, value in sorted(vars(owner).items()):
            if not isinstance(value, (wp.array, np.ndarray)) or id(value) in seen:
                continue
            seen.add(id(value))
            array = value.numpy() if isinstance(value, wp.array) else value
            contiguous = np.ascontiguousarray(array)
            label = f"{owner_name}.{name}:{contiguous.dtype}:{contiguous.shape}".encode()
            digest.update(label)
            digest.update(contiguous.tobytes())
            hashed_bytes += contiguous.nbytes
            array_count += 1
    return digest.hexdigest().upper(), hashed_bytes, array_count


def _maximum_penetration_uu(sim: IntegratedWorldSim, vehicle: Any) -> float:
    values = [float(np.max(vehicle.penetration_max, initial=0.0))]
    for owner, field in (
        (sim.ball_world, "contact_distance_bt"),
        (sim.car_ball, "manifold_distance_bt"),
        (sim.car_ball_b, "manifold_distance_bt"),
        (sim.car_car, "manifold_distance_bt"),
    ):
        distance = getattr(owner, field).numpy()
        values.append(float(max(0.0, -np.min(distance, initial=0.0) * 50.0)))
    return max(values)


def _component_costs(points: dict[str, dict[str, Any]]) -> dict[str, Any]:
    seconds_per_world_tick = {
        name: 1.0 / float(point["world_ticks_per_s_median"])
        for name, point in points.items()
    }
    static = seconds_per_world_tick["static"]
    ball_world = seconds_per_world_tick["ball_world"]
    car_ball = seconds_per_world_tick["car_ball"]
    car_car = seconds_per_world_tick["car_car"]
    integrated = seconds_per_world_tick["integrated"]
    return {
        "seconds_per_world_tick": seconds_per_world_tick,
        "incremental_microseconds_per_world_tick": {
            "ball_world_over_static": (ball_world - static) * 1.0e6,
            "car_ball_over_ball_world": (car_ball - ball_world) * 1.0e6,
            "car_car_over_static": (car_car - static) * 1.0e6,
            "integrated_shared_ordering_over_pairwise_sum_estimate": (
                integrated - (car_ball + car_car - static)
            )
            * 1.0e6,
        },
        "complete_slowdown_vs_static": integrated / static,
        "note": (
            "incremental timings use equal worlds, action tape, state distribution, "
            "warmup, and CUDA graph policy"
        ),
    }


def _best_stable(points: list[dict[str, Any]]) -> dict[str, Any]:
    stable = [
        point
        for point in points
        if point["stable"] and int(point["nan_or_error_count"]) == 0
    ]
    return max(stable or points, key=lambda point: point["world_ticks_per_s_median"])


def _read_required_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required gate evidence is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _print_point(point: dict[str, Any]) -> None:
    print(
        point["component"],
        point["worlds"],
        f"{point['aggregate_simulated_game_seconds_per_s_median']:.3f} sim-s/s",
        f"CV={point['coefficient_of_variation']:.5f}",
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
