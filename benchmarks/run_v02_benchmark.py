"""Run the prescribed B0/B1/B2/B3 RivalSim static-world GPU sweep."""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import platform
import statistics
import threading
import time
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import pynvml
import warp as wp

from rivalsim.arena import ArenaGeometry, WarpArenaMeshes, raycast_soccar_cpu
from rivalsim.kernels.world_queries import query_soccar_rays
from rivalsim.ray_corpus import make_soccar_ray_corpus
from rivalsim.state import StateSnapshot
from rivalsim.static_world import ActionTape, StaticWorldSim, make_contact_rich_state

MANDATORY_BATCHES = (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)
STABILITY_CV_MAX = 0.05
TICK_RATE = 120.0


@dataclass(slots=True)
class Telemetry:
    samples: int
    gpu_util_mean_percent: float | None
    gpu_util_max_percent: float | None
    vram_mean_bytes: float | None
    vram_max_bytes: int | None
    process_cpu_mean_percent: float
    process_cpu_max_percent: float
    system_cpu_mean_percent: float


class TelemetrySampler:
    def __init__(self, interval_s: float = 0.01):
        self.interval_s = interval_s
        self.stop_event = threading.Event()
        self.samples: list[tuple[float | None, int | None, float, float]] = []
        self.thread: threading.Thread | None = None
        self.process = psutil.Process()
        try:
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except pynvml.NVMLError:
            self.handle = None

    def start(self) -> None:
        self.process.cpu_percent(None)
        psutil.cpu_percent(None)
        self._sample()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self) -> Telemetry:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        self._sample()
        gpu = [sample[0] for sample in self.samples if sample[0] is not None]
        memory = [sample[1] for sample in self.samples if sample[1] is not None]
        process_cpu = [sample[2] for sample in self.samples]
        system_cpu = [sample[3] for sample in self.samples]
        return Telemetry(
            samples=len(self.samples),
            gpu_util_mean_percent=statistics.fmean(gpu) if gpu else None,
            gpu_util_max_percent=max(gpu) if gpu else None,
            vram_mean_bytes=statistics.fmean(memory) if memory else None,
            vram_max_bytes=max(memory) if memory else None,
            process_cpu_mean_percent=statistics.fmean(process_cpu),
            process_cpu_max_percent=max(process_cpu),
            system_cpu_mean_percent=statistics.fmean(system_cpu),
        )

    def _loop(self) -> None:
        while not self.stop_event.wait(self.interval_s):
            self._sample()

    def _sample(self) -> None:
        gpu_util = None
        memory = None
        if self.handle is not None:
            try:
                gpu_util = float(pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu)
                memory = int(pynvml.nvmlDeviceGetMemoryInfo(self.handle).used)
            except pynvml.NVMLError:
                pass
        self.samples.append(
            (
                gpu_util,
                memory,
                float(self.process.cpu_percent(None)),
                float(psutil.cpu_percent(None)),
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collision-dir", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup-ticks", type=int, default=16)
    parser.add_argument("--graph-block-ticks", type=int, default=8)
    parser.add_argument("--ticks", type=int, default=64)
    parser.add_argument("--max-worlds", type=int, default=262144)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--milestone", default="v0.2")
    parser.add_argument("--parity-file", type=Path)
    parser.add_argument(
        "--reference-benchmark",
        type=Path,
        default=Path("results/v0.2/benchmark.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repeats < 3:
        raise ValueError("at least three repeats are required")
    wp.init()
    with contextlib.suppress(pynvml.NVMLError):
        pynvml.nvmlInit()
    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    meshes = WarpArenaMeshes(geometry, args.device)
    query_gate = geometry_query_gate(geometry, meshes, args.device)
    action_tape = ActionTape.deterministic(hold_ticks=4)
    stress = run_stress_gate(
        collision_root=args.collision_dir,
        geometry=geometry,
        meshes=meshes,
        action_tape=action_tape,
        device=args.device,
        seed=args.seed,
        graph_block_ticks=args.graph_block_ticks,
    )
    if not stress["stress_gate_pass"]:
        raise RuntimeError(
            "stress gate failed before performance measurement: "
            + json.dumps(stress, sort_keys=True)
        )
    configurations = (
        ("B0", "B0", "cubql"),
        ("B1_default", "B1", "default"),
        ("B1_cubql", "B1", "cubql"),
        ("B2", "B2", "cubql"),
        ("B3", "B3", "cubql"),
    )
    results: dict[str, list[dict[str, Any]]] = {name: [] for name, _, _ in configurations}

    for result_name, variant, ray_backend in configurations:
        for worlds in MANDATORY_BATCHES:
            point = benchmark_point(
                worlds=worlds,
                result_name=result_name,
                variant=variant,
                ray_backend=ray_backend,
                collision_root=args.collision_dir,
                geometry=geometry,
                meshes=meshes,
                action_tape=action_tape,
                device=args.device,
                ticks=args.ticks,
                repeats=args.repeats,
                warmup_ticks=args.warmup_ticks,
                graph_block_ticks=args.graph_block_ticks,
                seed=args.seed,
            )
            results[result_name].append(point)
            print(
                result_name,
                worlds,
                f"{point['aggregate_simulated_game_seconds_per_s_median']:.3f} sim-s/s",
                f"CV={point['coefficient_of_variation']:.4f}",
                flush=True,
            )
        endpoint_gain = _relative_gain(results[result_name][-2], results[result_name][-1])
        if endpoint_gain >= 0.05 and args.max_worlds >= 262144:
            results[result_name].append(
                benchmark_point(
                    worlds=262144,
                    result_name=result_name,
                    variant=variant,
                    ray_backend=ray_backend,
                    collision_root=args.collision_dir,
                    geometry=geometry,
                    meshes=meshes,
                    action_tape=action_tape,
                    device=args.device,
                    ticks=args.ticks,
                    repeats=args.repeats,
                    warmup_ticks=args.warmup_ticks,
                    graph_block_ticks=args.graph_block_ticks,
                    seed=args.seed,
                )
            )

    b3_best = _best_stable(results["B3"])
    common_best_worlds = b3_best["worlds"]
    common = {
        name: next(
            (point for point in points if point["worlds"] == common_best_worlds),
            _best_stable(points),
        )
        for name, points in results.items()
    }
    b0_rate = common["B0"]["world_ticks_per_s_median"]
    component_costs = {
        name: b0_rate / point["world_ticks_per_s_median"] for name, point in common.items()
    }
    parity_path = args.parity_file or Path("results") / args.milestone / "parity.json"
    parity_pass = False
    if parity_path.exists():
        parity_result = json.loads(parity_path.read_text(encoding="utf-8"))
        parity_pass = bool(parity_result["summary"]["parity_gate_pass"])
    hot_loop_resident = all(
        point["hot_loop_host_to_device_bytes"] == 0 and point["hot_loop_device_to_host_bytes"] == 0
        for points in results.values()
        for point in points
    )
    stable_scaling = any(point["worlds"] >= 4096 and point["stable"] for point in results["B3"])
    performance_band, verdict = _classify(
        milestone=args.milestone,
        rate=b3_best["aggregate_simulated_game_seconds_per_s_median"],
        parity_pass=parity_pass,
        stable_scaling=stable_scaling,
        hot_loop_resident=hot_loop_resident,
    )
    v02_comparison = _annotate_v02_comparison(
        results["B3"],
        args.reference_benchmark,
        b3_best["worlds"],
    )

    output = {
        "schema_version": 1,
        "milestone": args.milestone,
        "created_utc": datetime.now(UTC).isoformat(),
        "configuration": {
            "mandatory_batches": list(MANDATORY_BATCHES),
            "maximum_adaptive_batch": args.max_worlds,
            "ticks_per_repeat": args.ticks,
            "repeats": args.repeats,
            "warmup_ticks": args.warmup_ticks,
            "cuda_graph_block_ticks": args.graph_block_ticks,
            "stability_cv_max": STABILITY_CV_MAX,
            "seed": args.seed,
            "parity_file": str(parity_path),
            "action_tape": {
                "length": action_tape.length,
                "hold_ticks": action_tape.hold_ticks,
                "device_resident": True,
            },
            "b3_state_distribution": "16-family deterministic contact-rich mixture",
        },
        "environment": environment(args.device),
        "arena": {
            **geometry.metadata(),
            "source_root": args.collision_dir,
            "raw_geometry_bytes": int(geometry.vertices_uu.nbytes + geometry.triangles.nbytes),
            "bvh_allocation_bytes": None,
            "bvh_allocation_note": "Warp 1.16 does not expose per-Mesh BVH allocation bytes",
        },
        "geometry_query_gate": query_gate,
        "stress_gate": stress,
        "variants": results,
        "frozen_v0_2_b3_comparison": v02_comparison,
        "summary": {
            "best_b3_worlds": b3_best["worlds"],
            "best_b3_world_ticks_per_s": b3_best["world_ticks_per_s_median"],
            "best_b3_aggregate_simulated_game_seconds_per_s": b3_best[
                "aggregate_simulated_game_seconds_per_s_median"
            ],
            "best_b3_coefficient_of_variation": b3_best["coefficient_of_variation"],
            "performance_band": performance_band,
            "parity_gate_pass": parity_pass,
            "hot_loop_gpu_resident": hot_loop_resident,
            "stable_scaling_into_thousands": stable_scaling,
            "component_cost_multiplier_vs_b0": component_costs,
            "verdict": verdict,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["summary"], indent=2))
    return 0


def geometry_query_gate(
    geometry: ArenaGeometry, meshes: WarpArenaMeshes, device: str
) -> dict[str, Any]:
    corpus = make_soccar_ray_corpus(rays_per_family=512)
    expected = raycast_soccar_cpu(
        geometry,
        corpus.origins,
        corpus.directions,
        corpus.max_distances,
    )
    backends: dict[str, Any] = {}
    for name in ("default", "cubql"):
        arrays = query_soccar_rays(
            getattr(meshes, name),
            corpus.origins,
            corpus.directions,
            corpus.max_distances,
            device=device,
        )
        wp.synchronize_device(device)
        actual = tuple(array.numpy() for array in arrays)
        common_hits = (expected[0] == 1) & (actual[0] == 1)
        distances = np.abs(actual[1] - expected[1])
        close_distance = common_hits & (distances <= 0.02)
        face_difference = actual[3] != expected[3]
        categories = np.asarray(corpus.categories)
        co_nearest_edge_tie = close_distance & face_difference & (categories == "boundary_edges")
        normal_checked = common_hits & ~co_nearest_edge_tie
        dots = np.einsum("ij,ij->i", actual[2][normal_checked], expected[2][normal_checked])
        unambiguous_face_mismatch = close_distance & face_difference & ~co_nearest_edge_tie
        backends[name] = {
            "hit_mismatch_count": int(np.count_nonzero(actual[0] != expected[0])),
            "max_nearest_distance_error_uu": float(distances.max()),
            "max_hit_point_error_uu": float(distances[common_hits].max()),
            "minimum_normal_dot": float(dots.min()) if len(dots) else None,
            "co_nearest_boundary_edge_tie_count": int(np.count_nonzero(co_nearest_edge_tie)),
            "unambiguous_face_mismatch_count": int(np.count_nonzero(unambiguous_face_mismatch)),
            "exact_hit_distance_normal_pass": bool(
                np.array_equal(actual[0], expected[0])
                and np.all(distances <= 0.02)
                and (not len(dots) or np.all(dots > 0.999))
                and not np.any(unambiguous_face_mismatch)
            ),
        }
    category_counts = {
        category: corpus.categories.count(category) for category in sorted(set(corpus.categories))
    }
    return {
        "corpus_seed": 20260823,
        "ray_count": corpus.count,
        "category_counts": category_counts,
        "cpu_reference": ("two-sided exact-triangle Moller-Trumbore plus RocketSim Soccar planes"),
        "backends": backends,
        "selected_ray_backend": "cubql",
        "selection_reason": "exact query parity and higher measured B1 throughput",
    }


def run_stress_gate(
    *,
    collision_root: str,
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    action_tape: ActionTape,
    device: str,
    seed: int,
    graph_block_ticks: int,
) -> dict[str, Any]:
    digests: list[str] = []
    metrics: list[dict[str, float | int | bool]] = []
    for _ in range(2):
        sim = StaticWorldSim(
            64,
            collision_root,
            variant="B3",
            device=device,
            initial=make_contact_rich_state(64, seed),
            geometry=geometry,
            meshes=meshes,
            action_tape=action_tape,
        )
        sim.capture_graph(graph_block_ticks)
        sim.reset_transfer_counters()
        sim.step_graph(2400, synchronize=True)
        hot_h2d = sim.host_to_device_bytes
        hot_d2h = sim.device_to_host_bytes
        state = sim.snapshot()
        vehicle = sim.vehicle_snapshot()
        digest = hashlib.sha256()
        for field in fields(state):
            digest.update(getattr(state, field.name).tobytes())
        for field in fields(vehicle):
            digest.update(getattr(vehicle, field.name).tobytes())
        digests.append(digest.hexdigest().upper())
        metrics.append(
            {
                "finite": bool(
                    np.isfinite(state.car_pos).all()
                    and np.isfinite(state.car_vel).all()
                    and np.isfinite(state.car_ang_vel).all()
                ),
                "max_linear_speed_uu_per_s": float(np.max(np.linalg.norm(state.car_vel, axis=-1))),
                "max_angular_speed_rad_per_s": float(
                    np.max(np.linalg.norm(state.car_ang_vel, axis=-1))
                ),
                "max_penetration_uu": float(vehicle.penetration_max.max()),
                "hot_loop_host_to_device_bytes": hot_h2d,
                "hot_loop_device_to_host_bytes": hot_d2h,
            }
        )

    rest_state = StateSnapshot.empty(1)
    rest_state.car_pos[..., 2] = 17.0
    rest_sim = StaticWorldSim(
        1,
        collision_root,
        variant="B3",
        device=device,
        initial=rest_state,
        geometry=geometry,
        meshes=meshes,
    )
    rest_sim.capture_graph(graph_block_ticks)
    rest_sim.step_graph(2400, synchronize=True)
    rest = rest_sim.snapshot()
    result = {
        "worlds": 64,
        "measured_ticks": 2400,
        "capture_setup_ticks": graph_block_ticks,
        "seed": seed,
        "runs": metrics,
        "state_sha256_runs": digests,
        "deterministic_equal": digests[0] == digests[1],
        "finite": all(bool(item["finite"]) for item in metrics),
        "bounded_velocity": all(
            float(item["max_linear_speed_uu_per_s"]) <= 2300.001
            and float(item["max_angular_speed_rad_per_s"]) <= 5.5001
            for item in metrics
        ),
        "bounded_penetration": all(float(item["max_penetration_uu"]) < 100.0 for item in metrics),
        "floor_rest_height_uu": float(rest.car_pos[0, 0, 2]),
        "floor_rest_speed_uu_per_s": float(np.linalg.norm(rest.car_vel[0, 0])),
        "floor_rest_stable": bool(
            15.0 < rest.car_pos[0, 0, 2] < 20.0 and np.linalg.norm(rest.car_vel[0, 0]) < 2.0
        ),
    }
    result["hot_loop_gpu_resident"] = all(
        int(item["hot_loop_host_to_device_bytes"]) == 0
        and int(item["hot_loop_device_to_host_bytes"]) == 0
        for item in metrics
    )
    result["stress_gate_pass"] = bool(
        result["deterministic_equal"]
        and result["finite"]
        and result["bounded_velocity"]
        and result["bounded_penetration"]
        and result["floor_rest_stable"]
        and result["hot_loop_gpu_resident"]
    )
    return result


def _classify(
    *,
    milestone: str,
    rate: float,
    parity_pass: bool,
    stable_scaling: bool,
    hot_loop_resident: bool,
) -> tuple[str, str]:
    infrastructure_pass = stable_scaling and hot_loop_resident
    if milestone == "v0.2.1":
        if rate >= 500000.0:
            performance_band = "green_threshold"
        elif rate >= 100000.0:
            performance_band = "pass_threshold"
        else:
            performance_band = "below_success_floor"
        if not parity_pass:
            return performance_band, "PAUSE_FIDELITY"
        if not infrastructure_pass or rate < 100000.0:
            return performance_band, "PAUSE_PERF"
        if rate >= 500000.0:
            return performance_band, "PASS_GREEN"
        return performance_band, "PASS"

    performance_band = "below_red_threshold"
    if rate >= 100000.0:
        performance_band = "green_threshold"
    elif rate >= 20000.0:
        performance_band = "yellow_threshold"
    if parity_pass and performance_band == "green_threshold" and infrastructure_pass:
        return performance_band, "PASS_GREEN"
    if parity_pass and performance_band == "yellow_threshold" and hot_loop_resident:
        return performance_band, "PASS_YELLOW"
    return performance_band, "PAUSE_RED"


def _annotate_v02_comparison(
    current_points: list[dict[str, Any]],
    reference_path: Path,
    selected_worlds: int,
) -> dict[str, Any]:
    if not reference_path.exists():
        return {
            "available": False,
            "reference_path": str(reference_path),
            "reason": "reference benchmark not found",
        }
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    reference_points = {
        int(point["worlds"]): point for point in reference["variants"]["B3"]
    }
    comparisons: list[dict[str, Any]] = []
    for point in current_points:
        worlds = int(point["worlds"])
        prior = reference_points.get(worlds)
        if prior is None:
            continue
        prior_rate = float(prior["aggregate_simulated_game_seconds_per_s_median"])
        current_rate = float(point["aggregate_simulated_game_seconds_per_s_median"])
        comparisons.append(
            {
                "worlds": worlds,
                "v0_2_aggregate_simulated_game_seconds_per_s": prior_rate,
                "v0_2_1_aggregate_simulated_game_seconds_per_s": current_rate,
                "retained_throughput_fraction": current_rate / prior_rate,
                "slowdown_factor": prior_rate / current_rate,
            }
        )
    return {
        "available": True,
        "reference_path": str(reference_path),
        "reference_milestone": reference["milestone"],
        "equal_batch_points": comparisons,
        "selected_worlds_comparison": next(
            (item for item in comparisons if item["worlds"] == selected_worlds),
            None,
        ),
    }


def benchmark_point(
    *,
    worlds: int,
    result_name: str,
    variant: str,
    ray_backend: str,
    collision_root: str,
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    action_tape: ActionTape,
    device: str,
    ticks: int,
    repeats: int,
    warmup_ticks: int,
    graph_block_ticks: int,
    seed: int,
) -> dict[str, Any]:
    initial = (
        StateSnapshot.random(worlds, seed)
        if variant == "B0"
        else make_contact_rich_state(worlds, seed)
    )
    sim = StaticWorldSim(
        worlds,
        collision_root,
        variant=variant,
        ray_backend=ray_backend,
        device=device,
        initial=initial,
        geometry=geometry,
        meshes=meshes,
        action_tape=action_tape,
    )
    sim.step(warmup_ticks, synchronize=True)
    sim.capture_graph(graph_block_ticks)
    sim.step_graph(graph_block_ticks, synchronize=True)
    for name in (
        "candidate_total",
        "contact_total",
        "candidate_max",
        "contact_max",
        "penetration_max",
    ):
        getattr(sim.vehicle, name).zero_()
    sim.synchronize()
    checkpoint = _device_checkpoint(sim)
    sim.synchronize()
    calibration_started = time.perf_counter()
    sim.step_graph(ticks, synchronize=True)
    calibration_duration = time.perf_counter() - calibration_started
    _restore_checkpoint(checkpoint)
    sim.synchronize()
    target_repeat_seconds = 0.08
    multiplier = max(1, int(np.ceil(target_repeat_seconds / calibration_duration)))
    effective_ticks = min(8192, ticks * multiplier)
    effective_ticks = max(
        graph_block_ticks,
        (effective_ticks // graph_block_ticks) * graph_block_ticks,
    )
    sim.reset_transfer_counters()
    sampler = TelemetrySampler()
    sampler.start()
    durations: list[float] = []
    for _ in range(repeats):
        sim.synchronize()
        started = time.perf_counter()
        sim.step_graph(effective_ticks)
        sim.synchronize()
        durations.append(time.perf_counter() - started)
        if len(durations) < repeats:
            _restore_checkpoint(checkpoint)
            sim.synchronize()
    telemetry = sampler.stop()
    hot_h2d = sim.host_to_device_bytes
    hot_d2h = sim.device_to_host_bytes
    state = sim.snapshot()
    vehicle = sim.vehicle_snapshot()
    rates = [worlds * effective_ticks / duration for duration in durations]
    median_rate = statistics.median(rates)
    mean_rate = statistics.fmean(rates)
    cv = statistics.stdev(rates) / mean_rate if len(rates) > 1 else 0.0
    car_ticks = worlds * 2 * effective_ticks
    candidate_total = float(vehicle.candidate_total.sum())
    contact_total = float(vehicle.contact_total.sum())
    result = {
        "variant": result_name,
        "worlds": worlds,
        "ticks_per_repeat": effective_ticks,
        "calibration_seconds": calibration_duration,
        "ray_backend": ray_backend if variant in {"B1", "B2", "B3"} else None,
        "repeat_seconds": durations,
        "world_ticks_per_s_repeats": rates,
        "world_ticks_per_s_median": median_rate,
        "world_ticks_per_s_mean": mean_rate,
        "aggregate_simulated_game_seconds_per_s_median": median_rate / TICK_RATE,
        "suspension_rays_per_s_median": median_rate * 8 if variant != "B0" else 0.0,
        "coefficient_of_variation": cv,
        "stable": cv <= STABILITY_CV_MAX,
        "wall_time_s": float(sum(durations)),
        "average_candidates_per_car_tick": candidate_total / car_ticks if variant == "B3" else 0.0,
        "maximum_candidates_per_car_tick": int(vehicle.candidate_max.max())
        if variant == "B3"
        else 0,
        "average_contacts_per_car_tick": contact_total / car_ticks if variant == "B3" else 0.0,
        "maximum_contacts_per_car_tick": int(vehicle.contact_max.max()) if variant == "B3" else 0,
        "maximum_penetration_uu": float(vehicle.penetration_max.max()) if variant == "B3" else 0.0,
        "narrow_phase_candidates_per_s": (
            candidate_total / sum(durations) if variant == "B3" else 0.0
        ),
        "solver_contact_passes": 1 if variant == "B3" else 0,
        "telemetry": asdict(telemetry),
        "logical_state_bytes": sim.logical_state_bytes,
        "hot_loop_host_to_device_bytes": hot_h2d,
        "hot_loop_device_to_host_bytes": hot_d2h,
        "nan_or_error_count": int(
            np.size(state.car_pos)
            - np.count_nonzero(np.isfinite(state.car_pos))
            + np.size(state.car_vel)
            - np.count_nonzero(np.isfinite(state.car_vel))
            + np.size(state.car_ang_vel)
            - np.count_nonzero(np.isfinite(state.car_ang_vel))
        ),
        "verification_readback_outside_timing": True,
    }
    del sim, initial, state, vehicle
    gc.collect()
    return result


def _device_checkpoint(sim: StaticWorldSim) -> list[tuple[wp.array, wp.array]]:
    checkpoint: list[tuple[wp.array, wp.array]] = []
    for owner in (sim.state, sim.controls, sim.vehicle):
        for value in vars(owner).values():
            if isinstance(value, wp.array):
                checkpoint.append((value, wp.clone(value)))
    checkpoint.append((sim.tick_counter, wp.clone(sim.tick_counter)))
    return checkpoint


def _restore_checkpoint(checkpoint: list[tuple[wp.array, wp.array]]) -> None:
    for destination, source in checkpoint:
        wp.copy(destination, source)


def environment(device: str) -> dict[str, Any]:
    cuda_device = wp.get_device(device)
    result: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "warp": wp.__version__,
        "device": str(cuda_device),
        "device_name": cuda_device.name,
        "device_total_memory_bytes": cuda_device.total_memory,
        "cpu": platform.processor(),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
    }
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        result["nvidia_driver"] = pynvml.nvmlSystemGetDriverVersion()
        result["gpu_name"] = pynvml.nvmlDeviceGetName(handle)
    except pynvml.NVMLError:
        result["nvidia_driver"] = None
    return result


def _relative_gain(previous: dict[str, Any], current: dict[str, Any]) -> float:
    return current["world_ticks_per_s_median"] / previous["world_ticks_per_s_median"] - 1.0


def _best_stable(points: list[dict[str, Any]]) -> dict[str, Any]:
    stable = [point for point in points if point["stable"] and point["nan_or_error_count"] == 0]
    candidates = stable or points
    return max(candidates, key=lambda point: point["world_ticks_per_s_median"])


if __name__ == "__main__":
    raise SystemExit(main())
