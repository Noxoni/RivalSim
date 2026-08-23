"""Run the prescribed repeated CPU/GPU v0.1 throughput sweep."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import sys
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import pynvml
import warp as wp

from rivalsim.constants import TICK_RATE
from rivalsim.controls import ControlBatch
from rivalsim.reference.cpu_simple import CpuSimulator
from rivalsim.simulator import RivalSim
from rivalsim.state import StateSnapshot

MANDATORY_BATCHES = (256, 512, 1024, 2048, 4096, 8192, 16384)
FULL_ROCKETSIM_REFERENCE_SIM_SECONDS_PER_SECOND = 200.65
STABILITY_CV_MAX = 0.05


@dataclass(slots=True)
class RunTelemetry:
    samples: int
    gpu_util_mean_percent: float | None
    gpu_util_max_percent: float | None
    gpu_memory_mean_bytes: float | None
    gpu_memory_max_bytes: int | None
    process_cpu_mean_percent: float
    process_cpu_max_percent: float
    system_cpu_mean_percent: float


class TelemetrySampler:
    def __init__(self, interval_s: float = 0.01):
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[tuple[float | None, int | None, float, float]] = []
        self._process = psutil.Process()
        self._nvml_handle: Any | None = None
        try:
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except pynvml.NVMLError:
            self._nvml_handle = None

    def start(self) -> None:
        self._process.cpu_percent(None)
        psutil.cpu_percent(None)
        self._sample()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> RunTelemetry:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._sample()
        gpu_utils = [item[0] for item in self._samples if item[0] is not None]
        gpu_memory = [item[1] for item in self._samples if item[1] is not None]
        process_cpu = [item[2] for item in self._samples]
        system_cpu = [item[3] for item in self._samples]
        return RunTelemetry(
            samples=len(self._samples),
            gpu_util_mean_percent=_mean_or_none(gpu_utils),
            gpu_util_max_percent=max(gpu_utils) if gpu_utils else None,
            gpu_memory_mean_bytes=_mean_or_none(gpu_memory),
            gpu_memory_max_bytes=max(gpu_memory) if gpu_memory else None,
            process_cpu_mean_percent=statistics.fmean(process_cpu) if process_cpu else 0.0,
            process_cpu_max_percent=max(process_cpu) if process_cpu else 0.0,
            system_cpu_mean_percent=statistics.fmean(system_cpu) if system_cpu else 0.0,
        )

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample()

    def _sample(self) -> None:
        gpu_util: float | None = None
        gpu_memory: int | None = None
        if self._nvml_handle is not None:
            try:
                gpu_util = float(pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle).gpu)
                gpu_memory = int(pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle).used)
            except pynvml.NVMLError:
                pass
        self._samples.append(
            (
                gpu_util,
                gpu_memory,
                float(self._process.cpu_percent(None)),
                float(psutil.cpu_percent(None)),
            )
        )


def run_benchmark(
    *,
    device: str,
    gpu_ticks: int,
    cpu_ticks: int,
    gpu_repeats: int,
    cpu_repeats: int,
    warmup_ticks: int,
    graph_block_ticks: int,
    max_worlds: int,
    seed: int,
) -> dict[str, Any]:
    wp.init()
    pynvml.nvmlInit()
    gpu_results: list[dict[str, Any]] = []
    cpu_results: list[dict[str, Any]] = []

    for worlds in MANDATORY_BATCHES:
        gpu_results.append(
            _benchmark_gpu(
                worlds,
                ticks=gpu_ticks,
                repeats=gpu_repeats,
                warmup_ticks=warmup_ticks,
                graph_block_ticks=graph_block_ticks,
                device=device,
                seed=seed,
            )
        )

    # Continue doubling only while the mandatory endpoint still scales materially.
    if _relative_gain(gpu_results[-2], gpu_results[-1]) >= 0.05:
        worlds = MANDATORY_BATCHES[-1] * 2
        while worlds <= max_worlds:
            extended = _benchmark_gpu(
                worlds,
                ticks=gpu_ticks,
                repeats=gpu_repeats,
                warmup_ticks=warmup_ticks,
                graph_block_ticks=graph_block_ticks,
                device=device,
                seed=seed,
            )
            gpu_results.append(extended)
            if _relative_gain(gpu_results[-2], gpu_results[-1]) < 0.05:
                break
            worlds *= 2

    for worlds in MANDATORY_BATCHES:
        cpu_results.append(
            _benchmark_cpu(
                worlds,
                ticks=cpu_ticks,
                repeats=cpu_repeats,
                seed=seed,
            )
        )

    best_gpu = max(
        (item for item in gpu_results if item["stable"]),
        key=lambda item: item["aggregate_simulated_game_seconds_per_s_median"],
    )
    best_cpu = max(
        (item for item in cpu_results if item["stable"]),
        key=lambda item: item["aggregate_simulated_game_seconds_per_s_median"],
    )
    speedup = (
        best_gpu["aggregate_simulated_game_seconds_per_s_median"]
        / best_cpu["aggregate_simulated_game_seconds_per_s_median"]
    )
    non_apples_ratio = (
        best_gpu["aggregate_simulated_game_seconds_per_s_median"]
        / FULL_ROCKETSIM_REFERENCE_SIM_SECONDS_PER_SECOND
    )
    scaling_points = [
        item for item in gpu_results if item["worlds"] in (2048, 4096, 8192, 16384)
    ]
    scaling_into_thousands = all(item["stable"] for item in scaling_points) and all(
        current["world_ticks_per_s_median"] > previous["world_ticks_per_s_median"]
        for previous, current in pairwise(scaling_points)
    )
    no_hot_transfers = all(
        item["hot_loop_host_to_device_bytes"] == 0 and item["hot_loop_device_to_host_bytes"] == 0
        for item in gpu_results
    )
    performance_conditions = {
        "gpu_at_least_4x_same_equation_cpu": speedup >= 4.0,
        "gpu_at_least_2000_simulated_game_seconds_per_s": best_gpu[
            "aggregate_simulated_game_seconds_per_s_median"
        ]
        >= 2000.0,
        "gpu_at_least_10x_full_rocketsim_system_reference": non_apples_ratio >= 10.0,
        "stable_scaling_into_thousands": scaling_into_thousands,
        "hot_loop_gpu_resident": no_hot_transfers,
        "no_nans_or_errors": all(
            not item["nan_or_error"] for item in (*gpu_results, *cpu_results)
        ),
    }
    return {
        "schema_version": 1,
        "milestone": "v0.1",
        "created_utc": datetime.now(UTC).isoformat(),
        "configuration": {
            "dt_s": 1.0 / TICK_RATE,
            "gpu_ticks_per_repeat": gpu_ticks,
            "extended_gpu_ticks_per_repeat": gpu_ticks,
            "cpu_ticks_per_repeat": cpu_ticks,
            "gpu_repeats": gpu_repeats,
            "cpu_repeats": cpu_repeats,
            "warmup_ticks": warmup_ticks,
            "cuda_graph_block_ticks": graph_block_ticks,
            "stability_cv_max": STABILITY_CV_MAX,
            "seed": seed,
            "mandatory_batches": list(MANDATORY_BATCHES),
            "maximum_adaptive_batch": max_worlds,
            "adaptive_extension_rule": (
                "double while median world-tick throughput gains >= 5 percent"
            ),
        },
        "environment": _environment(device),
        "gpu": gpu_results,
        "cpu_same_equation": cpu_results,
        "summary": {
            "best_gpu_worlds": best_gpu["worlds"],
            "best_gpu_world_ticks_per_s_median": best_gpu["world_ticks_per_s_median"],
            "best_gpu_aggregate_simulated_game_seconds_per_s": best_gpu[
                "aggregate_simulated_game_seconds_per_s_median"
            ],
            "best_cpu_worlds": best_cpu["worlds"],
            "best_cpu_world_ticks_per_s_median": best_cpu["world_ticks_per_s_median"],
            "best_cpu_aggregate_simulated_game_seconds_per_s": best_cpu[
                "aggregate_simulated_game_seconds_per_s_median"
            ],
            "same_equation_gpu_speedup": speedup,
            "full_rocketsim_system_reference_simulated_game_seconds_per_s": (
                FULL_ROCKETSIM_REFERENCE_SIM_SECONDS_PER_SECOND
            ),
            "non_apples_to_apples_ratio_vs_full_rocketsim_system_reference": non_apples_ratio,
            "performance_conditions": performance_conditions,
            "performance_gate_pass": all(performance_conditions.values()),
        },
        "comparison_note": (
            "The 200.65 sim-s/s Rival RocketSim/RLGym value is a full CPU system reference; "
            "its ratio to this contact-free kernel is explicitly not an apples-to-apples speedup."
        ),
    }


def _benchmark_gpu(
    worlds: int,
    *,
    ticks: int,
    repeats: int,
    warmup_ticks: int,
    graph_block_ticks: int,
    device: str,
    seed: int,
) -> dict[str, Any]:
    initial = StateSnapshot.random(worlds, seed + worlds)
    controls = _random_controls(worlds, seed ^ worlds)
    sim = RivalSim(worlds, device=device, randomize=False)
    sim.reset(initial)
    sim.set_controls(controls)
    sim.step(warmup_ticks, synchronize=True)
    start_event = wp.Event(device=device, enable_timing=True)
    end_event = wp.Event(device=device, enable_timing=True)
    wall_times: list[float] = []
    kernel_times: list[float] = []
    nan_or_error = False
    transfer_records: list[tuple[int, int]] = []

    for _ in range(repeats):
        sim.reset(initial)
        sim.set_controls(controls)
        sim.synchronize()
        sim.capture_graph(graph_block_ticks)
        sim.reset_transfer_counters()
        wp.record_event(start_event)
        started = time.perf_counter()
        sim.step_graph(ticks)
        wp.record_event(end_event)
        sim.synchronize()
        elapsed = time.perf_counter() - started
        wall_times.append(elapsed)
        kernel_times.append(float(wp.get_event_elapsed_time(start_event, end_event)) / 1000.0)
        transfer_records.append((sim.host_to_device_bytes, sim.device_to_host_bytes))
        try:
            sim.snapshot().validate()
        except ValueError:
            nan_or_error = True

    # Telemetry is sampled in a separate pass so NVML/psutil polling cannot perturb
    # the Python launch loop used for throughput and variance.
    sim.reset(initial)
    sim.set_controls(controls)
    sim.synchronize()
    sim.capture_graph(graph_block_ticks)
    sampler = TelemetrySampler()
    sampler.start()
    sim.step_graph(ticks, synchronize=True)
    telemetry = sampler.stop()

    throughputs = [worlds * ticks / elapsed for elapsed in wall_times]
    sim_seconds = [item / TICK_RATE for item in throughputs]
    coefficient = _coefficient_of_variation(sim_seconds)
    result = {
        "worlds": worlds,
        "ticks_per_repeat": ticks,
        "launch_mode": "cuda_graph",
        "cuda_graph_block_ticks": graph_block_ticks,
        "simulated_seconds_per_world_per_repeat": ticks / TICK_RATE,
        "wall_time_s_runs": wall_times,
        "kernel_loop_time_s_runs": kernel_times,
        "world_ticks_per_s_runs": throughputs,
        "world_ticks_per_s_median": statistics.median(throughputs),
        "aggregate_simulated_game_seconds_per_s_runs": sim_seconds,
        "aggregate_simulated_game_seconds_per_s_median": statistics.median(sim_seconds),
        "aggregate_simulated_game_seconds_per_s_mean": statistics.fmean(sim_seconds),
        "aggregate_simulated_game_seconds_per_s_stdev": statistics.stdev(sim_seconds)
        if len(sim_seconds) > 1
        else 0.0,
        "coefficient_of_variation": coefficient,
        "gpu_utilization_mean_percent": telemetry.gpu_util_mean_percent,
        "gpu_utilization_max_percent": telemetry.gpu_util_max_percent,
        "gpu_telemetry_samples": telemetry.samples,
        "vram_device_used_peak_bytes": telemetry.gpu_memory_max_bytes,
        "warp_mempool_used_current_bytes": int(wp.get_mempool_used_mem_current(device)),
        "warp_mempool_used_high_bytes": int(wp.get_mempool_used_mem_high(device)),
        "logical_state_bytes": initial.nbytes,
        "logical_controls_bytes": controls.nbytes,
        "process_cpu_mean_percent": telemetry.process_cpu_mean_percent,
        "process_cpu_max_percent": telemetry.process_cpu_max_percent,
        "system_cpu_mean_percent": telemetry.system_cpu_mean_percent,
        "telemetry_pass_ticks": ticks,
        "hot_loop_host_to_device_bytes": max(item[0] for item in transfer_records),
        "hot_loop_device_to_host_bytes": max(item[1] for item in transfer_records),
        "verification_readback_bytes_per_run": initial.nbytes,
        "nan_or_error": nan_or_error,
        "stable": (not nan_or_error) and coefficient <= STABILITY_CV_MAX,
    }
    del sim
    gc.collect()
    return _json_scalars(result)


def _benchmark_cpu(
    worlds: int,
    *,
    ticks: int,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    initial = StateSnapshot.random(worlds, seed + worlds)
    controls = _random_controls(worlds, seed ^ worlds)
    wall_times: list[float] = []
    nan_or_error = False
    warmup = CpuSimulator(initial, controls)
    warmup.step(min(30, ticks))
    for _ in range(repeats):
        sim = CpuSimulator(initial, controls)
        started = time.perf_counter()
        sim.step(ticks)
        wall_times.append(time.perf_counter() - started)
        try:
            sim.snapshot().validate()
        except ValueError:
            nan_or_error = True

    telemetry_sim = CpuSimulator(initial, controls)
    telemetry_ticks = min(ticks, 120)
    sampler = TelemetrySampler()
    sampler.start()
    telemetry_sim.step(telemetry_ticks)
    telemetry = sampler.stop()

    throughputs = [worlds * ticks / elapsed for elapsed in wall_times]
    sim_seconds = [item / TICK_RATE for item in throughputs]
    coefficient = _coefficient_of_variation(sim_seconds)
    return _json_scalars(
        {
            "worlds": worlds,
            "ticks_per_repeat": ticks,
            "simulated_seconds_per_world_per_repeat": ticks / TICK_RATE,
            "wall_time_s_runs": wall_times,
            "world_ticks_per_s_runs": throughputs,
            "world_ticks_per_s_median": statistics.median(throughputs),
            "aggregate_simulated_game_seconds_per_s_runs": sim_seconds,
            "aggregate_simulated_game_seconds_per_s_median": statistics.median(sim_seconds),
            "aggregate_simulated_game_seconds_per_s_mean": statistics.fmean(sim_seconds),
            "aggregate_simulated_game_seconds_per_s_stdev": statistics.stdev(sim_seconds)
            if len(sim_seconds) > 1
            else 0.0,
            "coefficient_of_variation": coefficient,
            "process_cpu_mean_percent": telemetry.process_cpu_mean_percent,
            "process_cpu_max_percent": telemetry.process_cpu_max_percent,
            "system_cpu_mean_percent": telemetry.system_cpu_mean_percent,
            "telemetry_pass_ticks": telemetry_ticks,
            "logical_state_bytes": initial.nbytes,
            "logical_controls_bytes": controls.nbytes,
            "nan_or_error": nan_or_error,
            "stable": (not nan_or_error) and coefficient <= STABILITY_CV_MAX,
        }
    )


def _environment(device: str) -> dict[str, Any]:
    gpu = wp.get_device(device)
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    cuda_driver_raw = pynvml.nvmlSystemGetCudaDriverVersion_v2()
    cuda_driver = f"{cuda_driver_raw // 1000}.{(cuda_driver_raw % 1000) // 10}"
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "numpy": np.__version__,
        "warp": wp.__version__,
        "warp_cuda_toolkit": "12.9",
        "nvidia_driver": pynvml.nvmlSystemGetDriverVersion(),
        "cuda_driver": cuda_driver,
        "gpu": pynvml.nvmlDeviceGetName(handle),
        "gpu_arch": f"sm_{gpu.arch}",
        "gpu_sm_count": gpu.sm_count,
        "gpu_total_memory_bytes": pynvml.nvmlDeviceGetMemoryInfo(handle).total,
        "device": str(gpu),
        "mempool_enabled": gpu.is_mempool_enabled,
    }


def _random_controls(worlds: int, seed: int) -> ControlBatch:
    rng = np.random.default_rng(seed)
    controls = ControlBatch.zeros(worlds)
    controls.throttle[...] = rng.uniform(-1.0, 1.0, controls.throttle.shape)
    controls.pitch[...] = rng.uniform(-1.0, 1.0, controls.pitch.shape)
    controls.yaw[...] = rng.uniform(-1.0, 1.0, controls.yaw.shape)
    controls.roll[...] = rng.uniform(-1.0, 1.0, controls.roll.shape)
    controls.boost[...] = rng.random(controls.boost.shape) < 0.35
    return controls


def _relative_gain(previous: dict[str, Any], current: dict[str, Any]) -> float:
    before = previous["world_ticks_per_s_median"]
    return (current["world_ticks_per_s_median"] - before) / before


def _coefficient_of_variation(values: list[float]) -> float:
    mean = statistics.fmean(values)
    return statistics.stdev(values) / mean if len(values) > 1 and mean else 0.0


def _mean_or_none(values: list[float | int]) -> float | None:
    return statistics.fmean(values) if values else None


def _mean_optional(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return statistics.fmean(present) if present else None


def _max_optional(values: list[float | int | None]) -> float | int | None:
    present = [value for value in values if value is not None]
    return max(present) if present else None


def _json_scalars(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_scalars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_scalars(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-ticks", type=int, default=6000)
    parser.add_argument("--cpu-ticks", type=int, default=360)
    parser.add_argument("--gpu-repeats", type=int, default=5)
    parser.add_argument("--cpu-repeats", type=int, default=3)
    parser.add_argument("--warmup-ticks", type=int, default=120)
    parser.add_argument("--graph-block-ticks", type=int, default=8)
    parser.add_argument("--max-worlds", type=int, default=1_048_576)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_benchmark(
        device=args.device,
        gpu_ticks=args.gpu_ticks,
        cpu_ticks=args.cpu_ticks,
        gpu_repeats=args.gpu_repeats,
        cpu_repeats=args.cpu_repeats,
        warmup_ticks=args.warmup_ticks,
        graph_block_ticks=args.graph_block_ticks,
        max_worlds=args.max_worlds,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
