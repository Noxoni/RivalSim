"""End-to-end Rival 2.0 rollout/PPO world-count benchmark."""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import statistics
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pynvml
import torch
import warp as wp

from benchmarks.run_v02_benchmark import TelemetrySampler, environment
from rivalsim.arena import ArenaGeometry, WarpArenaMeshes
from rivalsim.rival2_env import Rival2Env
from rivalsim.rival2_policy import sample_hybrid_action
from rivalsim.rival2_ppo import Rival2PPOConfig, ppo_update
from rivalsim.rival2_training import Rival2Trainer

WORLD_COUNTS = (8192, 16384, 32768, 65536, 131072)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collision-dir", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worlds", nargs="+", type=int, default=WORLD_COUNTS)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=65536)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260825)
    return parser.parse_args()


def _cuda_time[T](call: Callable[[], T]) -> tuple[T, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = call()
    end.record()
    end.synchronize()
    return result, float(start.elapsed_time(end)) / 1000.0


def _cv(values: list[float]) -> float:
    return statistics.pstdev(values) / statistics.fmean(values) if len(values) > 1 else 0.0


def benchmark_point(
    worlds: int,
    *,
    collision_dir: str,
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    args: argparse.Namespace,
) -> dict[str, Any]:
    config = Rival2PPOConfig(
        rollout_horizon=args.horizon,
        minibatch_size=args.minibatch_size,
        epochs=2,
        entropy_coefficient=0.0,
    )
    env = Rival2Env(
        worlds,
        collision_dir,
        device=args.device,
        seed=args.seed,
        geometry=geometry,
        meshes=meshes,
    )
    trainer = Rival2Trainer(env, ppo_config=config, seed=args.seed)
    warmup = trainer.collect_rollout()
    rollout_buffer_bytes = warmup.logical_bytes
    trainer.update(warmup)
    del warmup
    torch.cuda.synchronize()

    with torch.no_grad():
        (actor_output, _), actor_seconds = _cuda_time(
            lambda: trainer.model(env.observation.reshape(-1, env.observation.shape[-1]))
        )
        sample = sample_hybrid_action(
            actor_output.reshape(worlds, 2, 13),
            generator=trainer.policy_generator,
            config=trainer.policy_config,
        )
        profile = env.step_profiled(sample.action)
    del actor_output, sample

    repeats: list[dict[str, Any]] = []
    for repeat in range(args.repeats):
        torch.cuda.reset_peak_memory_stats(args.device)
        env.reset_transfer_counters()
        telemetry = TelemetrySampler(interval_s=0.005)
        telemetry.start()
        wall_start = time.perf_counter()
        rollout, rollout_seconds = _cuda_time(trainer.collect_rollout)

        def compute_gae(buffer=rollout):
            return buffer.compute_gae(config)

        _, gae_seconds = _cuda_time(compute_gae)

        def update(buffer=rollout):
            return ppo_update(
                trainer.model,
                trainer.optimizer,
                buffer,
                config,
                generator=trainer.policy_generator,
                policy_config=trainer.policy_config,
                gae_ready=True,
            )

        metrics, ppo_seconds = _cuda_time(update)
        torch.cuda.synchronize()
        wall_seconds = time.perf_counter() - wall_start
        telemetry_result = telemetry.stop()
        trainer.policy_version += 1
        trainer.iteration += 1
        samples = worlds * 2 * args.horizon
        simulated_seconds = worlds * args.horizon / 30.0
        repeats.append(
            {
                "repeat": repeat,
                "rollout_seconds": rollout_seconds,
                "gae_seconds": gae_seconds,
                "ppo_seconds": ppo_seconds,
                "wall_seconds": wall_seconds,
                "agent_decisions_per_s": samples / rollout_seconds,
                "ppo_samples_per_s": samples / ppo_seconds,
                "simulated_game_seconds_per_s": simulated_seconds / rollout_seconds,
                "complete_iteration_agent_samples_per_s": samples / wall_seconds,
                "update_frequency_hz": 1.0 / wall_seconds,
                "sample_age_policy_versions": 0,
                "hot_loop_h2d_bytes": env.world.host_to_device_bytes,
                "hot_loop_d2h_bytes": env.world.device_to_host_bytes,
                "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device),
                "vram_peak_observed_bytes": telemetry_result.vram_max_bytes,
                "gpu_utilization_mean_percent": telemetry_result.gpu_util_mean_percent,
                "metrics": {name: float(value) for name, value in metrics.items()},
            }
        )
        del rollout

    def median(name: str) -> float:
        return statistics.median(float(item[name]) for item in repeats)

    point = {
        "worlds": worlds,
        "horizon": args.horizon,
        "minibatch_size": args.minibatch_size,
        "repeats": repeats,
        "rollout_seconds_median": median("rollout_seconds"),
        "gae_seconds_median": median("gae_seconds"),
        "ppo_seconds_median": median("ppo_seconds"),
        "wall_seconds_median": median("wall_seconds"),
        "agent_decisions_per_s_median": median("agent_decisions_per_s"),
        "ppo_samples_per_s_median": median("ppo_samples_per_s"),
        "simulated_game_seconds_per_s_median": median("simulated_game_seconds_per_s"),
        "complete_iteration_agent_samples_per_s_median": median(
            "complete_iteration_agent_samples_per_s"
        ),
        "wall_time_cv": _cv([float(item["wall_seconds"]) for item in repeats]),
        "rollout_time_cv": _cv([float(item["rollout_seconds"]) for item in repeats]),
        "ppo_time_cv": _cv([float(item["ppo_seconds"]) for item in repeats]),
        "vram_peak_observed_bytes": max(
            int(item["vram_peak_observed_bytes"] or 0) for item in repeats
        ),
        "torch_peak_allocated_bytes": max(
            int(item["torch_peak_allocated_bytes"]) for item in repeats
        ),
        "rollout_buffer_bytes": rollout_buffer_bytes,
        "phase_profile_milliseconds": profile.milliseconds,
        "actor_critic_inference_milliseconds": actor_seconds * 1000.0,
        "all_tensor_aliases": all(item["aliases"] for item in env.bridge.alias_report().values()),
        "hot_loop_h2d_bytes": max(int(item["hot_loop_h2d_bytes"]) for item in repeats),
        "hot_loop_d2h_bytes": max(int(item["hot_loop_d2h_bytes"]) for item in repeats),
    }
    return point


def main() -> int:
    args = parse_args()
    if args.repeats < 5:
        raise ValueError("v0.5 release points require at least five repeats")
    wp.init()
    with contextlib.suppress(pynvml.NVMLError):
        pynvml.nvmlInit()
    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    meshes = WarpArenaMeshes(geometry, args.device)
    points: list[dict[str, Any]] = []
    boundaries: list[dict[str, Any]] = []
    for worlds in args.worlds:
        try:
            point = benchmark_point(
                worlds,
                collision_dir=args.collision_dir,
                geometry=geometry,
                meshes=meshes,
                args=args,
            )
            points.append(point)
            print(
                worlds,
                point["complete_iteration_agent_samples_per_s_median"],
                point["wall_time_cv"],
                point["vram_peak_observed_bytes"],
                flush=True,
            )
        except (RuntimeError, MemoryError) as exc:
            boundaries.append({"worlds": worlds, "error": str(exc)})
        gc.collect()
        torch.cuda.empty_cache()
    stable = [point for point in points if float(point["wall_time_cv"]) < 0.05]
    selected = max(
        stable,
        key=lambda point: float(point["complete_iteration_agent_samples_per_s_median"]),
    )
    result = {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "workload": "Rival2 complete rollout + GAE + PPO",
        "environment": environment(args.device),
        "ppo_config": {
            "horizon": args.horizon,
            "minibatch_size": args.minibatch_size,
            "epochs": 2,
            "entropy_coefficient": 0.0,
        },
        "points": points,
        "boundaries": boundaries,
        "selected_worlds": selected["worlds"],
        "selected_point": selected,
        "stability_cv_limit": 0.05,
        "verdict": "PASS_GREEN"
        if points
        and selected["wall_time_cv"] < 0.05
        and all(point["hot_loop_h2d_bytes"] == 0 for point in points)
        and all(point["hot_loop_d2h_bytes"] == 0 for point in points)
        else "FAIL_RED",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0 if result["verdict"] == "PASS_GREEN" else 1


if __name__ == "__main__":
    raise SystemExit(main())
