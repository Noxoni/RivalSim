"""Calibrate and test the source-exact ground-to-air possession option."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_rival2_aerial_option_v1 as aerial_v1  # noqa: E402
from rivalsim.arena import ArenaGeometry, WarpArenaMeshes  # noqa: E402
from rivalsim.rival2_aerial_intercept_teacher import (  # noqa: E402
    plan_aerial_intercept,
)
from rivalsim.rival2_aerial_option import FIELD  # noqa: E402
from rivalsim.rival2_contracts import (  # noqa: E402
    BALL_LINEAR_SPEED_SCALE,
    POSITION_SCALE,
    RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION,
)
from rivalsim.rival2_env import Rival2Env  # noqa: E402
from rivalsim.rival2_ground_to_air_option import (  # noqa: E402
    GROUND_TO_AIR_OPTION_VERSION,
    GroundToAirConfig,
    GroundToAirController,
    GroundToAirTracker,
    build_ground_to_air_scenarios,
)

OUTPUT = ROOT / "results/rival2/ground_to_air_option_v1/calibration.json"
CALIBRATION_SEED = 2_026_090_401
TEST_SEED = 2_026_090_499


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_one(
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    *,
    config: GroundToAirConfig,
    side: int,
    worlds: int,
    horizon: int,
    seed: int,
    device: str,
    collision_dir: Path,
) -> dict[str, Any]:
    batch = build_ground_to_air_scenarios(worlds, seed=seed ^ side, attacker_side=side)
    env = Rival2Env(
        worlds,
        str(collision_dir),
        device=device,
        seed=seed ^ side,
        reward_version=RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION,
        initial=batch.state,
        geometry=geometry,
        meshes=meshes,
        car_visitation_order="a_then_b",
    )
    controller = GroundToAirController(worlds, device=device, config=config)
    tracker = GroundToAirTracker(worlds, attacker_side=side, horizon=horizon)
    active = torch.ones(worlds, dtype=torch.bool, device=device)
    observation = env.observation
    zero = torch.zeros((worlds, 8), dtype=torch.float32, device=device)
    false = torch.zeros(worlds, dtype=torch.bool, device=device)
    inf = torch.full((worlds,), float("inf"), dtype=torch.float32, device=device)
    minimum_distance = inf.clone()
    closest_tick = torch.full((worlds,), -1, dtype=torch.int64, device=device)
    closest_relative = torch.zeros((worlds, 3), dtype=torch.float32, device=device)
    closest_car_height = torch.zeros(worlds, dtype=torch.float32, device=device)
    closest_ball_height = torch.zeros(worlds, dtype=torch.float32, device=device)
    maximum_car_height = torch.zeros(worlds, dtype=torch.float32, device=device)
    maximum_ball_height = torch.zeros(worlds, dtype=torch.float32, device=device)
    maximum_plan_alignment = torch.full((worlds,), -1.0, dtype=torch.float32, device=device)
    boosted_pursuit_ticks = torch.zeros(worlds, dtype=torch.int64, device=device)
    sampled_offsets = (10, 20, 30, 40, 50, 60, 75, 90)
    offset_samples = {
        offset: {
            name: torch.full((worlds,), float("nan"), dtype=torch.float32, device=device)
            for name in (
                "distance_uu",
                "relative_x_uu",
                "relative_y_uu",
                "relative_z_uu",
                "car_height_uu",
                "ball_height_uu",
                "forward_x",
                "forward_y",
                "forward_z",
                "relative_velocity_x_uu_per_second",
                "relative_velocity_y_uu_per_second",
                "relative_velocity_z_uu_per_second",
            )
        }
        for offset in sampled_offsets
    }
    for tick in range(horizon):
        active_before = active.clone()
        step = controller.step(
            zero,
            observation[:, side],
            kickoff_active=false,
            match_done=~active_before,
        )
        action = torch.zeros((worlds, 2, 8), dtype=torch.float32, device=device)
        action[:, side] = torch.where(active_before[:, None], step.action, 0.0)
        transition = env.step(action)
        scoring = env.bridge.views["rival2.scoring_team_latched"].to(torch.int64)
        goal_for = active_before & transition.terminated & (scoring == side)
        done = tracker.step(
            observation,
            transition.transition_observation,
            tick=tick,
            goal_for_attacker=goal_for,
            active=active_before,
        )
        attacker_observation = transition.transition_observation[:, side]
        after_pop = active_before & tracker.pop_touch
        relative = torch.stack(
            [attacker_observation[:, FIELD[f"relative.ball_position.{axis}"]] for axis in "xyz"],
            dim=-1,
        ) * torch.as_tensor(
            POSITION_SCALE,
            dtype=attacker_observation.dtype,
            device=attacker_observation.device,
        )
        distance = torch.linalg.vector_norm(relative, dim=-1)
        relative_velocity = (
            torch.stack(
                [
                    attacker_observation[:, FIELD[f"relative.ball_velocity.{axis}"]]
                    for axis in "xyz"
                ],
                dim=-1,
            )
            * BALL_LINEAR_SPEED_SCALE
        )
        forward = torch.stack(
            [attacker_observation[:, FIELD[f"self.forward.{axis}"]] for axis in "xyz"],
            dim=-1,
        )
        car_height = attacker_observation[:, FIELD["self.position.z"]] * POSITION_SCALE[2]
        ball_height = attacker_observation[:, FIELD["ball.position.z"]] * POSITION_SCALE[2]
        maximum_car_height.copy_(
            torch.where(
                after_pop, torch.maximum(maximum_car_height, car_height), maximum_car_height
            )
        )
        maximum_ball_height.copy_(
            torch.where(
                after_pop, torch.maximum(maximum_ball_height, ball_height), maximum_ball_height
            )
        )
        closer = after_pop & (distance < minimum_distance)
        minimum_distance.copy_(torch.where(closer, distance, minimum_distance))
        closest_tick.copy_(torch.where(closer, torch.full_like(closest_tick, tick), closest_tick))
        closest_relative.copy_(torch.where(closer[:, None], relative, closest_relative))
        closest_car_height.copy_(torch.where(closer, car_height, closest_car_height))
        closest_ball_height.copy_(torch.where(closer, ball_height, closest_ball_height))
        plan = plan_aerial_intercept(attacker_observation)
        pursuit_after_pop = after_pop & step.pursuit
        maximum_plan_alignment.copy_(
            torch.where(
                pursuit_after_pop,
                torch.maximum(maximum_plan_alignment, plan.nose_alignment),
                maximum_plan_alignment,
            )
        )
        boosted_pursuit_ticks += (pursuit_after_pop & (plan.action[:, 6] >= 0.5)).to(torch.int64)
        pop_offset = tick - tracker.pop_tick
        for offset, samples in offset_samples.items():
            sample = after_pop & (pop_offset == offset)
            samples["distance_uu"].copy_(torch.where(sample, distance, samples["distance_uu"]))
            for axis_index, axis in enumerate("xyz"):
                samples[f"relative_{axis}_uu"].copy_(
                    torch.where(
                        sample,
                        relative[:, axis_index],
                        samples[f"relative_{axis}_uu"],
                    )
                )
            samples["car_height_uu"].copy_(
                torch.where(sample, car_height, samples["car_height_uu"])
            )
            samples["ball_height_uu"].copy_(
                torch.where(sample, ball_height, samples["ball_height_uu"])
            )
            for axis_index, axis in enumerate("xyz"):
                samples[f"forward_{axis}"].copy_(
                    torch.where(sample, forward[:, axis_index], samples[f"forward_{axis}"])
                )
                samples[f"relative_velocity_{axis}_uu_per_second"].copy_(
                    torch.where(
                        sample,
                        relative_velocity[:, axis_index],
                        samples[f"relative_velocity_{axis}_uu_per_second"],
                    )
                )
        active &= ~(done | transition.terminated | transition.truncated)
        observation = transition.observation
        if not bool(active.any()):
            break
    torch.cuda.synchronize()
    telemetry = asdict(tracker.telemetry)
    observed = tracker.pop_touch

    def distribution(value: torch.Tensor, mask: torch.Tensor = observed) -> dict[str, float]:
        selected = value[mask].to(torch.float32)
        if selected.numel() == 0:
            return {"count": 0, "mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0}
        return {
            "count": int(selected.numel()),
            "mean": float(selected.mean()),
            "p10": float(torch.quantile(selected, 0.10)),
            "p50": float(torch.quantile(selected, 0.50)),
            "p90": float(torch.quantile(selected, 0.90)),
        }

    trajectory = {
        "minimum_ball_distance_uu": distribution(minimum_distance),
        "closest_tick": distribution(closest_tick),
        "closest_relative_x_uu": distribution(closest_relative[:, 0]),
        "closest_relative_y_uu": distribution(closest_relative[:, 1]),
        "closest_relative_z_uu": distribution(closest_relative[:, 2]),
        "closest_car_height_uu": distribution(closest_car_height),
        "closest_ball_height_uu": distribution(closest_ball_height),
        "maximum_car_height_uu": distribution(maximum_car_height),
        "maximum_ball_height_uu": distribution(maximum_ball_height),
        "maximum_plan_alignment": distribution(maximum_plan_alignment),
        "boosted_pursuit_ticks": distribution(boosted_pursuit_ticks),
        "offsets_after_pop": {
            str(offset): {
                name: distribution(value, torch.isfinite(value)) for name, value in samples.items()
            }
            for offset, samples in offset_samples.items()
        },
    }
    fractions = {
        key: telemetry[name] / worlds
        for key, name in (
            ("pop_touch", "pop_touches"),
            ("qualified_pop", "qualified_pops"),
            ("launch_near_pop", "launches_near_pop"),
            ("ball_rise_250", "ball_rise_250"),
            ("elevated_follow_touch", "elevated_follow_touches"),
            ("high_follow_touch", "high_follow_touches"),
            ("second_airborne_touch", "second_airborne_touches"),
            ("goal_after_pop", "goals_after_pop"),
        )
    }
    result = {
        "side": side,
        "worlds": worlds,
        "horizon": horizon,
        "seed": seed ^ side,
        "telemetry": telemetry,
        "fractions": fractions,
        "controller": controller.telemetry(),
        "trajectory": trajectory,
        "finite": bool(torch.isfinite(observation).all()),
    }
    del env
    gc.collect()
    torch.cuda.empty_cache()
    return result


def physical_score(rows: list[dict[str, Any]]) -> float:
    weights = {
        "pop_touch": 1.0,
        "qualified_pop": 2.0,
        "launch_near_pop": 1.0,
        "ball_rise_250": 3.0,
        "elevated_follow_touch": 8.0,
        "high_follow_touch": 4.0,
        "second_airborne_touch": 5.0,
        "goal_after_pop": 8.0,
    }
    return float(
        sum(
            weight * sum(row["fractions"][name] for row in rows) / len(rows)
            for name, weight in weights.items()
        )
    )


def candidate_configs(
    quick: bool, *, only_launch_delay: int | None = None
) -> list[GroundToAirConfig]:
    base = GroundToAirConfig()
    if only_launch_delay is not None:
        return [replace(base, launch_delay_ticks=only_launch_delay)]
    if quick:
        return [base]
    values = (
        (8, 8),
        (10, 8),
        (12, 8),
        (12, 10),
        (12, 12),
        (15, 10),
    )
    return [
        replace(
            base,
            first_jump_hold_ticks=hold_ticks,
            jump_release_ticks=release_ticks,
            pop_pitch=0.5,
            boost_during_pop=False,
            carry_ticks_after_second_jump=36,
            carry_pitch=1.0,
            carry_boost=True,
        )
        for hold_ticks, release_ticks in values
    ]


def run(args: argparse.Namespace) -> int:
    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    meshes = WarpArenaMeshes(geometry, args.device)
    calibration: list[dict[str, Any]] = []
    configs = candidate_configs(
        args.quick,
        only_launch_delay=args.only_launch_delay,
    )
    for index, config in enumerate(configs):
        rows = [
            run_one(
                geometry,
                meshes,
                config=config,
                side=side,
                worlds=args.calibration_worlds,
                horizon=args.horizon,
                seed=CALIBRATION_SEED,
                device=args.device,
                collision_dir=args.collision_dir,
            )
            for side in (0, 1)
        ]
        entry = {
            "candidate": index,
            "config": asdict(config),
            "score": physical_score(rows),
            "rows": rows,
        }
        calibration.append(entry)
        print(
            json.dumps(
                {
                    "candidate": index,
                    "candidates": len(configs),
                    "score": entry["score"],
                    "rows": [row["fractions"] for row in rows],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    selected = max(calibration, key=lambda row: (row["score"], -row["candidate"]))
    selected_config = GroundToAirConfig(**selected["config"])
    held_out = [
        run_one(
            geometry,
            meshes,
            config=selected_config,
            side=side,
            worlds=args.test_worlds,
            horizon=args.horizon,
            seed=TEST_SEED,
            device=args.device,
            collision_dir=args.collision_dir,
        )
        for side in (0, 1)
    ]
    result = {
        "format": "RIVAL2_GROUND_TO_AIR_OPTION_V1_CALIBRATION",
        "created_utc": utc_now(),
        "option_version": GROUND_TO_AIR_OPTION_VERSION,
        "source_exact_controller": True,
        "observation_only": True,
        "production_reward_unchanged": True,
        "optimizer_steps": 0,
        "ppo_updates": 0,
        "calibration_seed": CALIBRATION_SEED,
        "untouched_test_seed": TEST_SEED,
        "candidate_count": len(configs),
        "calibration": calibration,
        "selected_candidate": selected["candidate"],
        "selected_config": selected["config"],
        "held_out": held_out,
        "held_out_score": physical_score(held_out),
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--collision-dir", type=Path, default=aerial_v1.DEFAULT_COLLISION_DIR)
    parser.add_argument("--calibration-worlds", type=int, default=128)
    parser.add_argument("--test-worlds", type=int, default=1_024)
    parser.add_argument("--horizon", type=int, default=420)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--only-launch-delay", type=int)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
