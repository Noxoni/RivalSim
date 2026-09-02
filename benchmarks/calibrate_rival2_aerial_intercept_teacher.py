"""Read-only physical calibration of the ballistic aerial intercept teacher."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_rival2_aerial_option_v1 as v1  # noqa: E402
from benchmarks import run_rival2_aerial_training_pack_v1 as pack_runner  # noqa: E402
from rivalsim.arena import ArenaGeometry, WarpArenaMeshes  # noqa: E402
from rivalsim.rival2_aerial_intercept_teacher import plan_aerial_intercept  # noqa: E402
from rivalsim.rival2_aerial_option import FIELD  # noqa: E402
from rivalsim.rival2_aerial_option_v2 import apply_fast_aerial_initiation  # noqa: E402
from rivalsim.rival2_aerial_training_pack import (  # noqa: E402
    PACK_NAMES,
    AerialTrainingPackTracker,
    build_training_pack_scenarios,
)
from rivalsim.rival2_contracts import POSITION_SCALE, RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION  # noqa: E402
from rivalsim.rival2_env import Rival2Env  # noqa: E402

OUTPUT = ROOT / "results/rival2/aerial_intercept_teacher/calibration.json"
SEED = 2_026_090_311


def _quantiles(value: torch.Tensor) -> dict[str, float]:
    finite = value[torch.isfinite(value)].to(torch.float32)
    if finite.numel() == 0:
        return {"p10": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
    q = torch.quantile(finite, torch.tensor((0.1, 0.5, 0.9), device=value.device))
    return {
        "p10": float(q[0]),
        "p50": float(q[1]),
        "p90": float(q[2]),
        "max": float(finite.max()),
    }


def run_one(
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    *,
    pack: int,
    side: int,
    worlds: int,
    device: str,
    collision_dir: Path,
    horizon: int,
    deadline: int,
) -> dict[str, Any]:
    seed = SEED ^ (pack * 16 + side)
    batch = build_training_pack_scenarios(
        worlds, seed=seed, attacker_side=side, pack=pack
    )
    env = Rival2Env(
        worlds,
        str(collision_dir),
        device=device,
        seed=seed,
        reward_version=RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION,
        initial=batch.state,
        geometry=geometry,
        meshes=meshes,
        car_visitation_order="a_then_b",
    )
    tracker = AerialTrainingPackTracker(
        worlds,
        attacker_side=side,
        pack=pack,
        first_touch_deadline=deadline,
        horizon=horizon,
    )
    active = torch.ones(worlds, dtype=torch.bool, device=device)
    option_age = torch.zeros(worlds, dtype=torch.int64, device=device)
    observation = env.observation
    minimum_distance = torch.full((worlds,), torch.inf, device=device)
    closest_car_height = torch.zeros(worlds, device=device)
    closest_ball_height = torch.zeros(worlds, device=device)
    post_ticks = torch.zeros((), dtype=torch.float64, device=device)
    boost_ticks = torch.zeros((), dtype=torch.float64, device=device)
    alignment_sum = torch.zeros((), dtype=torch.float64, device=device)
    intercept_time_sum = torch.zeros((), dtype=torch.float64, device=device)
    analog_sum = torch.zeros(5, dtype=torch.float64, device=device)
    for tick in range(horizon):
        active_before = active.clone()
        plan = plan_aerial_intercept(observation[:, side])
        selected, primitive = apply_fast_aerial_initiation(
            plan.action, option_age, active_before
        )
        action = torch.zeros((worlds, 2, 8), device=device)
        action[:, side] = torch.where(active_before[:, None], selected, 0.0)
        transition = env.step(action)
        after = transition.transition_observation
        scoring = env.bridge.views["rival2.scoring_team_latched"].to(torch.int64)
        any_goal = active_before & transition.terminated & (scoring >= 0)
        goal_for = any_goal & (scoring == side)
        _reward, done = tracker.step(
            observation,
            after,
            tick=tick,
            goal_for_attacker=goal_for,
            any_goal=any_goal,
            active=active_before,
        )
        self_after = after[:, side]
        relative = torch.stack(
            [
                self_after[:, FIELD[f"relative.ball_position.{axis}"]]
                * POSITION_SCALE[index]
                for index, axis in enumerate("xyz")
            ],
            dim=-1,
        )
        distance = torch.linalg.vector_norm(relative, dim=-1)
        closer = active_before & (distance < minimum_distance)
        minimum_distance.copy_(torch.where(closer, distance, minimum_distance))
        car_height = self_after[:, FIELD["self.position.z"]] * POSITION_SCALE[2]
        ball_height = self_after[:, FIELD["ball.position.z"]] * POSITION_SCALE[2]
        closest_car_height.copy_(torch.where(closer, car_height, closest_car_height))
        closest_ball_height.copy_(torch.where(closer, ball_height, closest_ball_height))
        post = active_before & ~primitive
        post_float = post.to(torch.float32)
        emitted = transition.emitted_action[:, side]
        post_ticks += post.sum(dtype=torch.float64)
        boost_ticks += (emitted[:, 6] * post_float).sum(dtype=torch.float64)
        alignment_sum += (plan.nose_alignment * post_float).sum(dtype=torch.float64)
        intercept_time_sum += (plan.intercept_time * post_float).sum(dtype=torch.float64)
        analog_sum += (emitted[:, :5] * post_float[:, None]).sum(dim=0, dtype=torch.float64)
        active &= ~(done | transition.terminated | transition.truncated)
        option_age += active_before.to(torch.int64)
        observation = transition.observation
        if not bool(active.any()):
            break
    telemetry = asdict(tracker.telemetry)
    denominator = post_ticks.clamp_min(1.0)
    result = {
        "pack": PACK_NAMES[pack],
        "side": side,
        "worlds": worlds,
        "telemetry": telemetry,
        "fractions": {
            "launch": telemetry["launches"] / worlds,
            "high_touch": telemetry["first_high_touches"] / worlds,
            "goalward_first_touch": telemetry["goalward_first_touches"] / worlds,
            "second_airborne_touch": telemetry["second_airborne_touches"] / worlds,
            "qualified_goal": telemetry["goals"] / worlds,
            "ball_ground_failure": telemetry["ball_ground_failures"] / worlds,
            "missed_intercept_failure": telemetry["missed_intercept_failures"] / worlds,
        },
        "minimum_ball_distance_uu": _quantiles(minimum_distance),
        "car_height_at_closest_approach_uu": _quantiles(closest_car_height),
        "ball_height_at_closest_approach_uu": _quantiles(closest_ball_height),
        "post_primitive_boost_fraction": float(boost_ticks / denominator),
        "post_primitive_mean_nose_alignment": float(alignment_sum / denominator),
        "post_primitive_mean_intercept_time_seconds": float(
            intercept_time_sum / denominator
        ),
        "post_primitive_mean_analog_action": (analog_sum / denominator).cpu().tolist(),
        "finite": bool(torch.isfinite(observation).all()),
    }
    del env
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run(args: argparse.Namespace) -> int:
    authority = pack_runner.load_authority()
    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    meshes = WarpArenaMeshes(geometry, args.device)
    rows = []
    for pack, row in enumerate(authority["physical_training"]["packs"]):
        for side in (0, 1):
            rows.append(
                run_one(
                    geometry,
                    meshes,
                    pack=pack,
                    side=side,
                    worlds=args.worlds,
                    device=args.device,
                    collision_dir=args.collision_dir,
                    horizon=int(row["horizon_ticks"]),
                    deadline=int(row["first_high_touch_deadline_tick"]),
                )
            )
    output = {
        "format": "RIVAL2_AERIAL_INTERCEPT_TEACHER_CALIBRATION",
        "created_utc": v1.utc_now(),
        "observation_only": True,
        "optimizer_steps": 0,
        "rows": rows,
    }
    v1.write_json(args.output, output)
    print(json.dumps(output, indent=2, sort_keys=True), flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--collision-dir", type=Path, default=v1.DEFAULT_COLLISION_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
