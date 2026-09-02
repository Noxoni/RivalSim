"""Read-only trajectory diagnosis for a trained aerial option candidate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_rival2_aerial_option_v1 as runner  # noqa: E402
from rivalsim.arena import ArenaGeometry, WarpArenaMeshes  # noqa: E402
from rivalsim.rival2_aerial_option import (  # noqa: E402
    FIELD,
    PHASE_NAMES,
    AerialRewardTracker,
    build_aerial_scenarios,
)
from rivalsim.rival2_contracts import (  # noqa: E402
    CAR_LINEAR_SPEED_SCALE,
    POSITION_SCALE,
    RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION,
)
from rivalsim.rival2_env import Rival2Env  # noqa: E402
from rivalsim.rival2_policy import deterministic_hybrid_action  # noqa: E402


def _quantiles(value: torch.Tensor) -> dict[str, float]:
    finite = value[torch.isfinite(value)].to(torch.float32)
    if finite.numel() == 0:
        return {"p10": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
    result = torch.quantile(
        finite, torch.tensor((0.1, 0.5, 0.9), device=finite.device)
    )
    return {
        "p10": float(result[0]),
        "p50": float(result[1]),
        "p90": float(result[2]),
        "max": float(finite.max()),
    }


def run(args: argparse.Namespace) -> int:
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = runner.make_model(payload, args.device).eval()
    phase = PHASE_NAMES.index(args.phase)
    phase_authority = runner.load_authority()["physical_curriculum"]["phases"][phase]
    batch = build_aerial_scenarios(
        args.worlds, seed=args.seed, attacker_side=args.side, phase=phase
    )
    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    meshes = WarpArenaMeshes(geometry, args.device)
    env = Rival2Env(
        args.worlds,
        str(args.collision_dir),
        device=args.device,
        seed=args.seed,
        reward_version=RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION,
        initial=batch.state,
        geometry=geometry,
        meshes=meshes,
        car_visitation_order="a_then_b",
    )
    tracker = AerialRewardTracker(args.worlds, attacker_side=args.side, phase=phase)
    active = torch.ones(args.worlds, dtype=torch.bool, device=args.device)
    observation = env.observation
    maximum_height = torch.zeros(args.worlds, device=args.device)
    maximum_upward_speed = torch.full((args.worlds,), -torch.inf, device=args.device)
    minimum_distance = torch.full((args.worlds,), torch.inf, device=args.device)
    closest_car_height = torch.zeros(args.worlds, device=args.device)
    closest_ball_height = torch.zeros(args.worlds, device=args.device)
    first_launch_tick = torch.full((args.worlds,), -1, dtype=torch.int64, device=args.device)
    jump_ticks = torch.zeros(args.worlds, device=args.device)
    boost_ticks = torch.zeros(args.worlds, device=args.device)
    active_ticks = torch.zeros(args.worlds, device=args.device)
    double_jumped = torch.zeros(args.worlds, dtype=torch.bool, device=args.device)
    flipped = torch.zeros(args.worlds, dtype=torch.bool, device=args.device)
    analog_sum = torch.zeros(5, dtype=torch.float64, device=args.device)
    for tick in range(int(phase_authority["horizon_ticks"])):
        active_before = active.clone()
        with torch.no_grad():
            actor, _ = model(observation[:, args.side])
            selected = deterministic_hybrid_action(actor, model.config)
        if args.fast_aerial_macro:
            if tick <= 23:
                selected[:, 0] = 1.0
                selected[:, 2] = -1.0
                selected[:, 5] = 1.0
                selected[:, 6] = 1.0
            elif tick <= 27:
                selected[:, 0] = 1.0
                selected[:, 2] = -1.0
                selected[:, 5] = 0.0
                selected[:, 6] = 1.0
            elif tick == 28:
                selected.zero_()
                selected[:, 0] = 1.0
                selected[:, 5] = 1.0
                selected[:, 6] = 1.0
        action = torch.zeros((args.worlds, 2, 8), device=args.device)
        action[:, args.side] = torch.where(active_before[:, None], selected, 0.0)
        transition = env.step(action)
        after = transition.transition_observation
        scoring_team = env.bridge.views["rival2.scoring_team_latched"].to(torch.int64)
        goal_for = active_before & transition.terminated & (scoring_team == args.side)
        _reward, skill_done = tracker.step(
            observation,
            after,
            tick=tick,
            goal_for_attacker=goal_for,
            active=active_before,
        )
        self_after = after[:, args.side]
        height = self_after[:, FIELD["self.position.z"]] * POSITION_SCALE[2]
        upward = (
            self_after[:, FIELD["self.linear_velocity.z"]]
            * CAR_LINEAR_SPEED_SCALE
        )
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
        closest_car_height.copy_(torch.where(closer, height, closest_car_height))
        ball_height = self_after[:, FIELD["ball.position.z"]] * POSITION_SCALE[2]
        closest_ball_height.copy_(torch.where(closer, ball_height, closest_ball_height))
        maximum_height.copy_(torch.maximum(maximum_height, torch.where(active_before, height, 0.0)))
        maximum_upward_speed.copy_(
            torch.maximum(maximum_upward_speed, torch.where(active_before, upward, -torch.inf))
        )
        launched_now = active_before & (first_launch_tick < 0) & (
            self_after[:, FIELD["self.on_ground"]] < 0.5
        )
        first_launch_tick.copy_(
            torch.where(launched_now, torch.full_like(first_launch_tick, tick), first_launch_tick)
        )
        emitted = transition.emitted_action[:, args.side]
        active_float = active_before.to(torch.float32)
        jump_ticks += active_float * emitted[:, 5]
        boost_ticks += active_float * emitted[:, 6]
        active_ticks += active_float
        analog_sum += (emitted[:, :5] * active_float[:, None]).sum(dim=0, dtype=torch.float64)
        double_jumped |= active_before & (
            self_after[:, FIELD["self.has_double_jumped"]] >= 0.5
        )
        flipped |= active_before & (self_after[:, FIELD["self.has_flipped"]] >= 0.5)
        active &= ~(skill_done | transition.terminated | transition.truncated)
        observation = transition.observation
        if not bool(active.any()):
            break
    launched = first_launch_tick >= 0
    denominator = active_ticks.sum().clamp_min(1.0)
    result = {
        "format": "RIVAL2_AERIAL_OPTION_V1_TRAJECTORY_DIAGNOSTIC",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": runner.v1.sha256_file(args.checkpoint),
        "phase": args.phase,
        "fast_aerial_macro": bool(args.fast_aerial_macro),
        "side": args.side,
        "worlds": args.worlds,
        "physical_events": runner.asdict(tracker.telemetry),
        "launch_tick": _quantiles(first_launch_tick[launched]),
        "maximum_car_height_uu": _quantiles(maximum_height),
        "maximum_upward_speed_uu_per_second": _quantiles(maximum_upward_speed),
        "minimum_ball_distance_uu": _quantiles(minimum_distance),
        "car_height_at_closest_approach_uu": _quantiles(closest_car_height),
        "ball_height_at_closest_approach_uu": _quantiles(closest_ball_height),
        "jump_active_tick_fraction": float(jump_ticks.sum() / denominator),
        "boost_active_tick_fraction": float(boost_ticks.sum() / denominator),
        "mean_analog_action": (analog_sum / denominator).cpu().tolist(),
        "double_jump_world_fraction": float(double_jumped.to(torch.float32).mean()),
        "flip_world_fraction": float(flipped.to(torch.float32).mean()),
    }
    if args.output is not None:
        runner.write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--side", type=int, choices=(0, 1), required=True)
    parser.add_argument("--phase", choices=PHASE_NAMES, default="moving_intercept")
    parser.add_argument("--worlds", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2_026_090_299)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fast-aerial-macro", action="store_true")
    parser.add_argument(
        "--collision-dir",
        type=Path,
        default=runner.DEFAULT_COLLISION_DIR,
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
