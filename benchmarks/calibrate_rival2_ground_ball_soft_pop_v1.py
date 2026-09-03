"""Explore and then freeze a catchable, low-separation ground-ball pop."""

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

from benchmarks import run_rival2_codex_autonomous_v1 as base  # noqa: E402
from rivalsim.arena import ArenaGeometry, WarpArenaMeshes  # noqa: E402
from rivalsim.rival2_aerial_option import FIELD  # noqa: E402
from rivalsim.rival2_contracts import POSITION_SCALE  # noqa: E402
from rivalsim.rival2_env import Rival2Env  # noqa: E402
from rivalsim.rival2_ground_ball_pop import (  # noqa: E402
    PrecontactPopConfig,
    PrecontactPopController,
)
from rivalsim.rival2_ground_ball_soft_pop import (  # noqa: E402
    build_ground_ball_soft_pop_scenarios,
)
from rivalsim.rival2_policy import (  # noqa: E402
    Rival2ActorCritic,
    Rival2PolicyConfig,
    deterministic_hybrid_action,
)

OPTION = ROOT / "checkpoints/rival2/ground_to_air_goal_v3/rival2_ground_to_air_goal_v3.pt"
OUTPUT = ROOT / "results/rival2/ground_ball_soft_pop_v1/exploration.json"


def candidates() -> tuple[PrecontactPopConfig, ...]:
    return tuple(
        PrecontactPopConfig(
            trigger_distance_uu=trigger,
            first_jump_hold_ticks=8,
            jump_release_ticks=6,
            second_jump=second_jump,
            pitch=pitch,
            use_approach_boost=False,
        )
        for trigger in (145.0, 165.0, 185.0, 205.0)
        for second_jump in (False, True)
        for pitch in (-1.0, 0.0, 1.0)
    )


def evaluate(
    model: Rival2ActorCritic,
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    *,
    config: PrecontactPopConfig,
    side: int,
    worlds: int,
    horizon: int,
    seed: int,
    device: str,
    collision_root: Path,
) -> dict[str, Any]:
    initial = build_ground_ball_soft_pop_scenarios(worlds, seed=seed ^ side, attacker_side=side)
    env = Rival2Env(
        worlds,
        str(collision_root),
        device=device,
        seed=seed ^ side,
        initial=initial,
        geometry=geometry,
        meshes=meshes,
        car_visitation_order="a_then_b",
    )
    controller = PrecontactPopController(worlds, device=device, config=config)
    observation = env.observation
    active = torch.ones(worlds, dtype=torch.bool, device=device)
    launched = torch.zeros_like(active)
    first_touch = torch.zeros_like(active)
    second_touch = torch.zeros_like(active)
    close_follow = torch.zeros_like(active)
    elevated_follow = torch.zeros_like(active)
    first_touch_tick = torch.full((worlds,), -10_000, dtype=torch.int64, device=device)
    maximum_ball_height = torch.full((worlds,), 92.75, dtype=torch.float32, device=device)
    minimum_follow_distance = torch.full((worlds,), torch.inf, dtype=torch.float32, device=device)
    for tick in range(horizon):
        active_before = active.clone()
        with torch.inference_mode():
            actor, _ = model(observation[:, side])
            learned = deterministic_hybrid_action(actor)
        step = controller.step(learned, observation[:, side])
        launched |= active_before & step.launch_started
        action = torch.zeros((worlds, 2, 8), dtype=torch.float32, device=device)
        action[:, side] = torch.where(active_before[:, None], step.action, 0.0)
        transition = env.step(action)
        after = transition.transition_observation
        terminal = transition.terminated | transition.truncated
        touch = (
            active_before & ~terminal & (after[:, side, FIELD["lifecycle.self_touch_event"]] >= 0.5)
        )
        new_first = touch & ~first_touch
        new_second = touch & first_touch & ~second_touch
        first_touch |= touch
        second_touch |= new_second
        first_touch_tick = torch.where(
            new_first, torch.full_like(first_touch_tick, tick), first_touch_tick
        )
        scale = torch.as_tensor(POSITION_SCALE, dtype=after.dtype, device=after.device)
        relative = (
            torch.stack(
                [after[:, side, FIELD[f"relative.ball_position.{axis}"]] for axis in "xyz"],
                dim=-1,
            )
            * scale
        )
        distance = torch.linalg.vector_norm(relative, dim=-1)
        car_height = after[:, side, FIELD["self.position.z"]] * POSITION_SCALE[2]
        ball_height = after[:, side, FIELD["ball.position.z"]] * POSITION_SCALE[2]
        follow_window = (
            active_before
            & ~terminal
            & first_touch
            & ((tick - first_touch_tick) >= 4)
            & ((tick - first_touch_tick) <= 100)
            & (ball_height >= 180.0)
        )
        minimum_follow_distance = torch.where(
            follow_window,
            torch.minimum(minimum_follow_distance, distance),
            minimum_follow_distance,
        )
        close_follow |= follow_window & (distance <= 180.0) & (car_height >= 80.0)
        elevated_follow |= (
            new_second & (distance <= 180.0) & (car_height >= 120.0) & (ball_height >= 220.0)
        )
        maximum_ball_height = torch.where(
            active_before & ~terminal,
            torch.maximum(maximum_ball_height, ball_height),
            maximum_ball_height,
        )
        active &= ~terminal
        observation = transition.observation
        if not bool(active.any()):
            break
    finite_distance = minimum_follow_distance[torch.isfinite(minimum_follow_distance)]
    result = {
        "side": side,
        "worlds": worlds,
        "config": asdict(config),
        "launch_fraction": float(launched.float().mean()),
        "first_touch_fraction": float(first_touch.float().mean()),
        "second_touch_fraction": float(second_touch.float().mean()),
        "ball_rise_180_fraction": float((maximum_ball_height >= 180.0).float().mean()),
        "ball_rise_220_fraction": float((maximum_ball_height >= 220.0).float().mean()),
        "ball_rise_250_fraction": float((maximum_ball_height >= 250.0).float().mean()),
        "close_follow_fraction": float(close_follow.float().mean()),
        "elevated_follow_fraction": float(elevated_follow.float().mean()),
        "minimum_follow_distance_uu": {
            "p50": (
                float(torch.quantile(finite_distance, 0.5)) if finite_distance.numel() else None
            ),
            "p90": (
                float(torch.quantile(finite_distance, 0.9)) if finite_distance.numel() else None
            ),
        },
        "finite": bool(torch.isfinite(observation).all()),
    }
    del env
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run(args: argparse.Namespace) -> int:
    payload = torch.load(OPTION, map_location="cpu", weights_only=False)
    model = Rival2ActorCritic(Rival2PolicyConfig(**payload["policy_config"])).to(args.device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    geometry = ArenaGeometry.load_soccar(args.collision_root)
    meshes = WarpArenaMeshes(geometry, args.device)
    rows: list[dict[str, Any]] = []
    for index, config in enumerate(candidates()):
        sides = [
            evaluate(
                model,
                geometry,
                meshes,
                config=config,
                side=side,
                worlds=args.worlds_per_side,
                horizon=args.horizon,
                seed=args.seed,
                device=args.device,
                collision_root=args.collision_root,
            )
            for side in (0, 1)
        ]
        score = (
            4.0 * min(row["elevated_follow_fraction"] for row in sides)
            + 2.0 * min(row["close_follow_fraction"] for row in sides)
            + min(row["ball_rise_220_fraction"] for row in sides)
            + 0.25 * min(row["ball_rise_180_fraction"] for row in sides)
        )
        row = {"candidate": index, "score": score, "sides": sides}
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    selected = max(rows, key=lambda row: row["score"])
    result = {
        "format": "RIVAL2_GROUND_BALL_SOFT_POP_V1_EXPLORATION",
        "option_checkpoint_sha256": base.sha256_file(OPTION),
        "worlds_per_side": args.worlds_per_side,
        "horizon": args.horizon,
        "seed": args.seed,
        "candidates": rows,
        "selected": selected,
        "optimizer_steps": 0,
        "reward_changes": 0,
    }
    base.write_json(OUTPUT, result)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--collision-root",
        type=Path,
        default=Path(r"G:\dev\RLBot-Rival\bot\collision_meshes\soccar"),
    )
    parser.add_argument("--worlds-per-side", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=240)
    parser.add_argument("--seed", type=int, default=2026092601)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
