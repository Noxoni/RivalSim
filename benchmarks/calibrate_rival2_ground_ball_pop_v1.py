"""Search a bounded source controller for literal resting-ball pop outcomes."""

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
    build_ground_ball_pop_scenarios,
)
from rivalsim.rival2_policy import (  # noqa: E402
    Rival2ActorCritic,
    Rival2PolicyConfig,
    deterministic_hybrid_action,
)

OPTION = ROOT / "checkpoints/rival2/ground_to_air_goal_v3/rival2_ground_to_air_goal_v3.pt"
AUTHORITY = ROOT / "results/rival2/ground_ball_pop_v1/authority.json"
AUTHORITY_SHA256 = "03B5F83A448818B33F432B8630CDFE92357A1A757F3D66667318F8D0CD5C0DEB"
OUTPUT = ROOT / "results/rival2/ground_ball_pop_v1/calibration.json"


def candidates() -> tuple[PrecontactPopConfig, ...]:
    return tuple(
        PrecontactPopConfig(
            trigger_distance_uu=trigger,
            first_jump_hold_ticks=8,
            jump_release_ticks=6,
            second_jump=True,
            pitch=pitch,
            use_approach_boost=boost,
        )
        for trigger in (175.0, 185.0, 195.0, 205.0)
        for pitch in (-1.0, 0.0, 1.0)
        for boost in (True,)
    )


def load_authority(args: argparse.Namespace) -> dict[str, Any]:
    if base.sha256_file(AUTHORITY) != AUTHORITY_SHA256:
        raise RuntimeError("ground-ball pop calibration authority changed")
    authority = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    if authority.get("format") != "RIVAL2_GROUND_BALL_POP_V1_CALIBRATION_AUTHORITY":
        raise RuntimeError("unexpected ground-ball pop calibration authority")
    frozen = authority["execution"]
    observed = {
        "worlds_per_side": args.worlds_per_side,
        "horizon": args.horizon,
        "seed": args.seed,
    }
    if observed != frozen:
        raise RuntimeError(f"calibration execution differs from authority: {observed}")
    if [asdict(config) for config in candidates()] != authority["candidates"]:
        raise RuntimeError("calibration candidate grid differs from authority")
    for identity in authority["bound_inputs"].values():
        if base.sha256_file(ROOT / identity["path"]) != identity["sha256"]:
            raise RuntimeError(f"calibration input changed: {identity['path']}")
    return authority


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
    initial = build_ground_ball_pop_scenarios(worlds, seed=seed ^ side, attacker_side=side)
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
    touched = torch.zeros_like(active)
    elevated_touch = torch.zeros_like(active)
    maximum_ball_height = torch.full((worlds,), 92.75, dtype=torch.float32, device=device)
    goal_after_touch = torch.zeros_like(active)
    for _tick in range(horizon):
        with torch.inference_mode():
            actor, _ = model(observation[:, side])
            learned = deterministic_hybrid_action(actor)
        step = controller.step(learned, observation[:, side])
        launched |= step.launch_started
        action = torch.zeros((worlds, 2, 8), dtype=torch.float32, device=device)
        action[:, side] = torch.where(active[:, None], step.action, 0.0)
        transition = env.step(action)
        after = transition.observation
        touch = active & (after[:, side, FIELD["lifecycle.self_touch_event"]] >= 0.5)
        car_height = after[:, side, FIELD["self.position.z"]] * POSITION_SCALE[2]
        ball_height = after[:, side, FIELD["ball.position.z"]] * POSITION_SCALE[2]
        touched_before = touched.clone()
        touched |= touch
        elevated_touch |= touch & touched_before & (car_height >= 100.0) & (ball_height >= 250.0)
        maximum_ball_height = torch.maximum(maximum_ball_height, ball_height)
        scoring_team = env.bridge.views["rival2.scoring_team_latched"].to(torch.int64)
        goal_after_touch |= transition.terminated & touched & (scoring_team == side)
        active &= ~(transition.terminated | transition.truncated)
        observation = transition.observation
        if not bool(active.any()):
            break
    result = {
        "side": side,
        "worlds": worlds,
        "config": asdict(config),
        "launch_fraction": float(launched.float().mean()),
        "touch_fraction": float(touched.float().mean()),
        "ball_rise_180_fraction": float((maximum_ball_height >= 180.0).float().mean()),
        "ball_rise_250_fraction": float((maximum_ball_height >= 250.0).float().mean()),
        "ball_rise_400_fraction": float((maximum_ball_height >= 400.0).float().mean()),
        "elevated_follow_touch_fraction": float(elevated_touch.float().mean()),
        "goal_after_touch_fraction": float(goal_after_touch.float().mean()),
        "maximum_ball_height_uu": {
            "p50": float(torch.quantile(maximum_ball_height, 0.5)),
            "p90": float(torch.quantile(maximum_ball_height, 0.9)),
            "maximum": float(maximum_ball_height.max()),
        },
        "finite": bool(torch.isfinite(observation).all()),
    }
    del env
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run(args: argparse.Namespace) -> int:
    authority = load_authority(args)
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
                collision_root=Path(args.collision_root),
            )
            for side in (0, 1)
        ]
        score = (
            2.0 * min(row["ball_rise_250_fraction"] for row in sides)
            + 4.0 * min(row["elevated_follow_touch_fraction"] for row in sides)
            + 0.25 * min(row["ball_rise_180_fraction"] for row in sides)
        )
        row = {"candidate": index, "score": score, "sides": sides}
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    selected = max(rows, key=lambda row: row["score"])
    result = {
        "format": "RIVAL2_GROUND_BALL_POP_V1_CALIBRATION",
        "authority_sha256": AUTHORITY_SHA256,
        "option_checkpoint": {
            "path": OPTION.relative_to(ROOT).as_posix(),
            "sha256": base.sha256_file(OPTION),
        },
        "worlds_per_side": args.worlds_per_side,
        "horizon": args.horizon,
        "seed": args.seed,
        "candidates": rows,
        "selected_candidate": selected["candidate"],
        "selected": selected,
        "optimizer_steps": 0,
        "reward_changes": 0,
        "acceptance": authority["acceptance"],
    }
    base.write_json(OUTPUT, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--collision-root",
        type=Path,
        default=Path(r"G:\dev\RLBot-Rival\bot\collision_meshes\soccar"),
    )
    parser.add_argument("--worlds-per-side", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2026092101)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
