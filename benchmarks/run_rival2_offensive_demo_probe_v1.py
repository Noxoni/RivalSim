"""Deterministic no-learning probe for productive offensive demolitions."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_rival2_capability_curriculum_v1 as capability  # noqa: E402
from benchmarks import run_rival2_ground_to_air_natural_v4 as natural_v4  # noqa: E402
from rivalsim.arena import ArenaGeometry, WarpArenaMeshes  # noqa: E402
from rivalsim.rival2_contracts import RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION  # noqa: E402
from rivalsim.rival2_env import Rival2Env  # noqa: E402
from rivalsim.rival2_offensive_demo_v1 import (  # noqa: E402
    OFFENSIVE_DEMO_V1_VERSION,
    ROUTE_NAMES,
    OffensiveDemoOutcomeTracker,
    build_offensive_demo_scenarios,
)
from rivalsim.rival2_policy import (  # noqa: E402
    Rival2ActorCritic,
    deterministic_hybrid_action,
)

VERSION = "RIVAL2_OFFENSIVE_DEMO_CALIBRATION_V1"
DEFAULT_OUTPUT = ROOT / "results/rival2/offensive_demo_v1/v23_baseline.json"
DEFAULT_SEED = 2_029_700_000


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_attacker_policies(
    checkpoint: Path | None, device: str
) -> tuple[dict[int, Rival2ActorCritic], list[dict[str, Any]]]:
    if checkpoint is None:
        policies = natural_v4.load_defender_policies(device)
        sources = [
            {
                "side": side,
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": capability.sha256_file(path),
            }
            for side, path in ((0, natural_v4.BLUE), (1, natural_v4.ORANGE))
        ]
        return policies, sources
    resolved = checkpoint.resolve()
    digest = capability.sha256_file(resolved)
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    model = natural_v4.make_model(payload, device)
    return (
        {0: model, 1: model},
        [{"side": "shared", "path": resolved.as_posix(), "sha256": digest}],
    )


def collect_row(
    model: Rival2ActorCritic,
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    *,
    side: int,
    route: int,
    worlds: int,
    horizon: int,
    seed: int,
    device: str,
    collision_dir: Path,
) -> dict[str, Any]:
    batch = build_offensive_demo_scenarios(
        worlds,
        seed=seed ^ side,
        attacker_side=side,
        route=route,
    )
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
    route_tensor = torch.as_tensor(batch.route, dtype=torch.int64, device=device)
    tracker = OffensiveDemoOutcomeTracker(
        route_tensor,
        attacker_side=side,
        followup_window_ticks=3 * 120,
        minimum_goalward_progress_uu=300.0,
    )
    scripted_defender = torch.as_tensor(
        batch.scripted_action, dtype=torch.float32, device=device
    )
    active = torch.ones(worlds, dtype=torch.bool, device=device)
    observation = env.observation
    action_count = torch.zeros((), dtype=torch.float64, device=device)
    analog_sum = torch.zeros(5, dtype=torch.float64, device=device)
    button_sum = torch.zeros(3, dtype=torch.float64, device=device)
    saturation = torch.zeros(5, dtype=torch.float64, device=device)
    closure_sum = torch.zeros((), dtype=torch.float64, device=device)
    closure_ticks = torch.zeros((), dtype=torch.float64, device=device)
    goals_for = torch.zeros((), dtype=torch.int64, device=device)
    goals_against = torch.zeros((), dtype=torch.int64, device=device)
    other = 1 - side
    model.eval()
    for tick in range(horizon):
        active_before = active.clone()
        with torch.no_grad():
            actor, _ = model(observation[:, side])
            learned_action = deterministic_hybrid_action(actor, model.config)
        action = torch.zeros((worlds, 2, 8), dtype=torch.float32, device=device)
        action[:, side] = torch.where(
            active_before[:, None], learned_action, 0.0
        )
        action[:, other] = torch.where(
            active_before[:, None], scripted_defender[:, other], 0.0
        )
        transition = env.step(action)
        scoring_team = env.bridge.views["rival2.scoring_team_latched"].to(
            torch.int64
        )
        goal_for = active_before & transition.terminated & (scoring_team == side)
        goal_against = (
            active_before & transition.terminated & (scoring_team == other)
        )
        events = tracker.step(
            observation,
            transition.transition_observation,
            tick=tick,
            goal_for_attacker=goal_for,
            active=active_before,
        )
        emitted = transition.emitted_action[:, side]
        action_count += active_before.sum(dtype=torch.float64)
        analog_sum += (emitted[:, :5] * active_before[:, None]).sum(
            dim=0, dtype=torch.float64
        )
        button_sum += (emitted[:, 5:] * active_before[:, None]).sum(
            dim=0, dtype=torch.float64
        )
        saturation += (
            (emitted[:, :5].abs() > 0.95) & active_before[:, None]
        ).sum(dim=0, dtype=torch.float64)
        closure_sum += events.opponent_distance_gain_uu.sum(dtype=torch.float64)
        closure_ticks += (
            events.opponent_distance_gain_uu > 0.0
        ).sum(dtype=torch.float64)
        goals_for += goal_for.sum()
        goals_against += goal_against.sum()
        terminal = (
            transition.terminated
            | transition.truncated
            | events.expired_without_conversion
        )
        active &= ~terminal
        observation = transition.observation
        if not bool(active.any()):
            break
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()
    converted = tracker.touch_paid | tracker.progress_paid | tracker.goal_paid
    telemetry = asdict(tracker.telemetry)
    fractions = {
        "actual_demo": float(tracker.demo_seen.float().mean().cpu()),
        "offensive_context_demo": float(
            (tracker.demo_tick > -10_000).float().mean().cpu()
        ),
        "post_demo_touch": float(tracker.touch_paid.float().mean().cpu()),
        "post_demo_goalward_progress": float(
            tracker.progress_paid.float().mean().cpu()
        ),
        "post_demo_goal": float(tracker.goal_paid.float().mean().cpu()),
        "productive_conversion": float(converted.float().mean().cpu()),
        "expired_without_conversion": float(
            tracker.expiry_counted.float().mean().cpu()
        ),
    }
    result = {
        "route": ROUTE_NAMES[route],
        "route_id": route,
        "side": side,
        "worlds": worlds,
        "horizon": horizon,
        "seed": seed ^ side,
        "direct_policy_control": True,
        "defender_control": "constant throttle 0.35; no steering",
        "telemetry": telemetry,
        "fractions": fractions,
        "goals_for": int(goals_for.cpu()),
        "goals_against": int(goals_against.cpu()),
        "horizon_timeouts": int(active.sum().cpu()),
        "mean_offensive_context_closure_uu_per_active_tick": float(
            (closure_sum / action_count.clamp_min(1.0)).cpu()
        ),
        "offensive_context_closure_tick_fraction": float(
            (closure_ticks / action_count.clamp_min(1.0)).cpu()
        ),
        "action_ticks": int(action_count.cpu()),
        "mean_analog_action": (
            analog_sum / action_count.clamp_min(1.0)
        ).cpu().tolist(),
        "button_fraction": (
            button_sum / action_count.clamp_min(1.0)
        ).cpu().tolist(),
        "analog_saturation_fraction": (
            saturation / action_count.clamp_min(1.0)
        ).cpu().tolist(),
        "finite": bool(torch.isfinite(observation).all()),
    }
    del env
    gc.collect()
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    return result


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("offensive-demo calibration has no rows")
    result: dict[str, Any] = {}
    for route in ROUTE_NAMES:
        grouped = [row for row in rows if row["route"] == route]
        if not grouped:
            raise ValueError(f"offensive-demo calibration is missing route {route}")
        route_summary = {
            name: sum(float(row["fractions"][name]) for row in grouped)
            / len(grouped)
            for name in (
                "actual_demo",
                "offensive_context_demo",
                "post_demo_touch",
                "post_demo_goalward_progress",
                "post_demo_goal",
                "productive_conversion",
                "expired_without_conversion",
            )
        }
        route_summary["rows"] = len(grouped)
        route_summary["mean_offensive_context_closure_uu_per_active_tick"] = sum(
            float(row["mean_offensive_context_closure_uu_per_active_tick"])
            for row in grouped
        ) / len(grouped)
        result[route] = route_summary
    return result


def run(args: argparse.Namespace) -> int:
    if args.worlds_per_row <= 0 or args.horizon <= 0:
        raise ValueError("worlds and horizon must be positive")
    checkpoint = args.checkpoint.resolve() if args.checkpoint else None
    policies, sources = load_attacker_policies(checkpoint, args.device)
    hashes_before = {source["path"]: source["sha256"] for source in sources}
    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    meshes = WarpArenaMeshes(geometry, args.device)
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for route in range(len(ROUTE_NAMES)):
            for side in (0, 1):
                rows.append(
                    collect_row(
                        policies[side],
                        geometry,
                        meshes,
                        side=side,
                        route=route,
                        worlds=args.worlds_per_row,
                        horizon=args.horizon,
                        seed=args.seed + route * 100_000,
                        device=args.device,
                        collision_dir=args.collision_dir,
                    )
                )
    for source in sources:
        path = Path(source["path"])
        if not path.is_absolute():
            path = ROOT / path
        if capability.sha256_file(path) != hashes_before[source["path"]]:
            raise RuntimeError("offensive-demo calibration mutated a checkpoint")
    payload = {
        "format": VERSION,
        "created_utc": utc_now(),
        "scenario_and_tracker_identity": OFFENSIVE_DEMO_V1_VERSION,
        "checkpoints": sources,
        "worlds_per_row": args.worlds_per_row,
        "horizon": args.horizon,
        "seed": args.seed,
        "rows": rows,
        "summary": summarize(rows),
        "optimizer_steps": 0,
        "reward_used_for_selection": False,
        "state_mutation_beyond_normal_simulation": False,
        "attacker_action_injection": False,
        "verdict": "PASS",
    }
    write_json(args.output, payload)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True), flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--collision-dir", type=Path, default=natural_v4.DEFAULT_COLLISION_DIR
    )
    parser.add_argument("--worlds-per-row", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=600)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
