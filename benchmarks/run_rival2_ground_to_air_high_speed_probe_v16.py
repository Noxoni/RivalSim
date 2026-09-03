"""Read-only deterministic V3 probe on measured high-speed entry states."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_rival2_capability_curriculum_v1 as capability  # noqa: E402
from benchmarks import run_rival2_ground_to_air_natural_v4 as natural_v4  # noqa: E402
from benchmarks.run_rival2_ground_to_air_entry_probe_v11 import (  # noqa: E402
    human_envelope_config,
)
from rivalsim.arena import ArenaGeometry, WarpArenaMeshes  # noqa: E402
from rivalsim.rival2_contracts import (  # noqa: E402
    RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION,
)
from rivalsim.rival2_env import Rival2Env  # noqa: E402
from rivalsim.rival2_ground_to_air_entry_probe_v11 import (  # noqa: E402
    GroundToAirEntryProbeV11,
)
from rivalsim.rival2_ground_to_air_entry_v11 import (  # noqa: E402
    SETUP_RISING_DOUBLE_JUMP,
)
from rivalsim.rival2_ground_to_air_high_speed_v16 import (  # noqa: E402
    GROUND_TO_AIR_HIGH_SPEED_V16_VERSION,
    build_high_speed_ground_to_air_scenarios,
)
from rivalsim.rival2_policy import deterministic_hybrid_action  # noqa: E402

VERSION = "RIVAL2_GROUND_TO_AIR_HIGH_SPEED_PROBE_V16"
DEFAULT_OUTPUT = ROOT / "results/rival2/ground_to_air_high_speed_v16/baseline.json"


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _distribution(values: Any) -> dict[str, float]:
    tensor = torch.as_tensor(values, dtype=torch.float64)
    quantiles = torch.quantile(
        tensor,
        torch.tensor((0.1, 0.5, 0.9), dtype=tensor.dtype, device=tensor.device),
    )
    return {
        "minimum": float(tensor.min()),
        "p10": float(quantiles[0]),
        "p50": float(quantiles[1]),
        "p90": float(quantiles[2]),
        "maximum": float(tensor.max()),
        "mean": float(tensor.mean()),
    }


def collect_side(
    model: torch.nn.Module,
    defenders: dict[int, torch.nn.Module],
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    *,
    side: int,
    worlds: int,
    horizon: int,
    seed: int,
    device: str,
    collision_dir: Path,
) -> dict[str, Any]:
    batch = build_high_speed_ground_to_air_scenarios(
        worlds,
        seed=seed ^ side,
        attacker_side=side,
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
    probe = GroundToAirEntryProbeV11(
        torch.full(
            (worlds,),
            SETUP_RISING_DOUBLE_JUMP,
            dtype=torch.int64,
            device=device,
        ),
        attacker_side=side,
        envelope_config=human_envelope_config(),
        continuation_ticks=min(horizon, 240),
        separation_ticks=4,
        maximum_contacts=6,
    )
    active = torch.ones(worlds, dtype=torch.bool, device=device)
    goals_for = torch.zeros((), dtype=torch.int64, device=device)
    goals_against = torch.zeros_like(goals_for)
    observation = env.observation
    other = 1 - side
    defender = defenders[other]
    model.eval()
    for tick in range(horizon):
        active_before = active.clone()
        with torch.inference_mode():
            actor, _ = model(observation[:, side])
            learned = deterministic_hybrid_action(actor, model.config)
            defender_actor, _ = defender(observation[:, other])
            defender_action = deterministic_hybrid_action(
                defender_actor, defender.config
            )
        action = torch.zeros((worlds, 2, 8), dtype=torch.float32, device=device)
        action[:, side] = torch.where(active_before[:, None], learned, 0.0)
        action[:, other] = torch.where(
            active_before[:, None], defender_action, 0.0
        )
        transition = env.step(action)
        scoring_team = env.bridge.views["rival2.scoring_team_latched"].to(
            torch.int64
        )
        goal_for = active_before & transition.terminated & (scoring_team == side)
        goal_against = (
            active_before & transition.terminated & (scoring_team == other)
        )
        events = probe.step(
            observation,
            transition.transition_observation,
            tick=tick,
            active=active_before,
            goal_for_attacker=goal_for,
        )
        goals_for += goal_for.sum()
        goals_against += goal_against.sum()
        terminal = (
            transition.terminated
            | transition.truncated
            | events.ball_ground_failure
            | events.contact_budget_exceeded
        )
        active &= ~terminal
        observation = transition.observation
        if not bool(active.any()):
            break
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    telemetry = probe.telemetry()
    result = {
        "side": side,
        "worlds": worlds,
        "horizon": horizon,
        "seed": seed ^ side,
        "direct_policy_control": True,
        "scripted_actions": False,
        "goals_for": int(goals_for.cpu()),
        "goals_against": int(goals_against.cpu()),
        "horizon_timeouts": int(active.sum().cpu()),
        "telemetry": telemetry,
        "initial_state": {
            "planar_gap_uu": _distribution(batch.initial_planar_gap_uu),
            "ball_goalward_speed_uu_per_second": _distribution(
                batch.initial_ball_goalward_speed_uu_per_second
            ),
            "car_goalward_speed_uu_per_second": _distribution(
                batch.initial_car_goalward_speed_uu_per_second
            ),
            "boost_fraction": _distribution(batch.initial_boost_fraction),
            "opponent_ball_distance_uu": _distribution(
                batch.initial_opponent_ball_distance_uu
            ),
        },
        "finite": bool(torch.isfinite(observation).all()),
    }
    del env
    gc.collect()
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    return result


def run(args: argparse.Namespace) -> int:
    checkpoint = args.checkpoint.resolve()
    before = capability.sha256_file(checkpoint)
    expected = args.checkpoint_sha256 or natural_v4.PARENT_SHA256
    if before != expected:
        raise RuntimeError(
            f"high-speed probe checkpoint identity mismatch: {before} != {expected}"
        )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = natural_v4.make_model(payload, args.device).eval().requires_grad_(False)
    defenders = natural_v4.load_defender_policies(args.device)
    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    meshes = WarpArenaMeshes(geometry, args.device)
    rows = [
        collect_side(
            model,
            defenders,
            geometry,
            meshes,
            side=side,
            worlds=args.worlds_per_side,
            horizon=args.horizon,
            seed=args.seed,
            device=args.device,
            collision_dir=args.collision_dir,
        )
        for side in (0, 1)
    ]
    after = capability.sha256_file(checkpoint)
    if after != before:
        raise RuntimeError("high-speed probe mutated the protected V3 scorer")
    summary = {
        name: sum(float(row["telemetry"]["fractions"][name]) for row in rows)
        / len(rows)
        for name in (
            "entry_airborne_contact",
            "second_airborne_contact",
            "goal_within_contact_budget",
            "ball_ground_failure",
        )
    }
    summary["goals_for"] = sum(int(row["goals_for"]) for row in rows)
    summary["goals_against"] = sum(int(row["goals_against"]) for row in rows)
    result = {
        "format": VERSION,
        "created_utc": utc_now(),
        "scenario_identity": GROUND_TO_AIR_HIGH_SPEED_V16_VERSION,
        "checkpoint": {
            "path": checkpoint.as_posix(),
            "sha256": before,
            "unchanged": True,
        },
        "worlds_per_side": args.worlds_per_side,
        "horizon": args.horizon,
        "seed": args.seed,
        "rows": rows,
        "summary": summary,
        "optimizer_steps": 0,
        "reward_changes": 0,
        "verdict": "PASS",
    }
    write_json(args.output, result)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=natural_v4.PARENT)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--collision-dir", type=Path, default=natural_v4.DEFAULT_COLLISION_DIR
    )
    parser.add_argument("--worlds-per-side", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=600)
    parser.add_argument("--seed", type=int, default=2_026_090_322)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
