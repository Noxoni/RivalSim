"""Deterministic, read-only calibration of the V11 aerial-entry feeds."""

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
from rivalsim.rival2_ground_to_air_entry_probe_v11 import (  # noqa: E402
    GROUND_TO_AIR_ENTRY_PROBE_V11_VERSION,
    GroundToAirEntryProbeV11,
)
from rivalsim.rival2_ground_to_air_entry_v11 import (  # noqa: E402
    DEFENDER_LIVE,
    DEFENDER_PARKED,
    GROUND_TO_AIR_ENTRY_V11_VERSION,
    SETUP_NAMES,
    build_ground_to_air_entry_scenarios,
)
from rivalsim.rival2_ground_to_air_human_bridge_v11 import (  # noqa: E402
    HumanAerialEnvelopeConfig,
)
from rivalsim.rival2_policy import (  # noqa: E402
    Rival2ActorCritic,
    deterministic_hybrid_action,
)

VERSION = "RIVAL2_GROUND_TO_AIR_ENTRY_CALIBRATION_V11"
DEFAULT_OUTPUT = ROOT / "results/rival2/ground_to_air_natural_v11/entry_parent.json"
HUMAN_REFERENCE = (
    ROOT / "results/rival2/ground_to_air_human_physics_v1/human_transition.json"
)
DEFAULT_SEED = 2_029_600_000


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


def human_envelope_config() -> HumanAerialEnvelopeConfig:
    """Return the source-measured V11 event envelope.

    Targets are medians from the 35 accepted ground-to-air attempts.  The
    strict event requires median car/ball height, source-p10 upward speed, and
    source-p90 distance.  This runner records the values; it does not turn
    them into reward.
    """

    return HumanAerialEnvelopeConfig(
        target_car_height_uu=141.16314697265625,
        target_ball_height_uu=273.7956237792969,
        target_car_vertical_speed_uu_per_second=443.40289306640625,
        target_distance_uu=140.35697369650788,
        target_vertical_standoff_uu=132.3664093017578,
        distance_tolerance_uu=40.0,
        vertical_standoff_tolerance_uu=60.0,
        minimum_event_car_height_uu=141.0,
        minimum_event_ball_height_uu=274.0,
        minimum_event_car_vertical_speed_uu_per_second=265.0,
        maximum_event_distance_uu=157.0,
        maximum_bridge_ticks=180,
        car_height_weight=1.0,
        ball_height_weight=1.0,
        car_vertical_speed_weight=2.0,
        distance_weight=1.0,
        vertical_standoff_weight=1.0,
    )


def collect_row(
    model: Rival2ActorCritic,
    defenders: dict[int, Rival2ActorCritic],
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    *,
    side: int,
    setup: int,
    defender_mode: str,
    difficulty: float,
    worlds: int,
    horizon: int,
    seed: int,
    device: str,
    collision_dir: Path,
) -> dict[str, Any]:
    """Run one direct-policy row without reward or optimizer use."""

    batch = build_ground_to_air_entry_scenarios(
        worlds,
        seed=seed ^ side,
        attacker_side=side,
        setup=setup,
        difficulty=difficulty,
        defender_mode=defender_mode,
        attacker_boost_range=(35.0, 100.0),
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
    setup_tensor = torch.as_tensor(
        batch.setup, dtype=torch.int64, device=device
    )
    probe = GroundToAirEntryProbeV11(
        setup_tensor,
        attacker_side=side,
        envelope_config=human_envelope_config(),
        continuation_ticks=min(horizon, 240),
        separation_ticks=4,
        maximum_contacts=6,
    )
    defender_active = torch.as_tensor(
        batch.defender_active, dtype=torch.bool, device=device
    )
    active = torch.ones(worlds, dtype=torch.bool, device=device)
    observation = env.observation
    action_count = torch.zeros((), dtype=torch.float64, device=device)
    analog_sum = torch.zeros(5, dtype=torch.float64, device=device)
    button_sum = torch.zeros(3, dtype=torch.float64, device=device)
    saturation = torch.zeros(5, dtype=torch.float64, device=device)
    goals_for = torch.zeros((), dtype=torch.int64, device=device)
    goals_against = torch.zeros((), dtype=torch.int64, device=device)
    ball_ground_failures = torch.zeros((), dtype=torch.int64, device=device)
    contact_budget_failures = torch.zeros((), dtype=torch.int64, device=device)
    other = 1 - side
    defender = defenders[other]
    model.eval()
    for tick in range(horizon):
        active_before = active.clone()
        with torch.no_grad():
            actor, _ = model(observation[:, side])
            learned_action = deterministic_hybrid_action(actor, model.config)
            defender_actor, _ = defender(observation[:, other])
            defender_action = deterministic_hybrid_action(
                defender_actor, defender.config
            )
        action = torch.zeros((worlds, 2, 8), dtype=torch.float32, device=device)
        action[:, side] = torch.where(
            active_before[:, None], learned_action, 0.0
        )
        live_defender = active_before & defender_active
        action[:, other] = torch.where(
            live_defender[:, None], defender_action, 0.0
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
        goals_for += goal_for.sum()
        goals_against += goal_against.sum()
        ball_ground_failures += events.ball_ground_failure.sum()
        contact_budget_failures += events.contact_budget_exceeded.sum()
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
        torch.cuda.synchronize()
    telemetry = probe.telemetry()
    result = {
        "setup": SETUP_NAMES[setup],
        "setup_id": setup,
        "side": side,
        "defender_mode": defender_mode,
        "difficulty": difficulty,
        "worlds": worlds,
        "horizon": horizon,
        "seed": seed ^ side,
        "direct_policy_control": True,
        "scripted_entry_actions": False,
        "telemetry": telemetry,
        "goals_for": int(goals_for.cpu()),
        "goals_against": int(goals_against.cpu()),
        "ball_ground_failures": int(ball_ground_failures.cpu()),
        "contact_budget_failures": int(contact_budget_failures.cpu()),
        "horizon_timeouts": int(active.sum().cpu()),
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
        raise ValueError("V11 calibration has no rows")
    summary: dict[str, Any] = {}
    for difficulty in sorted({float(row["difficulty"]) for row in rows}):
        difficulty_rows = [
            row for row in rows if float(row["difficulty"]) == difficulty
        ]
        difficulty_summary: dict[str, Any] = {}
        for setup in SETUP_NAMES:
            grouped = [row for row in difficulty_rows if row["setup"] == setup]
            if not grouped:
                raise ValueError(f"V11 calibration is missing setup {setup}")
            fractions = [row["telemetry"]["fractions"] for row in grouped]
            setup_summary = {
                name: sum(float(item[name]) for item in fractions) / len(fractions)
                for name in (
                    "first_contact",
                    "entry_airborne_contact",
                    "human_envelope_reached",
                    "second_airborne_contact",
                    "goal_within_contact_budget",
                    "contact_budget_exceeded",
                    "ball_ground_failure",
                )
            }
            setup_summary["rows"] = len(grouped)
            difficulty_summary[setup] = setup_summary
        summary[f"difficulty_{difficulty:.3f}"] = difficulty_summary
    return summary


def run(args: argparse.Namespace) -> int:
    if args.worlds_per_row <= 0 or args.horizon <= 0:
        raise ValueError("worlds and horizon must be positive")
    if not args.difficulty or any(
        not 0.0 <= value <= 1.0 for value in args.difficulty
    ):
        raise ValueError("difficulty values must be inside [0,1]")
    checkpoint = args.checkpoint.resolve()
    checkpoint_hash_before = capability.sha256_file(checkpoint)
    source = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = natural_v4.make_model(source, args.device)
    defenders = natural_v4.load_defender_policies(args.device)
    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    meshes = WarpArenaMeshes(geometry, args.device)
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for difficulty_index, difficulty in enumerate(args.difficulty):
            for setup in range(len(SETUP_NAMES)):
                for defender_index, defender_mode in enumerate(
                    (DEFENDER_PARKED, DEFENDER_LIVE)
                ):
                    for side in (0, 1):
                        rows.append(
                            collect_row(
                                model,
                                defenders,
                                geometry,
                                meshes,
                                side=side,
                                setup=setup,
                                defender_mode=defender_mode,
                                difficulty=float(difficulty),
                                worlds=args.worlds_per_row,
                                horizon=args.horizon,
                                seed=(
                                    args.seed
                                    + difficulty_index * 10_000_000
                                    + setup * 100_000
                                    + defender_index * 10_000
                                ),
                                device=args.device,
                                collision_dir=args.collision_dir,
                            )
                        )
    checkpoint_hash_after = capability.sha256_file(checkpoint)
    if checkpoint_hash_after != checkpoint_hash_before:
        raise RuntimeError("V11 calibration mutated its source checkpoint")
    payload = {
        "format": VERSION,
        "created_utc": utc_now(),
        "scenario_identity": GROUND_TO_AIR_ENTRY_V11_VERSION,
        "probe_identity": GROUND_TO_AIR_ENTRY_PROBE_V11_VERSION,
        "checkpoint": {
            "path": checkpoint.as_posix(),
            "sha256": checkpoint_hash_before,
            "unchanged": True,
        },
        "source": {
            "human_reference_path": HUMAN_REFERENCE.relative_to(ROOT).as_posix(),
            "human_reference_sha256": capability.sha256_file(HUMAN_REFERENCE),
            "human_envelope": asdict(human_envelope_config()),
        },
        "worlds_per_row": args.worlds_per_row,
        "horizon": args.horizon,
        "difficulties": list(args.difficulty),
        "seed": args.seed,
        "rows": rows,
        "summary": summarize(rows),
        "optimizer_steps": 0,
        "reward_used_for_selection": False,
        "state_mutation_beyond_normal_simulation": False,
        "action_injection": False,
        "scripted_entry_actions": False,
        "verdict": "PASS",
    }
    write_json(args.output, payload)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True), flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=natural_v4.PARENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--collision-dir", type=Path, default=natural_v4.DEFAULT_COLLISION_DIR
    )
    parser.add_argument("--worlds-per-row", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=600)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--difficulty", type=float, nargs="+", default=[0.0])
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
