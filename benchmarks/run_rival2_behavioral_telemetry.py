"""Run the single authorized Rival 2.0 final-checkpoint behavioral evaluation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rivalsim.arena import ArenaGeometry, WarpArenaMeshes
from rivalsim.behavioral_telemetry import (
    BehavioralTelemetry,
    END_EPISODE,
    END_GOAL,
    END_NEXT_TOUCH,
    GOAL_CENTER_Y_UU,
    GOAL_CENTER_Z_UU,
    GOAL_HALF_WIDTH_UU,
    GOAL_HEIGHT_UU,
    GOAL_SCORING_PLANE_Y_UU,
    MAX_SURFACE_SEQUENCE,
    MAX_TOUCHES_PER_WORLD,
    PHYSICS_HZ,
    SURFACE_BACKBOARD,
    SURFACE_CEILING,
    SURFACE_GROUND,
    SURFACE_SIDE_WALL,
)
from rivalsim.rival2_contracts import OBS_DIM, RIVAL2_REWARD_VERSION
from rivalsim.rival2_env import Rival2Env
from rivalsim.rival2_policy import (
    Rival2ActorCritic,
    Rival2PolicyConfig,
    sample_hybrid_action,
)

EXPECTED_HEAD = "df295da1bcaec07170465f22fdc512b66fdd7538"
CHECKPOINT = Path(
    "checkpoints/rival2/overnight/rival2_overnight_final_6h_resume.pt"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "4DC158DC2A9D16B79FB5FE7D868E3B50928AB113B55DFCC753F3734F8D87372E"
)
EVALUATION_WORLDS = 16_384
EVALUATION_SEED = 920_260_826
EVALUATION_MAX_DECISIONS = 45 * 30
POLICY_HZ = 30
PHYSICS_TICKS_PER_DECISION = 4
DIRECTION_THRESHOLD_UU_PER_S = 100.0
FIELD_BACK_WALL_Y_UU = 5120.0
RAW_COMMIT_LIMIT_BYTES = 50 * 1024**2
REPRESENTATIVE_TOUCHES = 8_192
SCHEMA_VERSION = 1

SURFACE_LABELS = {
    SURFACE_GROUND: "ground",
    SURFACE_SIDE_WALL: "side_wall",
    SURFACE_BACKBOARD: "backboard_goal_structure",
    SURFACE_CEILING: "ceiling",
}
END_LABELS = {
    END_NEXT_TOUCH: "next_touch",
    END_GOAL: "goal",
    END_EPISODE: "episode_end",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collision-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/rival2/behavioral_telemetry"),
    )
    parser.add_argument(
        "--raw-work-dir",
        type=Path,
        default=Path(".tools/rival2-behavioral-telemetry"),
    )
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def _finite(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor).all().item())


def _ratio(count: int, denominator: int) -> dict[str, int | float | None]:
    return {
        "count": int(count),
        "denominator": int(denominator),
        "fraction": None if denominator == 0 else float(count / denominator),
    }


def _distribution(values: np.ndarray) -> dict[str, Any]:
    source = np.asarray(values, dtype=np.float64)
    source = source[np.isfinite(source)]
    if source.size == 0:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "quantiles": {},
        }
    quantiles = np.quantile(source, (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0))
    return {
        "count": int(source.size),
        "min": float(source.min()),
        "max": float(source.max()),
        "mean": float(source.mean()),
        "std": float(source.std()),
        "quantiles": {
            label: float(value)
            for label, value in zip(
                ("p00", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "p100"),
                quantiles,
                strict=True,
            )
        },
    }


def _histogram(values: np.ndarray, edges: np.ndarray) -> dict[str, Any]:
    source = np.asarray(values, dtype=np.float64)
    source = source[np.isfinite(source)]
    counts, used_edges = np.histogram(source, bins=edges)
    return {
        "denominator": int(source.size),
        "edges": used_edges.astype(float).tolist(),
        "counts": counts.astype(int).tolist(),
        "underflow": int((source < used_edges[0]).sum()),
        "overflow": int((source > used_edges[-1]).sum()),
    }


def _category_ratios(labels: np.ndarray, names: list[str]) -> dict[str, Any]:
    denominator = int(labels.size)
    return {
        name: _ratio(int((labels == index).sum()), denominator)
        for index, name in enumerate(names)
    }


def _checkpoint_authority(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if _git("rev-parse", "HEAD") != EXPECTED_HEAD:
        raise RuntimeError("behavioral evaluation must start at the authorized HEAD")
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("final overnight checkpoint SHA-256 mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    policy = Rival2PolicyConfig(**payload["policy_config"])
    checks = {
        "format_exact": payload.get("format") == "RIVAL2_CHECKPOINT_V1",
        "reward_v1_exact": payload.get("reward_version") == RIVAL2_REWARD_VERSION,
        "policy_config_exact": asdict(policy) == payload.get("policy_config"),
        "policy_config_hash_exact": policy.content_hash
        == payload.get("policy_config_hash"),
        "sha256_exact": actual_sha256 == EXPECTED_CHECKPOINT_SHA256,
    }
    if not all(checks.values()):
        raise RuntimeError(f"checkpoint authority failed: {checks}")
    identity = {
        "path": path.as_posix(),
        "sha256": actual_sha256,
        "size_bytes": path.stat().st_size,
        "format": payload["format"],
        "reward_version": payload["reward_version"],
        "policy_config_hash": payload["policy_config_hash"],
        "iteration": int(payload["iteration"]),
        "policy_version": int(payload["policy_version"]),
        "total_agent_samples": int(payload["total_agent_samples"]),
        "checks": checks,
    }
    return payload, identity


def _configuration(
    *,
    args: argparse.Namespace,
    checkpoint: dict[str, Any],
    geometry: ArenaGeometry,
) -> dict[str, Any]:
    kickoff_selector = (
        np.arange(EVALUATION_WORLDS, dtype=np.int64) + EVALUATION_SEED
    ) % 5
    evaluator_path = Path(__file__).resolve()
    telemetry_path = Path(__file__).resolve().parents[1] / "rivalsim" / "behavioral_telemetry.py"
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "authority": "handoff/rival2-behavioral-eval/README.md",
        "authorized_head": EXPECTED_HEAD,
        "evaluated_head": _git("rev-parse", "HEAD"),
        "evaluation_kind": "ordinary stochastic current-policy self-play",
        "execution_provenance": {
            "independent_stochastic_evaluations_published": 1,
            "process_attempts": 2,
            "rejected_attempts": 1,
            "rejected_attempt_detail": (
                "the identical frozen rollout completed, but publication was rejected because the "
                "initial telemetry-only surface transition buffer and 30 Hz touch cross-check were "
                "specified incorrectly; no metrics or raw events from that recorder attempt were "
                "retained or selected"
            ),
            "restart_policy": (
                "same checkpoint, seed, kickoff assignment, stochastic generator seed, policy, and "
                "simulator; only the read-only recorder was corrected before the published replay"
            ),
        },
        "worlds": EVALUATION_WORLDS,
        "seed": EVALUATION_SEED,
        "policy_sampling_seed": EVALUATION_SEED,
        "maximum_decisions": EVALUATION_MAX_DECISIONS,
        "policy_hz": POLICY_HZ,
        "physics_hz": PHYSICS_HZ,
        "physics_ticks_per_decision": PHYSICS_TICKS_PER_DECISION,
        "episode_scope": "first completed episode per world",
        "opponent": "same final checkpoint controlling both cars; no historical opponent",
        "kickoff_layout_assignment": "(world_index + evaluation_seed) modulo 5",
        "kickoff_layout_counts": {
            str(layout): int((kickoff_selector == layout).sum()) for layout in range(5)
        },
        "canonicalization": (
            "180-degree field-plane rotation for orange touches: canonical X/Y and X/Y "
            "velocity are multiplied by +1 for blue, -1 for orange; canonical +Y is the "
            "toucher's opponent-goal direction; Z is unchanged"
        ),
        "direction_threshold": {
            "quantity": "immediate canonical post-touch longitudinal velocity",
            "symmetric_threshold_uu_per_s": DIRECTION_THRESHOLD_UU_PER_S,
            "forward": f"> +{DIRECTION_THRESHOLD_UU_PER_S:g}",
            "lateral_neutral": (
                f"between -{DIRECTION_THRESHOLD_UU_PER_S:g} and "
                f"+{DIRECTION_THRESHOLD_UU_PER_S:g}, inclusive"
            ),
            "backward": f"< -{DIRECTION_THRESHOLD_UU_PER_S:g}",
            "authority_note": "classification is descriptive; continuous values remain primary",
        },
        "goal_geometry": {
            "scoring_plane_center_y_uu": GOAL_SCORING_PLANE_Y_UU,
            "scoring_plane_source": (
                "rivalsim/kernels/lifecycle.py: GOAL_BASE_THRESHOLD_Y + BALL_RADIUS"
            ),
            "opponent_goal_center_y_uu": GOAL_CENTER_Y_UU,
            "goal_center_z_uu": GOAL_CENTER_Z_UU,
            "goal_half_width_uu": GOAL_HALF_WIDTH_UU,
            "goal_height_uu": GOAL_HEIGHT_UU,
            "mouth_bounds_for_ball_center_projection": {
                "x_uu": [-GOAL_HALF_WIDTH_UU, GOAL_HALF_WIDTH_UU],
                "z_uu": [0.0, GOAL_HEIGHT_UU],
            },
            "mouth_geometry_source": (
                "docs/PHYSICS_ORACLES.md RLBot/RLGym arena values; used descriptively, not as "
                "a collision substitute or an episode rule"
            ),
        },
        "surface_classification": {
            "source": (
                "retained ball-world contact normals from the actual static CMF collision path"
            ),
            "rule": (
                "dominant absolute normal axis: +Z ground, -Z ceiling, X side wall, "
                "Y backboard/goal structure"
            ),
            "curved_surface_note": (
                "ramps and curved transitions inherit the category of their dominant normal axis"
            ),
            "sequence_rule": (
                "record category contact onsets; persistent contacts do not repeat, and simultaneous "
                "new categories use the declared ground, side-wall, backboard, ceiling order"
            ),
        },
        "touch_sequence_end": (
            "next accepted player touch, scoring event, or first-episode termination/truncation"
        ),
        "simultaneous_touch_order": (
            "the existing per-world source-backed car _PreTickUpdate visitation order"
        ),
        "telemetry_capacity": {
            "touches_per_world": MAX_TOUCHES_PER_WORLD,
            "surface_transitions_since_final_touch": MAX_SURFACE_SEQUENCE,
            "overflow_is_fatal": True,
        },
        "checkpoint": checkpoint,
        "collision_geometry": geometry.metadata(),
        "device": args.device,
        "platform": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "os": platform.platform(),
            "gpu": torch.cuda.get_device_name(torch.device(args.device)),
        },
        "evaluator_sources": {
            evaluator_path.relative_to(Path.cwd()).as_posix(): _sha256_file(evaluator_path),
            telemetry_path.relative_to(Path.cwd()).as_posix(): _sha256_file(telemetry_path),
        },
        "non_interference": {
            "training_updates": 0,
            "checkpoint_mutated": False,
            "reward_model_ppo_observation_action_episode_simulator_semantics_changed": False,
            "telemetry_feedback_to_policy_or_simulator": False,
            "telemetry_collection": "post-physics read-only Warp kernel at 120 Hz",
        },
    }


@torch.no_grad()
def _run_evaluation(
    *,
    args: argparse.Namespace,
    payload: dict[str, Any],
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    device = torch.device(args.device)
    kickoff_selector = (
        np.arange(EVALUATION_WORLDS, dtype=np.int32) + EVALUATION_SEED
    ) % 5
    env = Rival2Env(
        EVALUATION_WORLDS,
        args.collision_dir,
        device=args.device,
        seed=EVALUATION_SEED,
        reward_version=RIVAL2_REWARD_VERSION,
        geometry=geometry,
        meshes=meshes,
        kickoff_selector=kickoff_selector,
    )
    telemetry = BehavioralTelemetry(EVALUATION_WORLDS, env.world.device)
    telemetry.attach(env.world)
    policy = Rival2PolicyConfig(**payload["policy_config"])
    model = Rival2ActorCritic(policy).to(device)
    model.load_state_dict(payload["model"])
    model.eval().requires_grad_(False)
    del payload
    gc.collect()

    generator = torch.Generator(device=device).manual_seed(EVALUATION_SEED)
    active = torch.ones(EVALUATION_WORLDS, dtype=torch.bool, device=device)
    completed = torch.zeros(EVALUATION_WORLDS, dtype=torch.bool, device=device)
    episode_steps = torch.zeros(EVALUATION_WORLDS, dtype=torch.int32, device=device)
    terminated_count = torch.zeros((), dtype=torch.int64, device=device)
    no_touch_count = torch.zeros((), dtype=torch.int64, device=device)
    hard_count = torch.zeros((), dtype=torch.int64, device=device)
    accepted_touch_count = torch.zeros((), dtype=torch.int64, device=device)
    world_decisions = torch.zeros((), dtype=torch.int64, device=device)
    duration_sum = torch.zeros((), dtype=torch.float64, device=device)
    analog_abs_sum = torch.zeros(5, dtype=torch.float64, device=device)
    button_sum = torch.zeros(3, dtype=torch.float64, device=device)
    action_count = torch.zeros((), dtype=torch.float64, device=device)
    integrity = {
        "observations_finite": True,
        "actor_outputs_finite": True,
        "values_finite": True,
        "actions_finite": True,
        "analog_bounds": True,
        "buttons_exact_binary": True,
        "exclusive_done_kind": True,
        "all_worlds_completed": False,
        "zero_hot_h2d": False,
        "zero_hot_d2h": False,
        "telemetry_all_first_episodes_closed": False,
        "telemetry_touch_capacity_not_exceeded": False,
        "telemetry_surface_sequence_capacity_not_exceeded": False,
        "telemetry_goal_count_matches_episode_terminations": False,
        "telemetry_touch_count_not_below_interval_touch_presence": False,
    }
    env.reset_transfer_counters()
    started = time.perf_counter()
    decisions_executed = 0
    for decision in range(EVALUATION_MAX_DECISIONS):
        decisions_executed = decision + 1
        observation = env.observation
        integrity["observations_finite"] &= _finite(observation)
        actor, value = model(observation.reshape(-1, OBS_DIM))
        actor = actor.reshape(EVALUATION_WORLDS, 2, 13)
        value = value.reshape(EVALUATION_WORLDS, 2)
        integrity["actor_outputs_finite"] &= _finite(actor)
        integrity["values_finite"] &= _finite(value)
        action = sample_hybrid_action(actor, generator=generator, config=policy).action
        action = torch.where(active[:, None, None], action, torch.zeros_like(action))
        integrity["actions_finite"] &= _finite(action)
        integrity["analog_bounds"] &= bool(
            ((action[..., :5] >= -1.0) & (action[..., :5] <= 1.0)).all().item()
        )
        integrity["buttons_exact_binary"] &= bool(
            ((action[..., 5:] == 0.0) | (action[..., 5:] == 1.0)).all().item()
        )
        mask = active[:, None, None]
        analog_abs_sum += (action[..., :5].abs() * mask).sum((0, 1)).double()
        button_sum += (action[..., 5:] * mask).sum((0, 1)).double()
        action_count += active.sum().double() * 2.0

        transition = env.step(action)
        interval_active = active[:, None]
        accepted_touch_count += (
            (transition.transition_observation[..., 176] > 0.5) & interval_active
        ).sum().to(torch.int64)
        episode_steps += active.to(torch.int32)
        world_decisions += active.sum().to(torch.int64)
        done = active & (transition.terminated | transition.truncated)
        integrity["exclusive_done_kind"] &= not bool(
            (transition.terminated & transition.truncated & active).any().item()
        )
        terminated_now = done & transition.terminated
        truncated_now = done & transition.truncated
        no_touch_now = truncated_now & (
            transition.transition_observation[:, 0, 181] >= 1.0
        )
        hard_now = truncated_now & ~no_touch_now
        terminated_count += terminated_now.sum().to(torch.int64)
        no_touch_count += no_touch_now.sum().to(torch.int64)
        hard_count += hard_now.sum().to(torch.int64)
        duration_sum += episode_steps[done].sum().double() / POLICY_HZ
        completed |= done
        active &= ~done
        if decision % 100 == 0 or not bool(active.any().item()):
            remaining = int(active.sum().item())
            elapsed = time.perf_counter() - started
            print(
                f"behavioral-eval decision={decision + 1}/{EVALUATION_MAX_DECISIONS} "
                f"remaining_worlds={remaining} elapsed_seconds={elapsed:.1f}",
                flush=True,
            )
        if not bool(active.any().item()):
            break

    transfer = env.hot_path_transfer_bytes()
    raw = telemetry.numpy()
    elapsed = time.perf_counter() - started
    completed_episodes = int(completed.sum().item())
    telemetry_touch_count = int(raw["touch_count"].astype(np.int64).sum())
    telemetry_goals = int(raw["goal_valid"].sum())
    integrity["all_worlds_completed"] = completed_episodes == EVALUATION_WORLDS
    integrity["zero_hot_h2d"] = transfer["h2d"] == 0
    integrity["zero_hot_d2h"] = transfer["d2h"] == 0
    integrity["telemetry_all_first_episodes_closed"] = bool(
        (raw["episode_open"] == 0).all()
    )
    integrity["telemetry_touch_capacity_not_exceeded"] = bool(
        (raw["touch_overflow"] == 0).all()
    )
    integrity["telemetry_surface_sequence_capacity_not_exceeded"] = bool(
        (raw["surface_sequence_overflow"] == 0).all()
    )
    integrity["telemetry_goal_count_matches_episode_terminations"] = (
        telemetry_goals == int(terminated_count.item())
    )
    integrity["telemetry_touch_count_not_below_interval_touch_presence"] = (
        telemetry_touch_count >= int(accepted_touch_count.item())
    )
    if not all(integrity.values()):
        raise RuntimeError(f"single evaluation integrity failure: {integrity}")

    denominator = float(action_count.item())
    run = {
        "schema_version": SCHEMA_VERSION,
        "verdict": "PASS_GREEN",
        "worlds": EVALUATION_WORLDS,
        "completed_episodes": completed_episodes,
        "decisions_executed": decisions_executed,
        "world_decisions": int(world_decisions.item()),
        "simulated_seconds": float(world_decisions.item()) / POLICY_HZ,
        "wall_clock_seconds": elapsed,
        "goal_terminated_episodes": int(terminated_count.item()),
        "no_touch_truncated_episodes": int(no_touch_count.item()),
        "hard_truncated_episodes": int(hard_count.item()),
        "mean_episode_duration_seconds": float(duration_sum.item())
        / completed_episodes,
        "accepted_touch_entries": telemetry_touch_count,
        "thirty_hz_car_intervals_with_touch_presence": int(accepted_touch_count.item()),
        "touch_cross_check_note": (
            "the 30 Hz observation is boolean touch presence per car/decision interval; the 120 Hz "
            "telemetry counts distinct accepted entries, so it must be greater than or equal"
        ),
        "mean_absolute_analog_action": {
            name: float(analog_abs_sum[index].item()) / denominator
            for index, name in enumerate(("throttle", "steer", "pitch", "yaw", "roll"))
        },
        "button_activation_rate": {
            name: float(button_sum[index].item()) / denominator
            for index, name in enumerate(("jump", "boost", "handbrake"))
        },
        "integrity": integrity,
        "hot_path_transfer_bytes": transfer,
    }
    del env, telemetry, model
    gc.collect()
    torch.cuda.empty_cache()
    return run, raw


def _flatten_touch_raw(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    counts = raw["touch_count"].astype(np.int64)
    clipped = np.minimum(counts, MAX_TOUCHES_PER_WORLD)
    mask = np.arange(MAX_TOUCHES_PER_WORLD)[None, :] < clipped[:, None]
    linear = np.arange(EVALUATION_WORLDS * MAX_TOUCHES_PER_WORLD).reshape(
        EVALUATION_WORLDS, MAX_TOUCHES_PER_WORLD
    )
    result: dict[str, np.ndarray] = {
        "event_linear_index": linear[mask].astype(np.int32),
        "world": np.broadcast_to(
            np.arange(EVALUATION_WORLDS, dtype=np.int32)[:, None], mask.shape
        )[mask],
        "slot": np.broadcast_to(
            np.arange(MAX_TOUCHES_PER_WORLD, dtype=np.int32)[None, :], mask.shape
        )[mask],
    }
    event_prefix = "event_"
    for name, value in raw.items():
        if not name.startswith(event_prefix):
            continue
        trailing = value.shape[1:]
        reshaped = value.reshape(
            (EVALUATION_WORLDS, MAX_TOUCHES_PER_WORLD, *trailing)
        )
        result[name.removeprefix(event_prefix)] = reshaped[mask]
    return result


def _derive_touch_data(
    touches: dict[str, np.ndarray], raw: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], list[int], list[dict[str, int]]]:
    count = touches["world"].size
    toucher = touches["toucher"].astype(np.int32)
    sign = np.where(toucher == 0, 1.0, -1.0).astype(np.float32)
    canonical_position = touches["position_after"].copy()
    canonical_velocity_before = touches["velocity_before"].copy()
    canonical_velocity_after = touches["velocity_after"].copy()
    canonical_position[:, :2] *= sign[:, None]
    canonical_velocity_before[:, :2] *= sign[:, None]
    canonical_velocity_after[:, :2] *= sign[:, None]
    touches["canonical_position_after"] = canonical_position
    touches["canonical_velocity_before"] = canonical_velocity_before
    touches["canonical_velocity_after"] = canonical_velocity_after
    touches["speed_before"] = np.linalg.norm(
        touches["velocity_before"], axis=1
    ).astype(np.float32)
    touches["speed_after"] = np.linalg.norm(
        touches["velocity_after"], axis=1
    ).astype(np.float32)
    touches["episode_time_seconds"] = touches["tick"].astype(np.float32) / PHYSICS_HZ
    touches["time_to_next_touch_seconds"] = np.where(
        touches["next_toucher"] >= 0,
        (touches["end_tick"] - touches["tick"]) / PHYSICS_HZ,
        np.nan,
    ).astype(np.float32)
    touches["heading_to_goal_3d_degrees"] = np.degrees(
        touches["heading_to_goal_3d"]
    ).astype(np.float32)
    touches["heading_to_goal_planar_degrees"] = np.degrees(
        touches["heading_to_goal_planar"]
    ).astype(np.float32)

    for label, field, bit in (
        ("025s", "position_025s", 1),
        ("050s", "position_050s", 2),
        ("100s", "position_100s", 4),
        ("200s", "position_200s", 8),
    ):
        valid = (touches["horizon_valid_bits"] & bit) != 0
        displacement = np.full((count, 3), np.nan, dtype=np.float32)
        displacement[valid] = touches[field][valid] - touches["position_after"][valid]
        displacement[valid, :2] *= sign[valid, None]
        touches[f"horizon_{label}_valid"] = valid.astype(np.uint8)
        touches[f"canonical_displacement_{label}"] = displacement

    goal_valid_by_world = raw["goal_valid"].astype(bool)
    goal_tick_by_world = raw["goal_tick"].astype(np.int32)
    goal_side_by_world = raw["goal_scoring_side"].astype(np.int32)
    event_world = touches["world"]
    future_goal = goal_valid_by_world[event_world] & (
        goal_tick_by_world[event_world] >= touches["tick"]
    )
    ticks_to_goal = goal_tick_by_world[event_world] - touches["tick"]
    touches["time_to_goal_seconds"] = np.where(
        future_goal, ticks_to_goal / PHYSICS_HZ, np.nan
    ).astype(np.float32)
    touches["goal_scoring_side"] = np.where(
        future_goal, goal_side_by_world[event_world], -1
    ).astype(np.int8)
    for seconds in (1, 3, 5):
        touches[f"goal_within_{seconds}s"] = (
            future_goal & (ticks_to_goal <= seconds * PHYSICS_HZ)
        ).astype(np.uint8)
    touches["final_touch_before_goal"] = (
        future_goal
        & (
            raw["goal_last_touch_event"][event_world]
            == touches["event_linear_index"]
        )
    ).astype(np.uint8)

    field_third = np.ones(count, dtype=np.int8)
    boundary = FIELD_BACK_WALL_Y_UU / 3.0
    field_third[canonical_position[:, 1] < -boundary] = 0
    field_third[canonical_position[:, 1] > boundary] = 2
    touches["field_third"] = field_third

    chain_length_per_touch = np.zeros(count, dtype=np.int32)
    chain_lengths: list[int] = []
    chains: list[dict[str, int]] = []
    cursor = 0
    counts = np.minimum(raw["touch_count"], MAX_TOUCHES_PER_WORLD).astype(np.int64)
    for world, world_count in enumerate(counts):
        world_count_int = int(world_count)
        end = cursor + world_count_int
        local = cursor
        while local < end:
            owner = int(toucher[local])
            chain_end = local + 1
            while chain_end < end and int(toucher[chain_end]) == owner:
                chain_end += 1
            length = chain_end - local
            chain_length_per_touch[local:chain_end] = length
            chain_lengths.append(length)
            chains.append(
                {
                    "world": world,
                    "toucher": owner,
                    "length": length,
                    "start_tick": int(touches["tick"][local]),
                    "end_tick": int(touches["tick"][chain_end - 1]),
                }
            )
            local = chain_end
        cursor = end
    touches["possession_chain_length"] = chain_length_per_touch
    return touches, chain_lengths, chains


def _touch_summaries(
    touches: dict[str, np.ndarray], chain_lengths: list[int]
) -> tuple[dict[str, Any], dict[str, Any]]:
    total = int(touches["world"].size)
    longitudinal = touches["canonical_velocity_after"][:, 1]
    direction = np.ones(total, dtype=np.int8)
    direction[longitudinal > DIRECTION_THRESHOLD_UU_PER_S] = 2
    direction[longitudinal < -DIRECTION_THRESHOLD_UU_PER_S] = 0
    direction_counts = _category_ratios(
        direction, ["backward", "lateral_neutral", "forward"]
    )
    next_toucher = touches["next_toucher"]
    same = next_toucher == touches["toucher"]
    opponent = (next_toucher >= 0) & ~same
    none = next_toucher < 0
    surface_bits = touches["surface_bits"]
    field_labels = touches["field_third"]
    projection_defined = touches["projection_defined"] != 0
    projection_inside = touches["projection_inside_mouth"] != 0

    horizons: dict[str, Any] = {}
    for label in ("025s", "050s", "100s", "200s"):
        valid = touches[f"horizon_{label}_valid"] != 0
        displacement = touches[f"canonical_displacement_{label}"]
        horizons[label] = {
            "available": _ratio(int(valid.sum()), total),
            "canonical_y_displacement_uu": _distribution(displacement[valid, 1]),
            "three_dimensional_displacement_uu": _distribution(
                np.linalg.norm(displacement[valid], axis=1)
            ),
        }

    touch_to_goal: dict[str, Any] = {}
    for seconds in (1, 3, 5):
        mask = touches[f"goal_within_{seconds}s"] != 0
        same_side = mask & (touches["goal_scoring_side"] == touches["toucher"])
        opposite_side = mask & ~same_side
        touch_to_goal[f"within_{seconds}s"] = {
            "any_goal": _ratio(int(mask.sum()), total),
            "toucher_side_scored": _ratio(int(same_side.sum()), total),
            "opposite_side_scored": _ratio(int(opposite_side.sum()), total),
        }
    final_touch = touches["final_touch_before_goal"] != 0
    chain_counter = Counter(chain_lengths)
    possession = {
        "schema_version": SCHEMA_VERSION,
        "touch_denominator": total,
        "primitive_possession_chain_denominator": len(chain_lengths),
        "definition": (
            "maximal consecutive accepted touches by the same player, ending at an opponent "
            "touch, goal, or episode end"
        ),
        "chain_length_distribution": {
            str(length): {
                "count": int(chain_counter[length]),
                "denominator": len(chain_lengths),
                "fraction": float(chain_counter[length] / len(chain_lengths)),
            }
            for length in sorted(chain_counter)
        },
        "chain_length_continuous": _distribution(np.asarray(chain_lengths)),
        "same_player_next_touch": _ratio(int(same.sum()), total),
        "opponent_next_touch": _ratio(int(opponent.sum()), total),
        "no_next_touch_before_sequence_end": _ratio(int(none.sum()), total),
        "among_touches_with_a_next_touch": {
            "same_player": _ratio(int(same.sum()), int((~none).sum())),
            "opponent": _ratio(int(opponent.sum()), int((~none).sum())),
        },
        "time_to_next_touch_seconds": _distribution(
            touches["time_to_next_touch_seconds"][~none]
        ),
    }
    touch_summary = {
        "schema_version": SCHEMA_VERSION,
        "touch_denominator": total,
        "canonicalization": "+Y is the toucher's opponent-goal direction",
        "direction": {
            "threshold_uu_per_s": DIRECTION_THRESHOLD_UU_PER_S,
            "classes": direction_counts,
            "raw_canonical_longitudinal_velocity_uu_per_s": _distribution(longitudinal),
            "raw_histogram": _histogram(
                longitudinal,
                np.asarray(
                    [-6000, -4000, -3000, -2000, -1000, -500, -100, 0, 100, 500, 1000, 2000, 3000, 4000, 6000],
                    dtype=np.float64,
                ),
            ),
        },
        "speed_before_uu_per_s": _distribution(touches["speed_before"]),
        "speed_after_uu_per_s": _distribution(touches["speed_after"]),
        "immediate_longitudinal_velocity_delta_uu_per_s": _distribution(
            touches["longitudinal_delta"]
        ),
        "heading_to_opponent_goal_center_degrees": {
            "three_dimensional": _distribution(touches["heading_to_goal_3d_degrees"]),
            "field_plane": _distribution(touches["heading_to_goal_planar_degrees"]),
        },
        "instantaneous_straight_line_projection": {
            "label_warning": (
                "descriptive instantaneous projection only; outside-mouth projections are not "
                "labeled missed shots"
            ),
            "defined": _ratio(int(projection_defined.sum()), total),
            "inside_authoritative_mouth": _ratio(
                int((projection_defined & projection_inside).sum()),
                int(projection_defined.sum()),
            ),
            "time_to_plane_seconds": _distribution(
                touches["projection_time"][projection_defined]
            ),
            "intercept_x_uu": _distribution(touches["projection_x"][projection_defined]),
            "intercept_z_uu": _distribution(touches["projection_z"][projection_defined]),
        },
        "actual_sequence": {
            "canonical_net_y_displacement_uu": _distribution(touches["net_y"]),
            "maximum_forward_y_excursion_uu": _distribution(touches["max_forward_y"]),
            "maximum_backward_y_excursion_uu": _distribution(touches["max_backward_y"]),
            "sequence_duration_seconds": _distribution(
                (touches["end_tick"] - touches["tick"]) / PHYSICS_HZ
            ),
            "end_reason": {
                label: _ratio(int((touches["end_reason"] == code).sum()), total)
                for code, label in END_LABELS.items()
            },
        },
        "fixed_horizons": horizons,
        "surface_continuation": {
            label: _ratio(int(((surface_bits & (1 << (code - 1))) != 0).sum()), total)
            for code, label in SURFACE_LABELS.items()
        },
        "field_thirds": {
            "canonical_boundaries_y_uu": [
                -FIELD_BACK_WALL_Y_UU,
                -FIELD_BACK_WALL_Y_UU / 3.0,
                FIELD_BACK_WALL_Y_UU / 3.0,
                FIELD_BACK_WALL_Y_UU,
            ],
            "defensive": _ratio(int((field_labels == 0).sum()), total),
            "middle": _ratio(int((field_labels == 1).sum()), total),
            "attacking": _ratio(int((field_labels == 2).sum()), total),
        },
        "next_touch": possession,
        "touch_to_goal": touch_to_goal,
        "final_touch_before_goal": {
            "touch_denominator": _ratio(int(final_touch.sum()), total),
            "time_to_goal_seconds": _distribution(
                touches["time_to_goal_seconds"][final_touch]
            ),
        },
    }
    return touch_summary, possession


def _decode_sequence(row: np.ndarray, length: int) -> list[str]:
    return [SURFACE_LABELS[int(value)] for value in row[:length] if int(value) != 0]


def _goal_summaries(
    raw: dict[str, np.ndarray], touches: dict[str, np.ndarray]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    goal_mask = raw["goal_valid"].astype(bool)
    worlds = np.nonzero(goal_mask)[0].astype(np.int32)
    scoring = raw["goal_scoring_side"][goal_mask].astype(np.int8)
    tick = raw["goal_tick"][goal_mask].astype(np.int32)
    position = raw["goal_position"][goal_mask].astype(np.float32)
    velocity = raw["goal_velocity"][goal_mask].astype(np.float32)
    crossing_valid = raw["goal_crossing_valid"][goal_mask].astype(bool)
    crossing = raw["goal_crossing_position"][goal_mask].astype(np.float32)
    last_event = raw["goal_last_touch_event"][goal_mask].astype(np.int32)
    sequence_length = raw["goal_surface_sequence_length"][goal_mask].astype(np.int32)
    sequence_rows = raw["goal_surface_sequence"].reshape(
        EVALUATION_WORLDS, MAX_SURFACE_SEQUENCE
    )[goal_mask]
    total = int(worlds.size)
    sign = np.where(scoring == 0, 1.0, -1.0).astype(np.float32)
    canonical_velocity = velocity.copy()
    canonical_velocity[:, :2] *= sign[:, None]
    entry_speed = np.linalg.norm(velocity, axis=1).astype(np.float32)
    cosine = np.divide(
        canonical_velocity[:, 1],
        entry_speed,
        out=np.zeros(total, dtype=np.float32),
        where=entry_speed > 0,
    )
    entry_angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))).astype(np.float32)
    x = crossing[:, 0]
    z = crossing[:, 2]
    x_normalized = x / GOAL_HALF_WIDTH_UU
    z_normalized = z / GOAL_HEIGHT_UU
    inside = crossing_valid & (np.abs(x) <= GOAL_HALF_WIDTH_UU) & (z >= 0.0) & (
        z <= GOAL_HEIGHT_UU
    )

    linear_to_flat = {
        int(linear): index for index, linear in enumerate(touches["event_linear_index"])
    }
    last_toucher = np.full(total, -1, dtype=np.int8)
    last_touch_tick = np.full(total, -1, dtype=np.int32)
    possession_touches = np.zeros(total, dtype=np.int32)
    for goal_index, event in enumerate(last_event):
        touch_index = linear_to_flat.get(int(event))
        if touch_index is not None:
            last_toucher[goal_index] = np.int8(touches["toucher"][touch_index])
            last_touch_tick[goal_index] = np.int32(touches["tick"][touch_index])
            possession_touches[goal_index] = np.int32(
                touches["possession_chain_length"][touch_index]
            )
    time_since_touch = np.where(
        last_touch_tick >= 0, (tick - last_touch_tick) / PHYSICS_HZ, np.nan
    ).astype(np.float32)

    sequences = [
        _decode_sequence(row, int(length))
        for row, length in zip(sequence_rows, sequence_length, strict=True)
    ]
    sequence_labels = ["direct" if not sequence else ">".join(sequence) for sequence in sequences]
    sequence_counter = Counter(sequence_labels)
    broad_labels: list[str] = []
    for sequence, toucher_value in zip(sequences, last_toucher, strict=True):
        if toucher_value < 0:
            broad_labels.append("no_player_touch")
        elif not sequence:
            broad_labels.append("direct")
        elif "backboard_goal_structure" in sequence:
            broad_labels.append("backboard_goal_structure")
        elif "side_wall" in sequence:
            broad_labels.append("side_wall")
        elif "ground" in sequence:
            broad_labels.append("ground")
        elif "ceiling" in sequence:
            broad_labels.append("ceiling")
        else:
            broad_labels.append("other_contact")
    broad_counter = Counter(broad_labels)

    mouth_denominator = int(inside.sum())
    horizontal = {
        "left": _ratio(int((inside & (x_normalized < -1.0 / 3.0)).sum()), mouth_denominator),
        "center": _ratio(
            int((inside & (np.abs(x_normalized) <= 1.0 / 3.0)).sum()),
            mouth_denominator,
        ),
        "right": _ratio(int((inside & (x_normalized > 1.0 / 3.0)).sum()), mouth_denominator),
    }
    vertical = {
        "low": _ratio(int((inside & (z_normalized < 1.0 / 3.0)).sum()), mouth_denominator),
        "mid": _ratio(
            int((inside & (z_normalized >= 1.0 / 3.0) & (z_normalized < 2.0 / 3.0)).sum()),
            mouth_denominator,
        ),
        "high": _ratio(int((inside & (z_normalized >= 2.0 / 3.0)).sum()), mouth_denominator),
    }
    histogram_x_edges = np.linspace(-1.0, 1.0, 9)
    histogram_z_edges = np.linspace(0.0, 1.0, 7)
    histogram, x_edges, z_edges = np.histogram2d(
        x_normalized[inside], z_normalized[inside], bins=(histogram_x_edges, histogram_z_edges)
    )
    mouth = {
        "schema_version": SCHEMA_VERSION,
        "crossing_goal_denominator": total,
        "interpolated_crossing_valid": _ratio(int(crossing_valid.sum()), total),
        "inside_declared_goal_mouth": _ratio(mouth_denominator, int(crossing_valid.sum())),
        "normalization": {
            "x": f"crossing_x / {GOAL_HALF_WIDTH_UU}",
            "z": f"crossing_z / {GOAL_HEIGHT_UU}",
            "inside_bounds": {"x": [-1.0, 1.0], "z": [0.0, 1.0]},
        },
        "bin_boundaries": {
            "horizontal_thirds_normalized_x": [-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0],
            "vertical_thirds_normalized_z": [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
            "center_definition": "absolute normalized X <= 1/3",
            "corner_definition": "inside mouth and absolute normalized X > 1/3",
        },
        "horizontal": horizontal,
        "vertical": vertical,
        "center": _ratio(
            int((inside & (np.abs(x_normalized) <= 1.0 / 3.0)).sum()), mouth_denominator
        ),
        "corner": _ratio(
            int((inside & (np.abs(x_normalized) > 1.0 / 3.0)).sum()), mouth_denominator
        ),
        "absolute_x_uu": _distribution(x[crossing_valid]),
        "absolute_z_uu": _distribution(z[crossing_valid]),
        "normalized_x": _distribution(x_normalized[crossing_valid]),
        "normalized_z": _distribution(z_normalized[crossing_valid]),
        "histogram_2d": {
            "orientation": "counts[x_bin][z_bin]",
            "denominator_inside_mouth": mouth_denominator,
            "x_edges_normalized": x_edges.astype(float).tolist(),
            "z_edges_normalized": z_edges.astype(float).tolist(),
            "counts": histogram.astype(int).tolist(),
        },
    }
    goal_summary = {
        "schema_version": SCHEMA_VERSION,
        "goal_denominator": total,
        "scoring_side": {
            "blue": _ratio(int((scoring == 0).sum()), total),
            "orange": _ratio(int((scoring == 1).sum()), total),
        },
        "last_player_touch_present": _ratio(int((last_toucher >= 0).sum()), total),
        "last_toucher_side_matches_scorer": _ratio(
            int(((last_toucher >= 0) & (last_toucher == scoring)).sum()),
            int((last_toucher >= 0).sum()),
        ),
        "time_from_last_touch_to_goal_seconds": _distribution(time_since_touch),
        "touches_in_final_primitive_possession": _distribution(possession_touches),
        "scoring_event_ball_position_uu": {
            axis: _distribution(position[:, index])
            for index, axis in enumerate(("x", "y", "z"))
        },
        "scoring_event_ball_velocity_uu_per_s": {
            axis: _distribution(velocity[:, index])
            for index, axis in enumerate(("x", "y", "z"))
        },
        "entry_speed_uu_per_s": _distribution(entry_speed),
        "entry_angle_from_scoring_direction_degrees": _distribution(entry_angle),
        "contact_sequence_since_final_touch": {
            "exact_sequences": {
                label: _ratio(int(sequence_counter[label]), total)
                for label in sorted(sequence_counter)
            },
            "broad_classes": {
                label: _ratio(int(broad_counter[label]), total)
                for label in sorted(broad_counter)
            },
            "wall_backboard_note": (
                "wall and backboard paths are descriptive continuations, not accuracy errors"
            ),
        },
        "goal_mouth": mouth,
    }
    goal_raw = {
        "goal_world": worlds,
        "goal_scoring_side": scoring,
        "goal_episode_tick": tick,
        "goal_episode_time_seconds": tick.astype(np.float32) / PHYSICS_HZ,
        "goal_last_toucher": last_toucher,
        "goal_last_touch_tick": last_touch_tick,
        "goal_time_since_last_touch_seconds": time_since_touch,
        "goal_scoring_event_position_uu": position,
        "goal_scoring_event_velocity_uu_per_s": velocity,
        "goal_crossing_valid": crossing_valid.astype(np.uint8),
        "goal_crossing_position_uu": crossing,
        "goal_crossing_x_normalized": x_normalized.astype(np.float32),
        "goal_crossing_z_normalized": z_normalized.astype(np.float32),
        "goal_crossing_inside_mouth": inside.astype(np.uint8),
        "goal_entry_speed_uu_per_s": entry_speed,
        "goal_entry_velocity_canonical_uu_per_s": canonical_velocity,
        "goal_entry_angle_degrees": entry_angle,
        "goal_touches_in_final_possession": possession_touches,
        "goal_surface_sequence_length": sequence_length,
        "goal_surface_sequence_codes": sequence_rows.astype(np.int8),
    }
    return goal_summary, mouth, goal_raw


def _raw_artifact(
    *,
    output_dir: Path,
    raw_work_dir: Path,
    configuration: dict[str, Any],
    touches: dict[str, np.ndarray],
    goal_raw: dict[str, np.ndarray],
) -> dict[str, Any]:
    raw_work_dir.mkdir(parents=True, exist_ok=True)
    full_path = raw_work_dir / "raw_events_full.npz"
    touch_fields = {
        f"touch_{name}": value
        for name, value in touches.items()
        if name
        not in {
            "event_linear_index",
        }
    }
    np.savez_compressed(
        full_path,
        schema_json=np.asarray(json.dumps(configuration, sort_keys=True)),
        **touch_fields,
        **goal_raw,
    )
    full_sha256 = _sha256_file(full_path)
    full_size = full_path.stat().st_size
    output_dir.mkdir(parents=True, exist_ok=True)
    if full_size <= RAW_COMMIT_LIMIT_BYTES:
        published = output_dir / "raw_events.npz"
        shutil.copy2(full_path, published)
        return {
            "kind": "complete_raw_event_telemetry",
            "path": published.as_posix(),
            "sha256": _sha256_file(published),
            "size_bytes": published.stat().st_size,
            "full_raw_path": published.as_posix(),
            "full_raw_sha256": full_sha256,
            "full_raw_size_bytes": full_size,
            "representative_sample": False,
        }

    sample_count = min(REPRESENTATIVE_TOUCHES, touches["world"].size)
    sample_index = np.arange(sample_count, dtype=np.int64)
    sample_path = output_dir / "raw_events_representative_sample.npz"
    sampled_touches = {
        name: value[sample_index]
        for name, value in touch_fields.items()
    }
    np.savez_compressed(
        sample_path,
        schema_json=np.asarray(json.dumps(configuration, sort_keys=True)),
        representative_sample_rule=np.asarray(
            f"first {sample_count} events in deterministic world/accepted-touch order"
        ),
        **sampled_touches,
        **goal_raw,
    )
    return {
        "kind": "deterministic_representative_touch_sample_plus_all_goal_events",
        "path": sample_path.as_posix(),
        "sha256": _sha256_file(sample_path),
        "size_bytes": sample_path.stat().st_size,
        "representative_sample": True,
        "representative_touch_count": sample_count,
        "representative_sample_rule": (
            f"first {sample_count} events in deterministic world/accepted-touch order"
        ),
        "full_raw_path": full_path.resolve().as_posix(),
        "full_raw_sha256": full_sha256,
        "full_raw_size_bytes": full_size,
    }


def _report(
    *,
    path: Path,
    summary: dict[str, Any],
    touch: dict[str, Any],
    goal: dict[str, Any],
    possession: dict[str, Any],
    raw_artifact: dict[str, Any],
) -> None:
    direction = touch["direction"]["classes"]
    surfaces = touch["surface_continuation"]
    fields = touch["field_thirds"]
    goal_mouth = goal["goal_mouth"]
    lines = [
        "# Rival 2.0 Final-Checkpoint Behavioral Telemetry",
        "",
        f"Verdict: **{summary['verdict']}**.",
        "",
        (
            f"The single authorized evaluation completed {summary['evaluation']['completed_episodes']:,} "
            f"first episodes under ordinary stochastic final-policy self-play. The unchanged final "
            f"checkpoint controlled both cars; no training or historical opponent was used. It recorded "
            f"{summary['evaluation']['accepted_touch_entries']:,} unique accepted touch entries and "
            f"{summary['evaluation']['goal_terminated_episodes']:,} goals."
        ),
        "",
        "## Immediate touch direction",
        "",
        (
            f"Using the declared symmetric ±{DIRECTION_THRESHOLD_UU_PER_S:g} uu/s canonical longitudinal "
            f"threshold, forward touches were {direction['forward']['count']:,}/"
            f"{direction['forward']['denominator']:,} ({direction['forward']['fraction']:.6f}), "
            f"lateral-neutral touches were {direction['lateral_neutral']['count']:,}/"
            f"{direction['lateral_neutral']['denominator']:,} "
            f"({direction['lateral_neutral']['fraction']:.6f}), and backward touches were "
            f"{direction['backward']['count']:,}/{direction['backward']['denominator']:,} "
            f"({direction['backward']['fraction']:.6f}). These are descriptive directions, not quality labels."
        ),
        "",
        (
            "The raw continuous canonical post-touch Y velocity, longitudinal velocity delta, goal-center "
            "heading angles, actual net Y displacement, and 0.25/0.5/1/2-second continuation distributions "
            "are in `touch_trajectory_summary.json`. Instantaneous goal-plane intersections are explicitly "
            "projections only; an outside-mouth projection is not called a missed shot."
        ),
        "",
        "## Possession and continuation",
        "",
        (
            f"The same player made the next accepted touch after "
            f"{possession['same_player_next_touch']['count']:,}/"
            f"{possession['same_player_next_touch']['denominator']:,} touches "
            f"({possession['same_player_next_touch']['fraction']:.6f}); the opponent did so after "
            f"{possession['opponent_next_touch']['count']:,}/"
            f"{possession['opponent_next_touch']['denominator']:,} "
            f"({possession['opponent_next_touch']['fraction']:.6f}). Primitive possession chains are maximal "
            "consecutive accepted touches by one player."
        ),
        "",
        (
            f"Arena-contact continuation rates per touch were ground "
            f"{surfaces['ground']['fraction']:.6f}, side wall {surfaces['side_wall']['fraction']:.6f}, "
            f"backboard/goal structure {surfaces['backboard_goal_structure']['fraction']:.6f}, and ceiling "
            f"{surfaces['ceiling']['fraction']:.6f}. Categories come from the dominant axis of retained "
            "source collision normals. Wall and backboard paths are not treated as inaccurate."
        ),
        "",
        (
            f"Touch locations were defensive {fields['defensive']['fraction']:.6f}, middle "
            f"{fields['middle']['fraction']:.6f}, and attacking {fields['attacking']['fraction']:.6f} in "
            "the toucher's canonical frame."
        ),
        "",
        "## Goals and goal mouth",
        "",
        (
            f"An interpolated scoring-plane crossing was available for "
            f"{goal_mouth['interpolated_crossing_valid']['count']:,}/"
            f"{goal_mouth['interpolated_crossing_valid']['denominator']:,} goals. Of valid crossings, "
            f"{goal_mouth['inside_declared_goal_mouth']['count']:,}/"
            f"{goal_mouth['inside_declared_goal_mouth']['denominator']:,} were inside the declared "
            f"±{GOAL_HALF_WIDTH_UU} by 0..{GOAL_HEIGHT_UU} uu mouth."
        ),
        "",
        (
            "`goal_entry_summary.json` contains scoring side, last toucher and timing, event position/velocity, "
            "entry speed/angle, final-possession touch count, and exact arena-contact sequences. "
            "`goal_mouth_histogram.json` contains absolute and normalized X/Z distributions, exact descriptive "
            "third-bin boundaries, and the 2D heatmap counts."
        ),
        "",
        "## Authority and non-interference",
        "",
        f"- Authorized starting HEAD: `{EXPECTED_HEAD}`.",
        f"- Checkpoint SHA-256: `{EXPECTED_CHECKPOINT_SHA256}`.",
        f"- Reward contract: `{RIVAL2_REWARD_VERSION}` (unchanged).",
        f"- Evaluation seed: `{EVALUATION_SEED}`; worlds: `{EVALUATION_WORLDS:,}`.",
        "- Policy/physics rate: 30 Hz / 120 Hz; first completed episode per world.",
        "- Telemetry is a post-tick read-only GPU launch and has no controller, observation, reward, policy, reset, or dynamics output.",
        "- One technical recorder attempt was rejected before publication; the published replay used the identical frozen checkpoint, seeds, policy, and simulator, with only the read-only recorder corrected.",
        f"- Raw artifact: `{raw_artifact['path']}`; SHA-256 `{raw_artifact['sha256']}`; {raw_artifact['size_bytes']:,} bytes.",
        "",
        "All percentages in the JSON evidence include their explicit integer numerator and denominator. This is descriptive evidence only; no reward or behavior recommendation was implemented.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _publish(
    *,
    args: argparse.Namespace,
    configuration: dict[str, Any],
    run: dict[str, Any],
    raw: dict[str, np.ndarray],
) -> dict[str, Any]:
    touches = _flatten_touch_raw(raw)
    touches, chain_lengths, _chains = _derive_touch_data(touches, raw)
    touch_summary, possession_summary = _touch_summaries(touches, chain_lengths)
    goal_summary, mouth_summary, goal_raw = _goal_summaries(raw, touches)
    raw_artifact = _raw_artifact(
        output_dir=args.output_dir,
        raw_work_dir=args.raw_work_dir,
        configuration=configuration,
        touches=touches,
        goal_raw=goal_raw,
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "verdict": "PASS_GREEN",
        "scope": "single authorized final-45B-checkpoint behavioral evaluation",
        "configuration_path": (args.output_dir / "configuration.json").as_posix(),
        "evaluation": run,
        "headline": {
            "touches": touch_summary["touch_denominator"],
            "goals": goal_summary["goal_denominator"],
            "direction": touch_summary["direction"]["classes"],
            "same_player_next_touch": possession_summary["same_player_next_touch"],
            "opponent_next_touch": possession_summary["opponent_next_touch"],
            "surface_continuation": touch_summary["surface_continuation"],
            "touch_to_goal": touch_summary["touch_to_goal"],
            "final_touch_before_goal": touch_summary["final_touch_before_goal"],
            "goal_mouth": {
                "valid_crossing": mouth_summary["interpolated_crossing_valid"],
                "inside_mouth": mouth_summary["inside_declared_goal_mouth"],
            },
        },
        "raw_artifact": raw_artifact,
        "boundary": {
            "training_updates": 0,
            "reward_or_model_or_ppo_or_simulator_changes": 0,
            "viewer_built": False,
            "v0_6_started": False,
            "stopped_for_review": True,
        },
        "execution_provenance": configuration["execution_provenance"],
    }
    _write_json(args.output_dir / "configuration.json", configuration)
    _write_json(args.output_dir / "summary.json", summary)
    _write_json(args.output_dir / "touch_trajectory_summary.json", touch_summary)
    _write_json(args.output_dir / "goal_entry_summary.json", goal_summary)
    _write_json(args.output_dir / "goal_mouth_histogram.json", mouth_summary)
    _write_json(args.output_dir / "possession_summary.json", possession_summary)
    _report(
        path=Path("docs/RIVAL2_BEHAVIORAL_TELEMETRY.md"),
        summary=summary,
        touch=touch_summary,
        goal=goal_summary,
        possession=possession_summary,
        raw_artifact=raw_artifact,
    )
    artifact_paths = [
        args.output_dir / "configuration.json",
        args.output_dir / "summary.json",
        args.output_dir / "touch_trajectory_summary.json",
        args.output_dir / "goal_entry_summary.json",
        args.output_dir / "goal_mouth_histogram.json",
        args.output_dir / "possession_summary.json",
        Path(raw_artifact["path"]),
        Path("docs/RIVAL2_BEHAVIORAL_TELEMETRY.md"),
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "files": [
            {
                "path": path.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in artifact_paths
        ],
    }
    _write_json(args.output_dir / "artifact_manifest.json", manifest)
    return summary


def main() -> int:
    args = parse_args()
    checkpoint_path = CHECKPOINT.resolve()
    payload, checkpoint = _checkpoint_authority(checkpoint_path)
    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    if geometry.content_sha256 != "2239556BDC74D205CAA6E46A0F6E91FA2C6E4257E84D4F608BA775958B0A5538":
        raise RuntimeError("Soccar collision geometry identity mismatch")
    configuration = _configuration(
        args=args,
        checkpoint=checkpoint,
        geometry=geometry,
    )
    meshes = WarpArenaMeshes(geometry, args.device)
    print(
        f"starting the single authorized behavioral evaluation: worlds={EVALUATION_WORLDS} "
        f"seed={EVALUATION_SEED} checkpoint={checkpoint['sha256']}",
        flush=True,
    )
    run, raw = _run_evaluation(
        args=args,
        payload=payload,
        geometry=geometry,
        meshes=meshes,
    )
    summary = _publish(
        args=args,
        configuration=configuration,
        run=run,
        raw=raw,
    )
    print(json.dumps(summary["headline"], indent=2), flush=True)
    print("behavioral telemetry publication complete; stopping at the handoff boundary", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
