"""Run the authorized kickoff-free Rival 2.0 versus Nexto benchmark."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rivalsim.arena import ArenaGeometry
from rivalsim.behavioral_telemetry import (
    GOAL_HALF_WIDTH_UU,
    GOAL_HEIGHT_UU,
    GOAL_SCORING_PLANE_Y_UU,
)
from rivalsim.open_play import (
    DUEL_LIMIT_TICKS,
    DeviceContinuationBank,
    OpenPlayDuelRunner,
    build_face_mirror_maps,
    mirror_continuation_bank,
    mirror_involution_report,
    world_array_paths,
)
from rivalsim.rival2_contracts import OBS_DIM, RIVAL2_REWARD_VERSION
from rivalsim.rival2_env import Rival2Env, Rival2TensorBridge, Rival2WorldSim
from rivalsim.rival2_policy import (
    Rival2ActorCritic,
    Rival2PolicyConfig,
    sample_hybrid_action,
)
from third_party.nexto.adapter import MODEL_SHA256, NextoPolicyAdapter, NextoStateTensors

EXPECTED_HEAD = "0bcfc1864b4d8db75a39c6bc9deee8ed5dc5a32a"
CHECKPOINT = ROOT / "checkpoints" / "rival2" / "overnight" / "rival2_overnight_final_6h_resume.pt"
CHECKPOINT_SHA256 = "4DC158DC2A9D16B79FB5FE7D868E3B50928AB113B55DFCC753F3734F8D87372E"
NEXTO_COMMIT = "2e6ed7d6ed2b352e8ff529d4a12a0c9c70c28cca"
NEXTO_MODEL_SHA256 = "BF5343B5EEACAC6BF7CDB75DAC4A5C14BA0F94D820EAE75F00A211B6119D69FA"
COLLISION_ROOT = Path(r"G:\dev\RLBot-Rival\bot\collision_meshes")
OUTPUT_DIR = ROOT / "results" / "rival2" / "nexto_open_play"
DOC_PATH = ROOT / "docs" / "RIVAL2_NEXTO_OPEN_PLAY_RESULTS.md"
FIDELITY_PATH = ROOT / "results" / "rival2" / "nexto" / "fidelity.json"

SOURCE_WORLDS = 2_048
BASE_STATES = 4_096
DUELS = 16_384
MIN_CAPTURE_TICKS = 600
CAPTURE_TARGET_SPAN = 601
RIVAL_SOURCE_SEED = 2_026_082_701
NEXTO_SOURCE_SEED = 2_026_082_702
DUEL_SEED = 2_026_082_703
PROFILE_TICKS = 8
MAX_HARVEST_DECISIONS = 20_000
PHYSICS_HZ = 120
FIELD_BACK_WALL_Y_UU = 5_120.0

SOURCE_NAMES = np.asarray(("rival_stochastic_self_play", "nexto_deterministic_self_play"))
SIDE_NAMES = np.asarray(("Blue", "Orange"))
ROLE_NAMES = np.asarray(("original_blue_car", "original_orange_car"))
OUTCOME_NAMES = np.asarray(("Nexto", "Draw", "Rival"))

NEUTRALIZED_RESTORE_FIELDS = {
    "rival2.interval_tick",
    "rival2.goal_latched",
    "rival2.terminated",
    "rival2.truncated",
    "rival2.reset_mask",
    "rival2.kickoff_indicator",
    "rival2.touch_count",
    "rival2.touch_contact_latched",
    "rival2.demo_by_count",
    "rival2.demoed_event",
    "rival2.reward",
    "rival2.previous_action",
    "rival2.scoring_team_latched",
    "rival2.ball_y_before",
    "rival2.ball_y_after",
    "lifecycle.goal_scored",
    "lifecycle.kickoff_reset",
    "lifecycle.full_reset",
    "lifecycle.reset_required",
    "lifecycle.terminated",
    "lifecycle.truncated",
    "lifecycle.ball_scored_last",
    "lifecycle.scoring_team",
    "lifecycle.auto_kickoff",
}


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True, capture_output=True, text=True,
        encoding="utf-8",
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


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
        return {"count": 0, "min": None, "p25": None, "median": None, "mean": None, "p75": None, "max": None}
    return {
        "count": int(source.size),
        "min": float(source.min()),
        "p25": float(np.percentile(source, 25)),
        "median": float(np.median(source)),
        "mean": float(source.mean()),
        "p75": float(np.percentile(source, 75)),
        "max": float(source.max()),
    }


def _load_rival_policy(device: torch.device) -> tuple[Rival2ActorCritic, dict[str, Any]]:
    if _sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise RuntimeError("frozen Rival checkpoint SHA-256 mismatch")
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    config = Rival2PolicyConfig(**payload["policy_config"])
    checks = {
        "format": payload.get("format") == "RIVAL2_CHECKPOINT_V1",
        "reward": payload.get("reward_version") == RIVAL2_REWARD_VERSION,
        "config_hash": payload.get("policy_config_hash") == config.content_hash,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Rival checkpoint contract failed: {checks}")
    model = Rival2ActorCritic(config).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    identity = {
        "path": CHECKPOINT.relative_to(ROOT).as_posix(),
        "sha256": CHECKPOINT_SHA256,
        "size_bytes": CHECKPOINT.stat().st_size,
        "policy_version": int(payload["policy_version"]),
        "iteration": int(payload["iteration"]),
        "total_agent_samples": int(payload["total_agent_samples"]),
        "reward_version": payload["reward_version"],
        "policy_config": asdict(config),
        "policy_config_hash": config.content_hash,
    }
    del payload
    return model, identity


def _prospective_targets(
    world_index: torch.Tensor, reset_count: torch.Tensor, seed: int
) -> torch.Tensor:
    # Integer-only prospective rule.  It is fixed before observing any state,
    # action, advantage, or outcome; a reset merely establishes the next episode.
    mixed = (
        world_index.to(torch.int64) * 1_103_515_245
        + reset_count.to(torch.int64) * 12_345
        + int(seed)
    ) & 0x7FFF_FFFF
    return (MIN_CAPTURE_TICKS + mixed.remainder(CAPTURE_TARGET_SPAN)).to(torch.int32)


def _eligibility(
    bridge: Rival2TensorBridge,
    touch_since_reset: torch.Tensor,
    kickoff_active: torch.Tensor,
    target_age: torch.Tensor,
) -> torch.Tensor:
    views = bridge.views
    ball_y = views["ball_pos"].reshape(-1, 3)[:, 1]
    active = views["car_is_demoed"].reshape(-1, 2).eq(0).all(dim=1)
    no_pending_reset = ~views["rival2.reset_mask"].to(torch.bool)
    lifecycle = bridge.sim.lifecycle
    no_pending_reset &= wp.to_torch(lifecycle.goal_scored).eq(0)
    no_pending_reset &= wp.to_torch(lifecycle.kickoff_reset).eq(0)
    no_pending_reset &= wp.to_torch(lifecycle.full_reset).eq(0)
    no_pending_reset &= wp.to_torch(lifecycle.reset_required).eq(0)
    return (
        (views["rival2.episode_ticks"].to(torch.int32) >= target_age)
        & touch_since_reset
        & no_pending_reset
        & (ball_y.abs() < float(GOAL_SCORING_PLANE_Y_UU))
        & active
        & ~kickoff_active
    )


def _update_episode_trackers(
    reset_mask: torch.Tensor,
    interval_touch: torch.Tensor,
    ball_y: torch.Tensor,
    touch_since_reset: torch.Tensor,
    kickoff_active: torch.Tensor,
    reset_count: torch.Tensor,
    world_index: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    touch_since_reset.logical_or_(interval_touch)
    kickoff_active.logical_and_(~((ball_y != 0.0) | interval_touch))
    touch_since_reset.copy_(torch.where(reset_mask, torch.zeros_like(touch_since_reset), touch_since_reset))
    kickoff_active.copy_(torch.where(reset_mask, torch.ones_like(kickoff_active), kickoff_active))
    reset_count.add_(reset_mask.to(reset_count.dtype))
    return _prospective_targets(world_index, reset_count, seed)


def _harvest_rival(
    collision_root: Path, device: str
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any], dict[str, Any]]:
    print(f"harvest Rival: initializing {SOURCE_WORLDS:,} worlds", flush=True)
    kickoff_selector = np.arange(SOURCE_WORLDS, dtype=np.int32) % 5
    env = Rival2Env(
        SOURCE_WORLDS, str(collision_root), device=device, seed=RIVAL_SOURCE_SEED,
        kickoff_selector=kickoff_selector, car_lifecycle_seed=RIVAL_SOURCE_SEED,
    )
    model, identity = _load_rival_policy(env.device)
    generator = torch.Generator(device=env.device)
    generator.manual_seed(RIVAL_SOURCE_SEED)
    index = torch.arange(SOURCE_WORLDS, device=env.device)
    reset_count = torch.zeros(SOURCE_WORLDS, dtype=torch.int32, device=env.device)
    touch_since = torch.zeros(SOURCE_WORLDS, dtype=torch.bool, device=env.device)
    kickoff = torch.ones(SOURCE_WORLDS, dtype=torch.bool, device=env.device)
    target = _prospective_targets(index, reset_count, RIVAL_SOURCE_SEED)
    bank = DeviceContinuationBank(env.world)
    env.reset_transfer_counters()
    started = time.perf_counter()
    decisions = 0
    for decision in range(MAX_HARVEST_DECISIONS):
        decisions = decision + 1
        with torch.inference_mode():
            actor, _ = model(env.observation.reshape(-1, OBS_DIM))
            action = sample_hybrid_action(actor, generator=generator).action.reshape(SOURCE_WORLDS, 2, 8)
        transition = env.step(action)
        interval_touch = (transition.transition_observation[..., 176] > 0.5).any(dim=1)
        target = _update_episode_trackers(
            transition.reset_mask, interval_touch, env.bridge.views["ball_pos"].reshape(-1, 3)[:, 1],
            touch_since, kickoff, reset_count, index, RIVAL_SOURCE_SEED,
        )
        eligible = _eligibility(env.bridge, touch_since, kickoff, target)
        bank.capture(env.world, eligible, env.world.tick_count)
        if decisions % 32 == 0:
            complete = bank.complete_count()
            print(f"harvest Rival: decision {decisions:,}, captured {complete:,}/{SOURCE_WORLDS:,}", flush=True)
            if complete == SOURCE_WORLDS:
                break
    if bank.complete_count() != SOURCE_WORLDS:
        raise RuntimeError(f"Rival state bank incomplete after {decisions} decisions")
    values, ticks = bank.export()
    report = {
        "source": SOURCE_NAMES[0], "states": SOURCE_WORLDS, "seed": RIVAL_SOURCE_SEED,
        "action_mode": "stochastic final-45B Rival self-play", "decisions": decisions,
        "physics_ticks": decisions * 4, "wall_seconds": time.perf_counter() - started,
        "episode_resets_before_capture": _distribution(reset_count.detach().cpu().numpy()),
        "hot_path_h2d_bytes": int(env.world.host_to_device_bytes),
        "hot_path_d2h_bytes_before_bank_export": int(env.world.device_to_host_bytes),
    }
    del env, model, bank
    gc.collect(); torch.cuda.empty_cache()
    return values, ticks, report, identity


def _harvest_nexto(
    collision_root: Path, device: str
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    print(f"harvest Nexto: initializing {SOURCE_WORLDS:,} worlds", flush=True)
    kickoff_selector = np.arange(SOURCE_WORLDS, dtype=np.int32) % 5
    world = Rival2WorldSim(
        SOURCE_WORLDS, str(collision_root), device=device, seed=NEXTO_SOURCE_SEED,
        kickoff_selector=kickoff_selector, car_lifecycle_seed=NEXTO_SOURCE_SEED,
    )
    warp_stream = wp.Stream(world.device)
    torch_stream = wp.stream_to_torch(warp_stream)
    torch.cuda.set_stream(torch_stream)
    wp.set_stream(warp_stream, device=world.device, sync=False)
    bridge = Rival2TensorBridge(world)
    state = NextoStateTensors.from_bridge(bridge)
    adapters = (NextoPolicyAdapter(SOURCE_WORLDS, device=device), NextoPolicyAdapter(SOURCE_WORLDS, device=device))
    adapters[0].set_player_index(torch.zeros(SOURCE_WORLDS, dtype=torch.long, device=device))
    adapters[1].set_player_index(torch.ones(SOURCE_WORLDS, dtype=torch.long, device=device))
    index = torch.arange(SOURCE_WORLDS, device=device)
    reset_count = torch.zeros(SOURCE_WORLDS, dtype=torch.int32, device=device)
    touch_since = torch.zeros(SOURCE_WORLDS, dtype=torch.bool, device=device)
    kickoff = torch.ones(SOURCE_WORLDS, dtype=torch.bool, device=device)
    target = _prospective_targets(index, reset_count, NEXTO_SOURCE_SEED)
    actions = torch.zeros((SOURCE_WORLDS, 2, 8), dtype=torch.float32, device=device)
    bank = DeviceContinuationBank(world)
    world.capture_graph(block_ticks=1)
    world.reset_transfer_counters()
    started = time.perf_counter()
    host_tick = 0
    decisions = 0
    for decision in range(MAX_HARVEST_DECISIONS):
        decisions = decision + 1
        world.begin_decision()
        for _ in range(4):
            actions[:, 0] = adapters[0].tick_action(state, kickoff)[0]
            actions[:, 1] = adapters[1].tick_action(state, kickoff)[0]
            bridge.set_actions(actions)
            world.step_graph(1)
            host_tick += 1
            kickoff.logical_and_(wp.to_torch(world.state.ball_pos).reshape(-1, 3)[:, 1].eq(0.0))
        interval_touch = wp.to_torch(world.rival2.touch_count).reshape(-1, 2).gt(0).any(dim=1)
        reset_mask = wp.to_torch(world.rival2.reset_mask).to(torch.bool).clone()
        ball_y = wp.to_torch(world.state.ball_pos).reshape(-1, 3)[:, 1]
        target = _update_episode_trackers(
            reset_mask, interval_touch, ball_y, touch_since, kickoff, reset_count,
            index, NEXTO_SOURCE_SEED,
        )
        world.apply_interval_resets()
        adapters[0].notify_kickoff(reset_mask)
        adapters[1].notify_kickoff(reset_mask)
        eligible = _eligibility(bridge, touch_since, kickoff, target)
        bank.capture(world, eligible, world.tick_count)
        if decisions % 32 == 0:
            complete = bank.complete_count()
            print(f"harvest Nexto: decision {decisions:,}, captured {complete:,}/{SOURCE_WORLDS:,}", flush=True)
            if complete == SOURCE_WORLDS:
                break
    if bank.complete_count() != SOURCE_WORLDS:
        raise RuntimeError(f"Nexto state bank incomplete after {decisions} decisions")
    values, ticks = bank.export()
    report = {
        "source": SOURCE_NAMES[1], "states": SOURCE_WORLDS, "seed": NEXTO_SOURCE_SEED,
        "action_mode": "deterministic pinned Nexto self-play including stock kickoff controller only during source rollout",
        "decisions": decisions, "physics_ticks": host_tick,
        "wall_seconds": time.perf_counter() - started,
        "episode_resets_before_capture": _distribution(reset_count.detach().cpu().numpy()),
        "nexto_inference_calls_per_adapter": [int(item.inference_calls) for item in adapters],
        "hot_path_h2d_bytes": int(world.host_to_device_bytes),
        "hot_path_d2h_bytes_before_bank_export": int(world.device_to_host_bytes),
    }
    del world, bridge, state, adapters, bank
    gc.collect(); torch.cuda.empty_cache()
    return values, ticks, report


def _combine_banks(
    first: dict[str, np.ndarray], second: dict[str, np.ndarray],
    first_ticks: np.ndarray, second_ticks: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    if first.keys() != second.keys():
        raise RuntimeError("source state-bank schemas differ")
    values = {path: np.concatenate((first[path], second[path]), axis=0) for path in first}
    ticks = np.concatenate((first_ticks, second_ticks)).astype(np.int32)
    source = np.concatenate((np.zeros(SOURCE_WORLDS, dtype=np.int8), np.ones(SOURCE_WORLDS, dtype=np.int8)))
    return values, ticks, source


def _save_state_bank(
    values: dict[str, np.ndarray], ticks: np.ndarray, source: np.ndarray, output: Path
) -> tuple[dict[str, str], str]:
    field_keys = {path: f"field_{index:03d}" for index, path in enumerate(sorted(values))}
    payload: dict[str, np.ndarray] = {field_keys[path]: values[path] for path in sorted(values)}
    payload["capture_tick"] = ticks
    payload["source_policy"] = source
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    return field_keys, _sha256(output)


def _state_bank_statistics(values: dict[str, np.ndarray], source: np.ndarray) -> dict[str, Any]:
    ball_pos = values["state.ball_pos"].reshape(BASE_STATES, 3)
    ball_vel = values["state.ball_vel"].reshape(BASE_STATES, 3)
    car_pos = values["state.car_pos"].reshape(BASE_STATES, 2, 3)
    boost = values["state.boost"].reshape(BASE_STATES, 2)
    distances = np.linalg.norm(car_pos - ball_pos[:, None, :], axis=2)
    thirds = np.where(
        ball_pos[:, 1] < -FIELD_BACK_WALL_Y_UU / 3.0, "blue_defensive_third",
        np.where(ball_pos[:, 1] > FIELD_BACK_WALL_Y_UU / 3.0, "orange_defensive_third", "midfield_third"),
    )
    def stats(rows: np.ndarray) -> dict[str, Any]:
        return {
            "states": int(rows.sum()),
            "ball_x_uu": _distribution(ball_pos[rows, 0]),
            "ball_y_uu": _distribution(ball_pos[rows, 1]),
            "ball_height_uu": _distribution(ball_pos[rows, 2]),
            "ball_speed_uu_per_second": _distribution(np.linalg.norm(ball_vel[rows], axis=1)),
            "blue_car_ball_distance_uu": _distribution(distances[rows, 0]),
            "orange_car_ball_distance_uu": _distribution(distances[rows, 1]),
            "blue_boost": _distribution(boost[rows, 0]),
            "orange_boost": _distribution(boost[rows, 1]),
            "field_thirds": {name: int((thirds[rows] == name).sum()) for name in np.unique(thirds)},
        }
    return {
        "overall": stats(np.ones(BASE_STATES, dtype=bool)),
        SOURCE_NAMES[0]: stats(source == 0),
        SOURCE_NAMES[1]: stats(source == 1),
    }


def _four_way_metadata(
    values: dict[str, np.ndarray], source: np.ndarray
) -> dict[str, np.ndarray]:
    ball = values["state.ball_pos"].reshape(BASE_STATES, 3)
    cars = values["state.car_pos"].reshape(BASE_STATES, 2, 3)
    boost = values["state.boost"].reshape(BASE_STATES, 2)
    mirror_ball = ball * np.asarray((-1.0, -1.0, 1.0), dtype=np.float32)
    mirror_cars = cars[:, ::-1] * np.asarray((-1.0, -1.0, 1.0), dtype=np.float32)
    mirror_boost = boost[:, ::-1]
    variant = np.tile(np.arange(4, dtype=np.int8), BASE_STATES)
    base_id = np.repeat(np.arange(BASE_STATES, dtype=np.int32), 4)
    mirrored = variant >= 2
    rival_side = np.tile(np.asarray((0, 1, 0, 1), dtype=np.int8), BASE_STATES)
    role = np.tile(np.asarray((0, 1, 1, 0), dtype=np.int8), BASE_STATES)
    initial_ball = np.empty((BASE_STATES, 4, 3), dtype=np.float32)
    initial_cars = np.empty((BASE_STATES, 4, 2, 3), dtype=np.float32)
    initial_boost = np.empty((BASE_STATES, 4, 2), dtype=np.float32)
    initial_ball[:, :2] = ball[:, None]
    initial_ball[:, 2:] = mirror_ball[:, None]
    initial_cars[:, :2] = cars[:, None]
    initial_cars[:, 2:] = mirror_cars[:, None]
    initial_boost[:, :2] = boost[:, None]
    initial_boost[:, 2:] = mirror_boost[:, None]
    initial_ball = initial_ball.reshape(DUELS, 3)
    initial_cars = initial_cars.reshape(DUELS, 2, 3)
    initial_boost = initial_boost.reshape(DUELS, 2)
    distances = np.linalg.norm(initial_cars - initial_ball[:, None, :], axis=2)
    rival_distance = distances[np.arange(DUELS), rival_side]
    nexto_distance = distances[np.arange(DUELS), 1 - rival_side]
    closest = np.where(rival_distance < nexto_distance, "Rival", np.where(rival_distance > nexto_distance, "Nexto", "exact_tie"))
    rival_boost = initial_boost[np.arange(DUELS), rival_side]
    nexto_boost = initial_boost[np.arange(DUELS), 1 - rival_side]
    boost_advantage = rival_boost - nexto_boost
    field_third = np.where(
        initial_ball[:, 1] < -FIELD_BACK_WALL_Y_UU / 3.0, "blue_defensive_third",
        np.where(initial_ball[:, 1] > FIELD_BACK_WALL_Y_UU / 3.0, "orange_defensive_third", "midfield_third"),
    )
    height_bin = np.where(initial_ball[:, 2] < 300.0, "low_0_to_300_uu", np.where(initial_ball[:, 2] < 1_000.0, "middle_300_to_1000_uu", "high_1000_plus_uu"))
    boost_bin = np.where(boost_advantage < -25.0, "Rival_trails_by_more_than_25", np.where(boost_advantage > 25.0, "Rival_leads_by_more_than_25", "within_25"))
    return {
        "duel_id": np.arange(DUELS, dtype=np.int32),
        "base_state_id": base_id,
        "variant": variant,
        "mirrored": mirrored,
        "rival_side": rival_side,
        "source": np.repeat(source, 4),
        "original_role": role,
        "initial_ball": initial_ball,
        "initial_cars": initial_cars,
        "initial_boost": initial_boost,
        "initial_rival_distance": rival_distance,
        "initial_nexto_distance": nexto_distance,
        "initial_closest_policy": closest,
        "initial_rival_boost": rival_boost,
        "initial_nexto_boost": nexto_boost,
        "initial_boost_advantage": boost_advantage,
        "field_third": field_third,
        "height_bin": height_bin,
        "boost_bin": boost_bin,
    }


def _restore_fidelity_report(
    runner: OpenPlayDuelRunner,
    values: dict[str, np.ndarray], capture_tick: np.ndarray,
    face_map: np.ndarray, mesh_map: np.ndarray,
) -> dict[str, Any]:
    mirrored = mirror_continuation_bank(values, face_map, mesh_map)
    failures: list[dict[str, Any]] = []
    checked = 0
    maximum = 0.0
    destinations = world_array_paths(runner.world)
    for path, destination in destinations.items():
        if path in NEUTRALIZED_RESTORE_FIELDS or path == "world._dynamic_proxy_cell":
            continue
        source = values[path]
        rows = np.empty((BASE_STATES, 4, source.shape[1]), dtype=source.dtype)
        rows[:, 0] = source
        rows[:, 1] = source
        rows[:, 2] = mirrored[path]
        rows[:, 3] = mirrored[path]
        expected_host = rows.reshape(DUELS, -1)
        if path in {"car_ball.last_extra_impulse_tick", "car_ball_b.last_extra_impulse_tick"}:
            source_ticks = np.repeat(capture_tick[:, None], 4, axis=1).reshape(-1, 1)
            valid = expected_host >= 0
            expected_host = expected_host.copy()
            offsets = np.broadcast_to(1_000_000 - source_ticks, expected_host.shape)
            expected_host[valid] += offsets[valid]
        actual = wp.to_torch(destination).reshape(DUELS, -1)
        expected = torch.from_numpy(np.ascontiguousarray(expected_host)).to(device=actual.device, dtype=actual.dtype)
        exact = bool(torch.equal(actual, expected))
        error = 0.0
        if actual.dtype.is_floating_point:
            error = float((actual - expected).abs().max().item())
            maximum = max(maximum, error)
        if not exact:
            failures.append({"field": path, "max_abs_error": error})
        checked += 1
        del expected
    previous_action = wp.to_torch(runner.world.rival2.previous_action)
    neutral_memory_exact = bool(torch.count_nonzero(previous_action).item() == 0)
    report = {
        "fields_exactly_checked": checked,
        "excluded_neutral_boundary_fields": sorted(NEUTRALIZED_RESTORE_FIELDS),
        "dynamic_proxy_cell_excluded_reason": "recomputed from restored rigid state by the exact source broadphase kernel",
        "maximum_float_abs_error": maximum,
        "failed_fields": failures,
        "neutral_rival_previous_action_exact_zero": neutral_memory_exact,
        "neutral_nexto_previous_action_exact_zero": bool(torch.count_nonzero(runner.nexto.previous_action).item() == 0),
        "pass": not failures and neutral_memory_exact and bool(torch.count_nonzero(runner.nexto.previous_action).item() == 0),
    }
    return report


def _outcome_codes(raw: dict[str, np.ndarray], metadata: dict[str, np.ndarray]) -> np.ndarray:
    done = raw["done"] != 0
    winner = raw["winner"]
    rival_side = metadata["rival_side"]
    return np.where(~done, 0, np.where(winner == rival_side, 1, -1)).astype(np.int8)


def _outcome_summary(
    outcome: np.ndarray, raw: dict[str, np.ndarray], rows: np.ndarray
) -> dict[str, Any]:
    rows = np.asarray(rows, dtype=np.int64)
    selected = outcome[rows]
    rival = int((selected == 1).sum())
    nexto = int((selected == -1).sum())
    draws = int((selected == 0).sum())
    decisive = rival + nexto
    goal_seconds = raw["goal_tick"][rows].astype(np.float64) / PHYSICS_HZ
    return {
        "duels": int(rows.size),
        "rival_wins": rival,
        "nexto_wins": nexto,
        "draws": draws,
        "decisive_duel_rival_win_rate": _ratio(rival, decisive),
        "all_duel_rival_win_fraction": _ratio(rival, int(rows.size)),
        "time_to_goal_seconds_by_winner": {
            "Rival": _distribution(goal_seconds[selected == 1]),
            "Nexto": _distribution(goal_seconds[selected == -1]),
        },
    }


def _stratify_outcomes(
    outcome: np.ndarray, raw: dict[str, np.ndarray], metadata: dict[str, np.ndarray]
) -> dict[str, Any]:
    all_rows = np.arange(DUELS)
    specifications: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "rival_side": (metadata["rival_side"], SIDE_NAMES),
        "state_orientation": (metadata["mirrored"].astype(np.int8), np.asarray(("original", "mirrored"))),
        "source_policy": (metadata["source"], SOURCE_NAMES),
        "initial_physical_role_inherited_by_rival": (metadata["original_role"], ROLE_NAMES),
    }
    result: dict[str, Any] = {"overall": _outcome_summary(outcome, raw, all_rows)}
    for dimension, (labels, names) in specifications.items():
        result[dimension] = {
            str(name): _outcome_summary(outcome, raw, all_rows[labels == index])
            for index, name in enumerate(names)
        }
    for dimension in ("field_third", "height_bin", "initial_closest_policy", "boost_bin"):
        labels = metadata[dimension]
        result[dimension] = {
            str(label): _outcome_summary(outcome, raw, all_rows[labels == label])
            for label in sorted(np.unique(labels))
        }
    return result


def _paired_summary(outcome: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    family = outcome.reshape(BASE_STATES, 4)
    draw_count = (family == 0).sum(axis=1)
    rival_wins = (family == 1).sum(axis=1)
    nexto_wins = (family == -1).sum(axis=1)
    complete = draw_count == 0
    pattern = ["/".join(OUTCOME_NAMES[row + 1].tolist()) for row in family]
    report = {
        "families": BASE_STATES,
        "complete_without_draws": int(complete.sum()),
        "draws_prevent_complete_four_way_decision": int((~complete).sum()),
        "complete_family_rival_win_count": {
            f"Rival_wins_{wins}_of_4": int((complete & (rival_wins == wins)).sum())
            for wins in range(4, -1, -1)
        },
        "all_family_rival_win_count_including_drawn_families": {
            f"Rival_wins_{wins}_of_4": int((rival_wins == wins).sum())
            for wins in range(4, -1, -1)
        },
        "exact_four_outcome_pattern_histogram": dict(sorted(Counter(pattern).items())),
    }
    ledger = [
        {
            "base_state_id": index,
            "outcomes_variant_0_to_3": [str(OUTCOME_NAMES[code + 1]) for code in family[index]],
            "rival_wins": int(rival_wins[index]),
            "nexto_wins": int(nexto_wins[index]),
            "draws": int(draw_count[index]),
            "complete_four_way_decision": bool(complete[index]),
        }
        for index in range(BASE_STATES)
    ]
    return report, ledger


def _direction_counts(array: np.ndarray) -> dict[str, Any]:
    totals = np.asarray(array, dtype=np.int64).sum(axis=0)
    denominator = int(totals.sum())
    return {
        label: _ratio(int(totals[index]), denominator)
        for index, label in enumerate(("backward", "neutral", "forward"))
    }


def _goal_mouth(raw: dict[str, np.ndarray], rows: np.ndarray, side: np.ndarray) -> dict[str, Any]:
    scored = (raw["done"][rows] != 0) & (raw["winner"][rows] == side)
    goal_rows = rows[scored]
    valid = raw["goal_entry_valid"][goal_rows] != 0
    x = raw["goal_entry_x"][goal_rows][valid]
    z = raw["goal_entry_z"][goal_rows][valid]
    inside = (np.abs(x) <= GOAL_HALF_WIDTH_UU) & (z >= 0.0) & (z <= GOAL_HEIGHT_UU)
    x_edges = np.linspace(-GOAL_HALF_WIDTH_UU, GOAL_HALF_WIDTH_UU, 11)
    z_edges = np.linspace(0.0, GOAL_HEIGHT_UU, 7)
    histogram, _, _ = np.histogram2d(x[inside], z[inside], bins=(x_edges, z_edges))
    return {
        "goals": int(goal_rows.size),
        "interpolated_crossing_valid": _ratio(int(valid.sum()), int(goal_rows.size)),
        "inside_declared_goal_mouth": _ratio(int(inside.sum()), int(valid.sum())),
        "canonical_x_uu": _distribution(x),
        "z_uu": _distribution(z),
        "histogram_x_by_z": {"x_edges_uu": x_edges.tolist(), "z_edges_uu": z_edges.tolist(), "counts": histogram.astype(int).tolist()},
    }


def _policy_telemetry(
    raw: dict[str, np.ndarray], metadata: dict[str, np.ndarray], rows: np.ndarray, policy: str
) -> dict[str, Any]:
    rows = np.asarray(rows, dtype=np.int64)
    rival_side = metadata["rival_side"][rows].astype(np.int64)
    side = rival_side if policy == "Rival" else 1 - rival_side
    opponent = 1 - side
    car_index = (rows, side)
    opponent_index = (rows, opponent)
    touches = raw["touch_count"][car_index].astype(np.int64)
    opponent_touches = raw["touch_count"][opponent_index].astype(np.int64)
    possession_total = raw["possession_total"][car_index].astype(np.int64)
    finalized = raw["displacement_count"][car_index].sum(axis=1)
    first = raw["first_toucher"][rows]
    first_resolved = first >= 0
    policy_first = first_resolved & (first == side)
    policy_goals = (raw["done"][rows] != 0) & (raw["winner"][rows] == side)
    final_touch_valid = policy_goals & (raw["last_toucher"][rows] >= 0)
    final_touch_policy = final_touch_valid & (raw["last_toucher"][rows] == side)
    final_touch_opponent = final_touch_valid & (raw["last_toucher"][rows] == opponent)
    final_ticks = raw["final_touch_to_goal_ticks"][rows]
    return {
        "duels": int(rows.size),
        "touches": int(touches.sum()),
        "touch_share": _ratio(int(touches.sum()), int(touches.sum() + opponent_touches.sum())),
        "first_accepted_touch_after_restore": _ratio(int(policy_first.sum()), int(first_resolved.sum())),
        "next_touch_possession": {
            "total_resolved": int(possession_total.sum()),
            "same_player_retention": _ratio(int(raw["possession_same"][car_index].sum()), int(possession_total.sum())),
            "opponent_handoff": _ratio(int(raw["possession_opponent"][car_index].sum()), int(possession_total.sum())),
        },
        "immediate_touch_direction": _direction_counts(raw["direction_count"][car_index]),
        "net_ball_displacement_before_next_touch_or_goal": _direction_counts(raw["displacement_count"][car_index]),
        "wall_continuation": _ratio(int(raw["wall_continuation_count"][car_index].sum()), int(finalized.sum())),
        "backboard_continuation": _ratio(int(raw["backboard_continuation_count"][car_index].sum()), int(finalized.sum())),
        "demos": int(raw["demo_count"][car_index].sum()),
        "goal_entry": _goal_mouth(raw, rows, side),
        "final_touch_to_goal": {
            "policy_goals": int(policy_goals.sum()),
            "last_toucher_available": int(final_touch_valid.sum()),
            "last_touch_by_scoring_policy": int(final_touch_policy.sum()),
            "last_touch_by_opponent": int(final_touch_opponent.sum()),
            "seconds_when_available": _distribution(final_ticks[final_touch_valid] / PHYSICS_HZ),
            "scorer_matches_last_toucher": _ratio(int(raw["scorer_matches_last_toucher"][rows][policy_goals].sum()), int(final_touch_valid.sum())),
        },
    }


def _behavioral_summary(raw: dict[str, np.ndarray], metadata: dict[str, np.ndarray]) -> dict[str, Any]:
    all_rows = np.arange(DUELS)
    result: dict[str, Any] = {
        "overall": {
            policy: _policy_telemetry(raw, metadata, all_rows, policy)
            for policy in ("Rival", "Nexto")
        }
    }
    for side_index, side_name in enumerate(SIDE_NAMES):
        rows = all_rows[metadata["rival_side"] == side_index]
        result[f"Rival_as_{side_name}"] = {
            policy: _policy_telemetry(raw, metadata, rows, policy)
            for policy in ("Rival", "Nexto")
        }
    return result


def _write_duel_ledger(
    path: Path, outcome: np.ndarray, raw: dict[str, np.ndarray], metadata: dict[str, np.ndarray]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "duel_id", "base_state_id", "variant", "mirror", "source_distribution",
        "rival_side", "initial_physical_role_inherited_by_rival", "initial_ball_x_uu",
        "initial_ball_y_uu", "initial_ball_z_uu", "initial_field_third", "initial_height_bin",
        "initial_closest_policy", "initial_rival_boost", "initial_nexto_boost",
        "initial_boost_advantage", "initial_boost_advantage_bin", "outcome",
        "time_to_goal_ticks", "time_to_goal_seconds", "physical_scoring_team",
        "first_toucher_physical_team", "last_toucher_physical_team", "final_touch_to_goal_ticks",
        "goal_entry_valid", "canonical_goal_entry_x_uu", "goal_entry_z_uu",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in range(DUELS):
            goal_tick = int(raw["goal_tick"][row])
            scorer = int(raw["winner"][row])
            first = int(raw["first_toucher"][row])
            last = int(raw["last_toucher"][row])
            writer.writerow({
                "duel_id": row,
                "base_state_id": int(metadata["base_state_id"][row]),
                "variant": int(metadata["variant"][row]),
                "mirror": int(metadata["mirrored"][row]),
                "source_distribution": str(SOURCE_NAMES[metadata["source"][row]]),
                "rival_side": str(SIDE_NAMES[metadata["rival_side"][row]]),
                "initial_physical_role_inherited_by_rival": str(ROLE_NAMES[metadata["original_role"][row]]),
                "initial_ball_x_uu": float(metadata["initial_ball"][row, 0]),
                "initial_ball_y_uu": float(metadata["initial_ball"][row, 1]),
                "initial_ball_z_uu": float(metadata["initial_ball"][row, 2]),
                "initial_field_third": str(metadata["field_third"][row]),
                "initial_height_bin": str(metadata["height_bin"][row]),
                "initial_closest_policy": str(metadata["initial_closest_policy"][row]),
                "initial_rival_boost": float(metadata["initial_rival_boost"][row]),
                "initial_nexto_boost": float(metadata["initial_nexto_boost"][row]),
                "initial_boost_advantage": float(metadata["initial_boost_advantage"][row]),
                "initial_boost_advantage_bin": str(metadata["boost_bin"][row]),
                "outcome": str(OUTCOME_NAMES[outcome[row] + 1]),
                "time_to_goal_ticks": "" if goal_tick < 0 else goal_tick,
                "time_to_goal_seconds": "" if goal_tick < 0 else goal_tick / PHYSICS_HZ,
                "physical_scoring_team": "" if scorer < 0 else str(SIDE_NAMES[scorer]),
                "first_toucher_physical_team": "" if first < 0 else str(SIDE_NAMES[first]),
                "last_toucher_physical_team": "" if last < 0 else str(SIDE_NAMES[last]),
                "final_touch_to_goal_ticks": "" if raw["final_touch_to_goal_ticks"][row] < 0 else int(raw["final_touch_to_goal_ticks"][row]),
                "goal_entry_valid": int(raw["goal_entry_valid"][row]),
                "canonical_goal_entry_x_uu": float(raw["goal_entry_x"][row]),
                "goal_entry_z_uu": float(raw["goal_entry_z"][row]),
            })


def _write_family_ledger(path: Path, ledger: list[dict[str, Any]], source: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = (
            "base_state_id", "source_distribution", "original_rival_blue", "original_rival_orange",
            "mirror_rival_blue", "mirror_rival_orange", "rival_wins", "nexto_wins", "draws",
            "complete_four_way_decision",
        )
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for item in ledger:
            outcomes = item["outcomes_variant_0_to_3"]
            writer.writerow({
                "base_state_id": item["base_state_id"],
                "source_distribution": str(SOURCE_NAMES[source[item["base_state_id"]]]),
                "original_rival_blue": outcomes[0],
                "original_rival_orange": outcomes[1],
                "mirror_rival_blue": outcomes[2],
                "mirror_rival_orange": outcomes[3],
                "rival_wins": item["rival_wins"], "nexto_wins": item["nexto_wins"],
                "draws": item["draws"], "complete_four_way_decision": int(item["complete_four_way_decision"]),
            })


def _outcome_line(value: dict[str, Any]) -> str:
    decisive = value["decisive_duel_rival_win_rate"]["fraction"]
    all_fraction = value["all_duel_rival_win_fraction"]["fraction"]
    return (
        f"Rival {value['rival_wins']:,}, Nexto {value['nexto_wins']:,}, draws {value['draws']:,}; "
        f"decisive Rival win rate {100.0 * decisive:.3f}%, all-duel Rival win fraction {100.0 * all_fraction:.3f}%"
    )


def _write_document(
    summary: dict[str, Any], outcomes: dict[str, Any], paired: dict[str, Any], behavior: dict[str, Any]
) -> None:
    overall = outcomes["overall"]
    blue = outcomes["rival_side"]["Blue"]
    orange = outcomes["rival_side"]["Orange"]
    rival_behavior = behavior["overall"]["Rival"]
    nexto_behavior = behavior["overall"]["Nexto"]
    lines = [
        "# Rival 2.0 vs pinned public Nexto — kickoff-free open play",
        "",
        f"Verdict: **{summary['verdict']}**.",
        "",
        "This benchmark begins from physically continuous mid-play states and contains no kickoff or reset anywhere in the measured duel. Every base state is replayed four ways to balance physical role, Blue/Orange assignment, and an exact 180-degree team mirror.",
        "",
        "## Headline outcome",
        "",
        f"- Overall: {_outcome_line(overall)}.",
        f"- Rival as **Blue**: {_outcome_line(blue)}.",
        f"- Rival as **Orange**: {_outcome_line(orange)}.",
        "",
        "Blue and Orange are reported separately because the prior full-match benchmark exposed a material team-side scoring asymmetry; the overall figure never replaces these side-specific results.",
        "",
        "| Dimension | Stratum | Duels | Rival wins | Nexto wins | Draws | Decisive Rival win rate |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for dimension in (
        "state_orientation", "source_policy", "initial_physical_role_inherited_by_rival",
        "field_third", "height_bin", "initial_closest_policy", "boost_bin",
    ):
        for label, value in outcomes[dimension].items():
            fraction = value["decisive_duel_rival_win_rate"]["fraction"]
            lines.append(
                f"| {dimension} | {label} | {value['duels']:,} | {value['rival_wins']:,} | "
                f"{value['nexto_wins']:,} | {value['draws']:,} | {100.0 * fraction:.3f}% |"
            )
    lines += [
        "",
        "## Four-way paired-state control",
        "",
        f"- Complete families without a draw: `{paired['complete_without_draws']:,}/{paired['families']:,}`.",
        f"- Families where draws prevent a complete four-way decision: `{paired['draws_prevent_complete_four_way_decision']:,}`.",
    ]
    for label, count in paired["complete_family_rival_win_count"].items():
        lines.append(f"- {label.replace('_', ' ')}: `{count:,}`.")
    lines += [
        "",
        "## Open-play behavior",
        "",
        "| Policy | Touches | Touch share | First touch share | Same next touch | Opponent handoff | Wall continuations | Backboard continuations | Demos |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, value in (("Rival", rival_behavior), ("Nexto", nexto_behavior)):
        lines.append(
            f"| {label} | {value['touches']:,} | {value['touch_share']['fraction']:.6f} | "
            f"{value['first_accepted_touch_after_restore']['fraction']:.6f} | "
            f"{value['next_touch_possession']['same_player_retention']['fraction']:.6f} | "
            f"{value['next_touch_possession']['opponent_handoff']['fraction']:.6f} | "
            f"{value['wall_continuation']['fraction']:.6f} | {value['backboard_continuation']['fraction']:.6f} | {value['demos']:,} |"
        )
    lines += [
        "",
        "The machine-readable telemetry also contains forward/neutral/backward immediate direction and net displacement, goal-entry X/Z histograms, final-touch-to-goal distributions, scorer/last-toucher agreement, and the same policy-separated metrics for Rival-as-Blue and Rival-as-Orange. These are descriptive categories; backward, wall, and backboard play are not labeled inherently bad.",
        "",
        "## State bank and integrity",
        "",
        f"- Base states: `{summary['state_bank']['states']:,}`: `{SOURCE_WORLDS:,}` stochastic final-45B Rival self-play and `{SOURCE_WORLDS:,}` deterministic pinned-Nexto self-play.",
        f"- Capture rule: `{summary['state_bank']['capture_rule']}`.",
        f"- Full continuation fields: `{summary['state_bank']['continuation_fields']}`; compressed bank SHA-256 `{summary['state_bank']['artifact_sha256']}`.",
        "- Each capture is at least 600 active physics ticks old, follows an accepted touch, has two active cars, is inside the scoring plane, and has no pending reset.",
        "- The one-decision policy-memory boundary is neutral: Rival and Nexto previous actions are all zeros; physical boost/demo/jump/flip/pad/lifecycle timers are preserved.",
        f"- Mirror involution max error: `{summary['integrity']['mirror_involution']['maximum_numeric_error_quaternion_sign_equivalent']}`; failed fields `{len(summary['integrity']['mirror_involution']['failed_fields'])}`.",
        f"- Capture/restore exact-field audit: `{summary['integrity']['restore_fidelity']['fields_exactly_checked']}` fields, max float error `{summary['integrity']['restore_fidelity']['maximum_float_abs_error']}`, failed fields `{len(summary['integrity']['restore_fidelity']['failed_fields'])}`.",
        f"- Duel-loop profiled H2D/D2H events: `{summary['performance']['profiled_h2d_d2h_event_count']}`; runtime `{summary['performance']['long_world_ticks_per_second']:,.2f}` world-ticks/s; peak CUDA `{summary['performance']['peak_cuda_bytes'] / 2**30:.3f}` GiB.",
        f"- Actual kickoff/reset events after restored start: `{summary['integrity']['kickoff_event_total']}` / `{summary['integrity']['reset_event_total']}`.",
        "",
        "## Frozen identities",
        "",
        f"- Rival checkpoint SHA-256: `{summary['identity']['rival_checkpoint']['sha256']}`; policy version `{summary['identity']['rival_checkpoint']['policy_version']}`, cumulative samples `{summary['identity']['rival_checkpoint']['total_agent_samples']:,}`.",
        f"- Nexto upstream commit: `{summary['identity']['nexto_upstream_commit']}`; model SHA-256 `{summary['identity']['nexto_model_sha256']}`.",
        "- Pinned Nexto material remains under CC BY-NC-SA 4.0 and is unchanged by this benchmark.",
        "",
        "## Evidence",
        "",
        "- `results/rival2/nexto_open_play/summary.json`",
        "- `results/rival2/nexto_open_play/outcomes.json`",
        "- `results/rival2/nexto_open_play/behavioral_telemetry.json`",
        "- `results/rival2/nexto_open_play/state_bank_description.json`",
        "- `results/rival2/nexto_open_play/state_bank.npz`",
        "- `results/rival2/nexto_open_play/per_duel_ledger.csv`",
        "- `results/rival2/nexto_open_play/paired_family_ledger.csv`",
        "- `results/rival2/nexto_open_play/paired_summary.json`",
        "- `results/rival2/nexto_open_play/evidence_manifest.json`",
        "",
        "## Explicitly deferred",
        "",
        "Fake-kickoff curriculum work—including retreat/backflip-to-boost opponents that intentionally concede first contact—is recorded as future work only. No training, reward/PPO/model/physics/controller change, viewer work, v0.6 work, or fake-kickoff implementation occurred here.",
        "",
    ]
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collision-dir", type=Path, default=COLLISION_ROOT)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    started = time.perf_counter()
    if _git("rev-parse", "HEAD") != EXPECTED_HEAD:
        raise RuntimeError("open-play benchmark must start from the authorized HEAD")
    if not args.collision_dir.is_dir():
        raise FileNotFoundError(args.collision_dir)
    fidelity = json.loads(FIDELITY_PATH.read_text(encoding="utf-8"))
    identity_checks = {
        "rival_checkpoint_sha_exact": _sha256(CHECKPOINT) == CHECKPOINT_SHA256,
        "nexto_fidelity_green": fidelity.get("verdict") == "PASS_GREEN",
        "nexto_upstream_commit_exact": fidelity["provenance"]["upstream_commit"] == NEXTO_COMMIT,
        "nexto_model_sha_exact": fidelity["provenance"]["model_sha256"] == NEXTO_MODEL_SHA256 == MODEL_SHA256,
        "accepted_rival_policy_runtime_unchanged": subprocess.run(["git", "diff", "--quiet", EXPECTED_HEAD, "--", "rivalsim/rival2_policy.py", "rivalsim/rival2_env.py"], cwd=ROOT).returncode == 0,
        "accepted_nexto_adapter_unchanged": subprocess.run(["git", "diff", "--quiet", EXPECTED_HEAD, "--", "third_party/nexto"], cwd=ROOT).returncode == 0,
        "accepted_physics_unchanged": subprocess.run(["git", "diff", "--quiet", EXPECTED_HEAD, "--", "rivalsim/static_world.py", "rivalsim/kernels", "rivalsim/vehicle.py"], cwd=ROOT).returncode == 0,
    }
    if not all(identity_checks.values()):
        raise RuntimeError(f"frozen identity/scope check failed: {identity_checks}")

    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    face_map, mesh_map, mirror_map_evidence = build_face_mirror_maps(geometry)
    rival_values, rival_ticks, rival_harvest, rival_identity = _harvest_rival(args.collision_dir, args.device)
    nexto_values, nexto_ticks, nexto_harvest = _harvest_nexto(args.collision_dir, args.device)
    values, capture_tick, source = _combine_banks(rival_values, nexto_values, rival_ticks, nexto_ticks)
    del rival_values, nexto_values
    gc.collect(); torch.cuda.empty_cache()

    if len(values) != 299:
        raise RuntimeError(f"continuation schema changed: expected 299 arrays, got {len(values)}")
    capture_integrity = {
        "all_episode_age_at_least_600": bool(np.all(values["rival2.episode_ticks"].reshape(-1) >= MIN_CAPTURE_TICKS)),
        "all_inside_scoring_plane": bool(np.all(np.abs(values["state.ball_pos"].reshape(-1, 3)[:, 1]) < GOAL_SCORING_PLANE_Y_UU)),
        "both_cars_active": bool(np.all(values["car_car.car_is_demoed"].reshape(-1, 2) == 0)),
        "no_goal_pending": bool(np.all(values["lifecycle.goal_scored"] == 0)),
        "no_reset_pending": bool(np.all(values["rival2.reset_mask"] == 0) and np.all(values["lifecycle.reset_required"] == 0)),
        "not_training_kickoff_indicator": bool(np.all(values["rival2.kickoff_indicator"] == 0)),
        "source_counts_exact": bool(np.array_equal(np.bincount(source, minlength=2), np.asarray((SOURCE_WORLDS, SOURCE_WORLDS)))),
    }
    if not all(capture_integrity.values()):
        raise RuntimeError(f"captured bank integrity failed: {capture_integrity}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    field_keys, bank_sha = _save_state_bank(values, capture_tick, source, OUTPUT_DIR / "state_bank.npz")
    state_bank_description = {
        "schema": "RIVAL2_OPEN_PLAY_STATE_BANK_V1",
        "states": BASE_STATES,
        "continuation_fields": len(values),
        "source_counts": {str(SOURCE_NAMES[index]): int((source == index).sum()) for index in range(2)},
        "capture_seeds": {str(SOURCE_NAMES[0]): RIVAL_SOURCE_SEED, str(SOURCE_NAMES[1]): NEXTO_SOURCE_SEED},
        "capture_rule": "For each seeded source world and episode, prospectively choose age 600 + ((world_index*1103515245 + reset_count*12345 + seed) & 0x7fffffff) % 601; capture the first subsequent 30 Hz boundary satisfying every eligibility condition. The rule never observes policy advantage or duel outcome.",
        "eligibility": {
            "minimum_active_ticks_since_reset": MIN_CAPTURE_TICKS,
            "accepted_touch_since_reset_required": True,
            "goal_or_reset_pending_forbidden": True,
            "ball_beyond_scoring_plane_forbidden": True,
            "both_cars_active_required": True,
            "kickoff_control_forbidden": True,
        },
        "neutral_policy_memory_boundary": {
            "rival_previous_action": "all zero at restored duel boundary",
            "nexto_previous_action": "all zero at restored duel boundary",
            "physical_and_lifecycle_timers": "preserved from source continuation",
        },
        "field_key_map": field_keys,
        "artifact": "state_bank.npz",
        "artifact_sha256": bank_sha,
        "artifact_size_bytes": (OUTPUT_DIR / "state_bank.npz").stat().st_size,
        "capture_integrity": capture_integrity,
        "harvest": [rival_harvest, nexto_harvest],
        "distribution": _state_bank_statistics(values, source),
    }
    _write_json(OUTPUT_DIR / "state_bank_description.json", state_bank_description)

    mirror_involution = mirror_involution_report(values, face_map, mesh_map)
    if not mirror_involution["pass"]:
        raise RuntimeError(f"mirror involution failed: {mirror_involution}")
    metadata = _four_way_metadata(values, source)
    assignment_counts = {
        "duels": int(metadata["duel_id"].size),
        "base_states": int(np.unique(metadata["base_state_id"]).size),
        "original_rival_blue": int(((metadata["variant"] == 0)).sum()),
        "original_rival_orange": int(((metadata["variant"] == 1)).sum()),
        "mirror_rival_blue": int(((metadata["variant"] == 2)).sum()),
        "mirror_rival_orange": int(((metadata["variant"] == 3)).sum()),
        "rival_as_blue": int((metadata["rival_side"] == 0).sum()),
        "rival_as_orange": int((metadata["rival_side"] == 1).sum()),
    }
    if set(assignment_counts.values()) - {BASE_STATES, DUELS, DUELS // 2}:
        raise RuntimeError(f"four-way assignment count failed: {assignment_counts}")

    print(f"duels: initializing all {DUELS:,} restored worlds", flush=True)
    runner = OpenPlayDuelRunner(
        str(args.collision_dir), CHECKPOINT, values, capture_tick, face_map, mesh_map,
        evaluation_seed=DUEL_SEED, device=args.device,
    )
    restore_fidelity = _restore_fidelity_report(runner, values, capture_tick, face_map, mesh_map)
    if not restore_fidelity["pass"]:
        raise RuntimeError(f"capture/restore fidelity failed: {restore_fidelity}")
    initial_no_kickoff = bool(torch.count_nonzero(runner.no_kickoff).item() == 0)
    del values
    gc.collect()

    profile_timing, transfer_events = runner.profile_ticks(PROFILE_TICKS)
    print(f"duels: running remaining {DUEL_LIMIT_TICKS - PROFILE_TICKS:,} physics ticks", flush=True)
    long_timing = runner.run_ticks(DUEL_LIMIT_TICKS - PROFILE_TICKS)
    exported = runner.export()
    raw = exported.pop("raw")
    outcome = _outcome_codes(raw, metadata)
    outcomes = _stratify_outcomes(outcome, raw, metadata)
    paired, family_ledger = _paired_summary(outcome)
    behavior = _behavioral_summary(raw, metadata)
    _write_json(OUTPUT_DIR / "outcomes.json", outcomes)
    _write_json(OUTPUT_DIR / "paired_summary.json", paired)
    _write_json(OUTPUT_DIR / "behavioral_telemetry.json", behavior)
    _write_duel_ledger(OUTPUT_DIR / "per_duel_ledger.csv", outcome, raw, metadata)
    _write_family_ledger(OUTPUT_DIR / "paired_family_ledger.csv", family_ledger, source)

    performance = {
        "profile_ticks": PROFILE_TICKS,
        "profile_seconds": profile_timing.seconds,
        "profile_world_ticks_per_second": profile_timing.world_ticks_per_second,
        "profiled_h2d_d2h_event_count": len(transfer_events),
        "profiled_h2d_d2h_event_names": transfer_events,
        "long_ticks": DUEL_LIMIT_TICKS - PROFILE_TICKS,
        "long_seconds": long_timing.seconds,
        "long_world_ticks_per_second": long_timing.world_ticks_per_second,
        "peak_cuda_bytes": exported["peak_cuda_bytes"],
        "world_h2d_bytes_after_initialization": exported["world_host_to_device_bytes_after_initialization"],
        "world_d2h_bytes_before_final_export": exported["world_device_to_host_bytes_before_export"],
        "nexto_timed_h2d_bytes": exported["nexto_timed_h2d_bytes"],
        "nexto_timed_d2h_bytes": exported["nexto_timed_d2h_bytes"],
        "nexto_inference_calls": exported["nexto_inference_calls"],
    }
    kickoff_total = int(raw["kickoff_event_count"].sum())
    reset_total = int(raw["reset_event_count"].sum())
    integrity = {
        "identity": identity_checks,
        "capture": capture_integrity,
        "mirror_map": mirror_map_evidence,
        "mirror_involution": mirror_involution,
        "restore_fidelity": restore_fidelity,
        "four_way_assignment_counts": assignment_counts,
        "four_way_assignment_exact": assignment_counts == {
            "duels": DUELS, "base_states": BASE_STATES,
            "original_rival_blue": BASE_STATES, "original_rival_orange": BASE_STATES,
            "mirror_rival_blue": BASE_STATES, "mirror_rival_orange": BASE_STATES,
            "rival_as_blue": DUELS // 2, "rival_as_orange": DUELS // 2,
        },
        "no_duel_begins_in_kickoff_control": initial_no_kickoff,
        "kickoff_event_total": kickoff_total,
        "reset_event_total": reset_total,
        "no_kickoff_or_reset_after_start": kickoff_total == 0 and reset_total == 0,
        "every_duel_classified": int((outcome == 1).sum() + (outcome == -1).sum() + (outcome == 0).sum()) == DUELS,
        "goal_duels_end_at_first_goal": bool(np.all(raw["done"][raw["done"] != 0] == 1)),
        "draws_reach_exact_limit": bool(np.all(raw["elapsed_ticks"][raw["done"] == 0] == DUEL_LIMIT_TICKS)),
        "all_goal_times_within_limit": bool(np.all((raw["goal_tick"][raw["done"] != 0] > 0) & (raw["goal_tick"][raw["done"] != 0] <= DUEL_LIMIT_TICKS))),
        "telemetry_capacity_model": "aggregate counters plus one terminal goal per duel",
        "telemetry_overflow_count": 0,
        "timed_hot_path_host_transfers_zero": len(transfer_events) == 0 and exported["nexto_timed_h2d_bytes"] == 0 and exported["nexto_timed_d2h_bytes"] == 0,
    }
    gate_keys = (
        "four_way_assignment_exact", "no_duel_begins_in_kickoff_control",
        "no_kickoff_or_reset_after_start", "every_duel_classified",
        "goal_duels_end_at_first_goal", "draws_reach_exact_limit",
        "all_goal_times_within_limit", "timed_hot_path_host_transfers_zero",
    )
    pass_green = all(integrity[key] for key in gate_keys) and all(identity_checks.values()) and all(capture_integrity.values()) and mirror_involution["pass"] and restore_fidelity["pass"]
    summary = {
        "verdict": "PASS_GREEN" if pass_green else "FAIL_RED",
        "scope": "authorized kickoff-free open-play evaluation only; no policy training or policy/reward/PPO/physics change",
        "identity": {
            "authorized_start_head": EXPECTED_HEAD,
            "rival_checkpoint": rival_identity,
            "nexto_upstream_commit": NEXTO_COMMIT,
            "nexto_model_sha256": NEXTO_MODEL_SHA256,
            "nexto_fidelity_evidence": FIDELITY_PATH.relative_to(ROOT).as_posix(),
        },
        "state_bank": {
            "states": BASE_STATES,
            "continuation_fields": len(field_keys),
            "artifact_sha256": bank_sha,
            "capture_rule": state_bank_description["capture_rule"],
        },
        "duel_contract": {
            "duels": DUELS, "physics_hz": PHYSICS_HZ,
            "maximum_ticks": DUEL_LIMIT_TICKS, "maximum_simulated_seconds": DUEL_LIMIT_TICKS / PHYSICS_HZ,
            "first_goal_wins": True, "kickoff_at_start": False, "goal_reset": False,
            "rival_action_mode": "deterministic deployment at 30 Hz",
            "nexto_action_mode": "deterministic beta=1 argmax at 15 Hz; kickoff controller disabled for all duel ticks",
        },
        "headline_outcomes": {
            "overall": outcomes["overall"],
            "rival_as_blue": outcomes["rival_side"]["Blue"],
            "rival_as_orange": outcomes["rival_side"]["Orange"],
        },
        "integrity": integrity,
        "performance": performance,
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__, "warp": wp.__version__,
            "cuda_device": torch.cuda.get_device_name(torch.device(args.device)),
            "collision_root": str(args.collision_dir),
        },
        "wall_seconds": time.perf_counter() - started,
        "future_work_not_executed": "fake-kickoff/backflip-to-boost opponent curriculum",
    }
    _write_json(OUTPUT_DIR / "summary.json", summary)
    _write_document(summary, outcomes, paired, behavior)

    evidence_files = [
        DOC_PATH,
        OUTPUT_DIR / "summary.json", OUTPUT_DIR / "outcomes.json",
        OUTPUT_DIR / "behavioral_telemetry.json", OUTPUT_DIR / "state_bank_description.json",
        OUTPUT_DIR / "state_bank.npz", OUTPUT_DIR / "per_duel_ledger.csv",
        OUTPUT_DIR / "paired_family_ledger.csv", OUTPUT_DIR / "paired_summary.json",
    ]
    evidence_manifest = {
        "schema": "RIVAL2_NEXTO_OPEN_PLAY_EVIDENCE_V1",
        "verdict": summary["verdict"],
        "files": {
            path.relative_to(ROOT).as_posix(): {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
            for path in evidence_files
        },
    }
    _write_json(OUTPUT_DIR / "evidence_manifest.json", evidence_manifest)
    print(json.dumps({
        "verdict": summary["verdict"],
        "overall": outcomes["overall"],
        "rival_as_blue": outcomes["rival_side"]["Blue"],
        "rival_as_orange": outcomes["rival_side"]["Orange"],
        "paired_complete": paired["complete_family_rival_win_count"],
        "wall_seconds": summary["wall_seconds"],
    }, indent=2), flush=True)
    return 0 if pass_green else 1


if __name__ == "__main__":
    raise SystemExit(main())
