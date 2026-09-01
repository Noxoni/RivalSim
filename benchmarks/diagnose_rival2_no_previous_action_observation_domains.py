"""Read-only human-demo versus native RivalSim observation-domain diagnostic.

This is deliberately a bounded diagnostic, not a trainer or a new acceptance
framework.  It loads the selected no-previous-action Stage-1 checkpoint without
modifying it, materializes the exact human BC input path, constructs matched native
RivalSim states from recorded physical telemetry, and probes the five standard native
kickoffs.  No optimizer, reward computation, or physics step is executed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_rival2_fresh_human_seed_v1 as stage1  # noqa: E402
from benchmarks.run_rival2_fresh_human_seed_no_previous_action_v1 import (  # noqa: E402
    CHECKPOINT,
    CHECKPOINT_FORMAT,
    RESULTS,
)
from rivalsim.human_demo.bc_observation_bridge import FIELD_QUALITY_SPECS  # noqa: E402
from rivalsim.human_demo.missing_feature_distillation import file_sha256  # noqa: E402
from rivalsim.human_demo.observation_adapter_v2 import native_pad_overlay  # noqa: E402
from rivalsim.human_demo.reader import SessionReader  # noqa: E402
from rivalsim.human_demo.training_adapter import _orientation_basis  # noqa: E402
from rivalsim.math import matrix_to_quat  # noqa: E402
from rivalsim.rival2_contracts import (  # noqa: E402
    ACTION_NAMES,
    EPISODE_AGE_SCALE_TICKS,
    NO_TOUCH_AGE_SCALE_TICKS,
    OBS_DIM,
    OBS_FIELD_NAMES,
    RIVAL2_ACTION_V2_120HZ_VERSION,
    RIVAL2_OBS_V2_120HZ_VERSION,
)
from rivalsim.rival2_env import (  # noqa: E402
    REWARD_MODE_GAMEPLAY_120_V1,
    Rival2TensorBridge,
    Rival2WorldSim,
)
from rivalsim.rival2_policy import (  # noqa: E402
    PREVIOUS_ACTION_OBSERVATION_INDICES,
    Rival2ActorCritic,
    Rival2PolicyConfig,
    deterministic_hybrid_action,
)

FORMAT = "RIVAL2_NO_PREVIOUS_ACTION_OBSERVATION_DOMAIN_DIAGNOSTIC_V1"
EXPECTED_CHECKPOINT_SHA256 = (
    "DE1B16086405744FCB2FF23FB7384FDD1AB0273384E33E1E1EB2314083271DDE"
)
SESSION_UUID = "CD6E7DB1-2761-4B8B-BD37-F21C7F135722"
FRAME_COUNT = 58_306
OUTPUT_DIR = RESULTS / "observation_domain_diagnostic"
OUTPUT_JSON = OUTPUT_DIR / "diagnostic.json"
OUTPUT_REPORT = OUTPUT_DIR / "REPORT.md"
DEFAULT_COLLISION_ROOT = Path(r"G:\dev\RLBot-Rival\bot\collision_meshes")

GROUPS = {
    "ball": (0, 9),
    "self_car": (9, 48),
    "opponent": (48, 87),
    "relative_state": (87, 99),
    "boost_pads": (99, 167),
    "previous_action_zeros": (167, 175),
    "lifecycle_timers": (175, 182),
}

LAYOUTS = (
    ("diagonal_left", (-2048.0, -2560.0), (2048.0, 2560.0)),
    ("diagonal_right", (2048.0, -2560.0), (-2048.0, 2560.0)),
    ("off_center_left", (-256.0, -3840.0), (256.0, 3840.0)),
    ("off_center_right", (256.0, -3840.0), (-256.0, 3840.0)),
    ("center", (0.0, -4608.0), (0.0, 4608.0)),
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _human_and_opponent(frame: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    humans = [
        car for car in frame["cars"] if car.get("flags", {}).get("is_local_human")
    ]
    if len(humans) != 1:
        raise RuntimeError(f"frame {frame['sequence']} lacks one unique human car")
    human = humans[0]
    opponents = [car for car in frame["cars"] if car is not human]
    if len(opponents) != 1:
        raise RuntimeError(f"frame {frame['sequence']} lacks one unique opponent")
    if int(human["team"]) != 0 or int(opponents[0]["team"]) != 1:
        raise RuntimeError("diagnostic expects the reviewed Blue-human 1v1 recording")
    return human, opponents[0]


def _layout_for_frame(frame: dict[str, Any], tolerance_uu: float = 8.0) -> int | None:
    human, opponent = _human_and_opponent(frame)
    human_xy = np.asarray(human["position"][:2], dtype=np.float64)
    opponent_xy = np.asarray(opponent["position"][:2], dtype=np.float64)
    scores = [
        max(
            float(np.linalg.norm(human_xy - np.asarray(blue))),
            float(np.linalg.norm(opponent_xy - np.asarray(orange))),
        )
        for _name, blue, orange in LAYOUTS
    ]
    best = int(np.argmin(scores))
    return best if scores[best] <= tolerance_uu else None


def _settled_kickoff(frame: dict[str, Any]) -> bool:
    layout = _layout_for_frame(frame)
    if layout is None:
        return False
    human, opponent = _human_and_opponent(frame)
    ball = frame["ball"]
    flags = frame["match"]["flags"]
    return bool(
        flags.get("kickoff_or_countdown")
        and np.linalg.norm(np.asarray(ball["position"][:2], dtype=np.float64)) <= 8.0
        and np.linalg.norm(np.asarray(ball["linear_velocity"], dtype=np.float64)) <= 20.0
        and all(
            15.0 <= float(car["position"][2]) <= 20.0
            and np.linalg.norm(np.asarray(car["linear_velocity"], dtype=np.float64)) <= 8.0
            and bool(car["flags"].get("on_ground"))
            for car in (human, opponent)
        )
    )


def _action_is_clear(action: np.ndarray) -> bool:
    return bool(
        abs(float(action[0])) >= 0.75
        or abs(float(action[1])) >= 0.35
        or bool(np.any(action[5:] >= 0.5))
    )


def select_representatives(
    frames: list[dict[str, Any]], actions: np.ndarray
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select one reset per layout and three transparent early-play phases."""

    settled_indices = [index for index, frame in enumerate(frames) if _settled_kickoff(frame)]
    runs: list[list[int]] = []
    for index in settled_indices:
        if not runs or index > runs[-1][-1] + 1:
            runs.append([index])
        else:
            runs[-1].append(index)

    episodes: list[dict[str, Any]] = []
    for run in runs:
        ready = run[0]
        layout = _layout_for_frame(frames[ready])
        if layout is None:
            continue
        search_stop = min(len(frames), run[-1] + 721)
        onset = next(
            (index for index in range(ready, search_stop) if _action_is_clear(actions[index])),
            None,
        )
        if onset is None:
            continue
        human, _opponent = _human_and_opponent(frames[ready])
        spawn_xy = np.asarray(human["position"][:2], dtype=np.float64)
        movement_stop = min(len(frames), ready + 1201)
        movement = next(
            (
                index
                for index in range(ready, movement_stop)
                if np.linalg.norm(
                    np.asarray(
                        _human_and_opponent(frames[index])[0]["linear_velocity"][:2],
                        dtype=np.float64,
                    )
                )
                >= 100.0
                or np.linalg.norm(
                    np.asarray(
                        _human_and_opponent(frames[index])[0]["position"][:2],
                        dtype=np.float64,
                    )
                    - spawn_xy
                )
                >= 16.0
            ),
            None,
        )
        if movement is None:
            continue
        episodes.append(
            {
                "layout": layout,
                "layout_name": LAYOUTS[layout][0],
                "ready_index": ready,
                "onset_index": onset,
                "movement_index": movement,
                "settled_run_first": run[0],
                "settled_run_last": run[-1],
            }
        )

    selected: list[dict[str, Any]] = []
    used_layouts: set[int] = set()
    for episode in episodes:
        layout = int(episode["layout"])
        if layout in used_layouts:
            continue
        used_layouts.add(layout)
        ready = int(episode["ready_index"])
        onset = int(episode["onset_index"])
        movement = int(episode["movement_index"])
        phases = (
            ("settled_reference", ready),
            ("clear_action_onset", onset),
            ("movement_onset", movement),
            ("early_play_30", movement + 30),
        )
        used_indices: set[int] = set()
        for phase, index in phases:
            if index >= len(frames) or index in used_indices:
                continue
            used_indices.add(index)
            if int(frames[index]["sequence"]) != index:
                raise RuntimeError("reviewed trajectory sequence/index identity changed")
            selected.append(
                {
                    "layout": layout,
                    "layout_name": LAYOUTS[layout][0],
                    "phase": phase,
                    "frame_index": index,
                    "sequence": int(frames[index]["sequence"]),
                    "physics_frame": int(frames[index]["physics_frame"]),
                }
            )
    if len(used_layouts) != 5:
        raise RuntimeError(
            f"reviewed gameplay did not expose all five kickoff layouts: {used_layouts}"
        )
    return selected, episodes


def _rotation_quaternion(car: dict[str, Any]) -> np.ndarray:
    forward, up = _orientation_basis(car["rotation"])
    right = np.cross(up, forward).astype(np.float32)
    matrix = np.column_stack((forward, right, up)).astype(np.float32)
    return matrix_to_quat(matrix)


def _component_time(car: dict[str, Any], component: str, *keys: str) -> float:
    row = car.get(component, {})
    return max((float(row.get(key, 0.0)) for key in keys), default=0.0)


def _native_car_values(car: dict[str, Any]) -> dict[str, Any]:
    flags = car["flags"]
    jumped = bool(flags.get("jumped"))
    double_jumped = bool(flags.get("double_jumped"))
    has_native_flip = bool(flags.get("has_flip"))
    is_flipping = bool(car.get("dodge_component", {}).get("active")) or bool(
        car.get("flip_component", {}).get("active")
    )
    boost_value = float(car.get("boost", 0.0))
    if boost_value <= 1.0001:
        boost_value *= 100.0
    wheels = {int(row.get("index", -1)): row for row in car.get("wheels", ())}
    return {
        "car_pos": np.asarray(car["position"], dtype=np.float32),
        "car_vel": np.asarray(car["linear_velocity"], dtype=np.float32),
        "car_quat": _rotation_quaternion(car),
        "car_ang_vel": np.asarray(car["angular_velocity"], dtype=np.float32),
        "boost": boost_value,
        "boosting_time": _component_time(car, "boost_component", "activity_time")
        if car.get("boost_component", {}).get("active")
        else 0.0,
        "is_boosting": int(bool(car.get("boost_component", {}).get("active"))),
        # These three are unavailable in the recorder schema and deliberately remain
        # neutral in the matched native state.  Their discrepancies are reported.
        "time_since_boosted": 0.0,
        "sticky_ticks": 0.0,
        "supersonic_time": 0.0,
        "on_ground": int(bool(flags.get("on_ground"))),
        "has_jumped": int(jumped),
        "is_jumping": int(bool(car.get("jump_component", {}).get("active"))),
        "has_double_jumped": int(double_jumped),
        "has_flipped": int(jumped and not double_jumped and not has_native_flip),
        "is_flipping": int(is_flipping),
        "jump_time": _component_time(car, "jump_component", "activity_time"),
        "air_time": float(car.get("time_off_ground", 0.0)),
        "air_time_since_jump": float(car.get("time_off_ground", 0.0)) if jumped else 0.0,
        "flip_time": max(
            _component_time(car, "dodge_component", "activity_time"),
            _component_time(car, "flip_component", "activity_time", "flip_time"),
        ),
        "is_supersonic": int(bool(flags.get("supersonic"))),
        "car_is_demoed": int(bool(flags.get("demolished"))),
        "demo_respawn_timer": float(car.get("respawn_time_remaining", 0.0)),
        "wheel_contact": np.asarray(
            [int(bool(wheels.get(index, {}).get("has_world_contact"))) for index in range(4)],
            dtype=np.int32,
        ),
    }


def _copy_view(bridge: Rival2TensorBridge, name: str, value: np.ndarray | list[Any]) -> None:
    destination = bridge.views[name]
    source = torch.as_tensor(value, dtype=destination.dtype, device=destination.device)
    if source.shape != destination.shape:
        source = source.reshape(destination.shape)
    destination.copy_(source)


def construct_matched_native_observations(
    frames: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    human_observations: np.ndarray,
    collision_root: Path,
    device: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Use native RivalSim storage + observation builder on recorded physical states."""

    count = len(selected)
    sim = Rival2WorldSim(
        count,
        str(collision_root),
        device=device,
        seed=2026090111,
        reward_mode=REWARD_MODE_GAMEPLAY_120_V1,
        kickoff_selector=np.asarray([row["layout"] for row in selected], dtype=np.int32),
    )
    bridge = Rival2TensorBridge(sim)
    car_rows: dict[str, list[Any]] = defaultdict(list)
    ball_pos: list[Any] = []
    ball_vel: list[Any] = []
    ball_ang_vel: list[Any] = []
    pad_cooldown = np.zeros((count, 34), dtype=np.float32)
    episode_ticks: list[int] = []
    no_touch_ticks: list[int] = []
    kickoff: list[int] = []
    touch_count: list[list[int]] = []
    demoed_event: list[list[int]] = []
    pad_support_counts: list[int] = []

    for row_index, selected_row in enumerate(selected):
        frame_index = int(selected_row["frame_index"])
        frame = frames[frame_index]
        human, opponent = _human_and_opponent(frame)
        for car in (human, opponent):
            values = _native_car_values(car)
            for name, value in values.items():
                car_rows[name].append(value)
        ball_pos.append(frame["ball"]["position"])
        ball_vel.append(frame["ball"]["linear_velocity"])
        ball_ang_vel.append(frame["ball"]["angular_velocity"])
        overlay = native_pad_overlay(frame)
        supported_pad_count = 0
        for pad in range(34):
            active_index = 99 + 2 * pad
            cooldown_index = active_index + 1
            if bool(overlay.supported[cooldown_index]):
                duration = 10.0 if pad < 6 else 4.0
                pad_cooldown[row_index, pad] = float(overlay.values[cooldown_index]) * duration
                supported_pad_count += 1
        pad_support_counts.append(supported_pad_count)
        human_obs = human_observations[frame_index]
        episode_ticks.append(round(float(human_obs[180]) * EPISODE_AGE_SCALE_TICKS))
        no_touch_ticks.append(round(float(human_obs[181]) * NO_TOUCH_AGE_SCALE_TICKS))
        kickoff.append(int(bool(frame["match"]["flags"].get("kickoff_or_countdown"))))
        touch_count.append([int(human_obs[176] >= 0.5), int(human_obs[177] >= 0.5)])
        demoed_event.append([int(human_obs[178] >= 0.5), int(human_obs[179] >= 0.5)])

    for name in (
        "car_pos",
        "car_vel",
        "car_quat",
        "car_ang_vel",
        "boost",
        "boosting_time",
        "is_boosting",
        "time_since_boosted",
        "on_ground",
        "has_jumped",
        "is_jumping",
        "has_double_jumped",
        "has_flipped",
        "is_flipping",
        "sticky_ticks",
        "jump_time",
        "air_time",
        "air_time_since_jump",
        "flip_time",
        "is_supersonic",
        "supersonic_time",
    ):
        _copy_view(bridge, name, np.asarray(car_rows[name]))
    _copy_view(bridge, "wheel_contact", np.asarray(car_rows["wheel_contact"]))
    _copy_view(bridge, "car_is_demoed", np.asarray(car_rows["car_is_demoed"]))
    _copy_view(bridge, "demo_respawn_timer", np.asarray(car_rows["demo_respawn_timer"]))
    _copy_view(bridge, "ball_pos", np.asarray(ball_pos, dtype=np.float32))
    _copy_view(bridge, "ball_vel", np.asarray(ball_vel, dtype=np.float32))
    _copy_view(bridge, "ball_ang_vel", np.asarray(ball_ang_vel, dtype=np.float32))
    _copy_view(bridge, "pad_cooldown", pad_cooldown)
    _copy_view(bridge, "rival2.episode_ticks", np.asarray(episode_ticks, dtype=np.int32))
    _copy_view(bridge, "rival2.no_touch_ticks", np.asarray(no_touch_ticks, dtype=np.int32))
    _copy_view(bridge, "rival2.kickoff_indicator", np.asarray(kickoff, dtype=np.int32))
    _copy_view(bridge, "rival2.touch_count", np.asarray(touch_count, dtype=np.int32))
    _copy_view(bridge, "rival2.demoed_event", np.asarray(demoed_event, dtype=np.int32))
    bridge.views["rival2.previous_action"].zero_()

    native = bridge.observation()[:, 0].detach().cpu().numpy().astype(np.float32)
    native[:, list(PREVIOUS_ACTION_OBSERVATION_INDICES)] = 0.0
    return native, {
        "method": (
            "recorded physical telemetry transplanted into native Rival2WorldSim "
            "storage, then Rival2TensorBridge.observation"
        ),
        "physics_advanced": False,
        "reward_computed": False,
        "policy_previous_action_forced_zero": True,
        "recording_unavailable_native_state_neutralized": [
            "self.time_since_boosted",
            "self.supersonic_time",
            "self.sticky_ticks",
            "opponent.time_since_boosted",
            "opponent.supersonic_time",
            "opponent.sticky_ticks",
        ],
        "unseen_pad_assumption": "active/cooldown=0; event-observed pads use native overlay timers",
        "event_and_age_semantics": (
            "exact event flags and BC trajectory lower-bound lifecycle counters copied "
            "into native storage"
        ),
        "pad_support_count_by_sample": pad_support_counts,
    }


def load_policy(device: str) -> tuple[Rival2ActorCritic, dict[str, Any]]:
    actual_sha = file_sha256(CHECKPOINT)
    if actual_sha != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(f"selected checkpoint SHA mismatch: {actual_sha}")
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise RuntimeError("selected checkpoint format mismatch")
    config = Rival2PolicyConfig(**payload["policy_config"])
    if not config.zero_previous_action_inputs:
        raise RuntimeError("selected checkpoint does not enforce the permanent input mask")
    model = Rival2ActorCritic(config).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval().requires_grad_(False)
    return model, {
        "path": CHECKPOINT.relative_to(ROOT).as_posix(),
        "sha256": actual_sha,
        "format": payload["format"],
        "selected_step": int(payload["selected_step"]),
        "zero_previous_action_inputs": True,
    }


@torch.no_grad()
def policy_actions(model: Rival2ActorCritic, observation: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(np.ascontiguousarray(observation)).to(next(model.parameters()).device)
    actor, _critic = model(tensor)
    return deterministic_hybrid_action(actor, model.config).cpu().numpy().astype(np.float32)


def native_kickoff_probe(
    model: Rival2ActorCritic, collision_root: Path, device: str
) -> list[dict[str, Any]]:
    sim = Rival2WorldSim(
        5,
        str(collision_root),
        device=device,
        seed=2026090112,
        reward_mode=REWARD_MODE_GAMEPLAY_120_V1,
        kickoff_selector=np.arange(5, dtype=np.int32),
    )
    bridge = Rival2TensorBridge(sim)
    bridge.views["rival2.previous_action"].zero_()
    observations = bridge.observation()[:, 0].detach().cpu().numpy().astype(np.float32)
    observations[:, list(PREVIOUS_ACTION_OBSERVATION_INDICES)] = 0.0
    actions = policy_actions(model, observations)
    positions = bridge.views["car_pos"].reshape(5, 2, 3).detach().cpu().numpy()
    rows: list[dict[str, Any]] = []
    for layout in range(5):
        rows.append(
            {
                "layout": layout,
                "layout_name": LAYOUTS[layout][0],
                "blue_position": positions[layout, 0].astype(float).tolist(),
                "orange_position": positions[layout, 1].astype(float).tolist(),
                "controller_output": {
                    name: float(actions[layout, index]) for index, name in enumerate(ACTION_NAMES)
                },
            }
        )
    return rows


def _metric(values: np.ndarray, native: np.ndarray) -> dict[str, float]:
    absolute = np.abs(values)
    rmse = float(np.sqrt(np.mean(np.square(values), dtype=np.float64)))
    native_rms = float(np.sqrt(np.mean(np.square(native), dtype=np.float64)))
    return {
        "mean_absolute": float(np.mean(absolute, dtype=np.float64)),
        "rmse_observation_contract_units": rmse,
        "maximum_absolute": float(np.max(absolute)),
        "native_rms": native_rms,
        "rmse_divided_by_max_native_rms_or_0_05": float(rmse / max(native_rms, 0.05)),
    }


def compare_observations(human: np.ndarray, native: np.ndarray) -> dict[str, Any]:
    difference = human - native
    groups: dict[str, Any] = {}
    for name, (start, stop) in GROUPS.items():
        metrics = _metric(difference[:, start:stop], native[:, start:stop])
        local = np.abs(difference[:, start:stop])
        flat = int(np.argmax(local))
        _sample, field_offset = np.unravel_index(flat, local.shape)
        metrics["largest_field"] = OBS_FIELD_NAMES[start + int(field_offset)]
        groups[name] = metrics

    fields: list[dict[str, Any]] = []
    for index, name in enumerate(OBS_FIELD_NAMES):
        metrics = _metric(difference[:, index], native[:, index])
        spec = FIELD_QUALITY_SPECS[index]
        fields.append(
            {
                "index": index,
                "field": name,
                "human_bridge_quality": spec.classification,
                **metrics,
            }
        )
    fields.sort(key=lambda row: (-float(row["mean_absolute"]), int(row["index"])))
    return {
        "sample_count": int(human.shape[0]),
        "all_fields": _metric(difference, native),
        "groups": groups,
        "fields_ranked_by_mean_absolute": fields,
    }


def compare_actions(human: np.ndarray, native: np.ndarray) -> dict[str, Any]:
    difference = human - native
    rows = []
    for index, name in enumerate(ACTION_NAMES):
        rows.append(
            {
                "channel": name,
                "mean_absolute": float(np.mean(np.abs(difference[:, index]))),
                "maximum_absolute": float(np.max(np.abs(difference[:, index]))),
                "binary_disagreement_count": (
                    int(np.count_nonzero(human[:, index] != native[:, index]))
                    if index >= 5
                    else None
                ),
            }
        )
    return {
        "complete_action_rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "complete_action_mean_absolute": float(np.mean(np.abs(difference))),
        "maximum_absolute": float(np.max(np.abs(difference))),
        "channels": rows,
    }


def temporal_ambiguity_probe(
    observations: np.ndarray,
    actions: np.ndarray,
    frames: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    """Measure action spread among near-identical stationary kickoff observations."""

    rows: list[int] = []
    for episode in episodes:
        start = int(episode["settled_run_first"])
        stop = int(episode["onset_index"]) + 1
        rows.extend(range(start, min(stop, len(frames)), 4))
    rows = sorted(set(rows))
    if len(rows) < 2:
        return {"candidate_count": len(rows), "near_pair_count": 0}
    obs = observations[rows]
    target = actions[rows]
    pair_rows: list[dict[str, Any]] = []
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            if _layout_for_frame(frames[rows[left]]) != _layout_for_frame(frames[rows[right]]):
                continue
            obs_rmse = float(np.sqrt(np.mean(np.square(obs[left] - obs[right]))))
            target_max = float(np.max(np.abs(target[left] - target[right])))
            if obs_rmse <= 0.02 and target_max >= 0.5:
                pair_rows.append(
                    {
                        "left_sequence": int(frames[rows[left]]["sequence"]),
                        "right_sequence": int(frames[rows[right]]["sequence"]),
                        "layout": int(_layout_for_frame(frames[rows[left]]) or 0),
                        "observation_rmse": obs_rmse,
                        "target_max_absolute_difference": target_max,
                    }
                )
    pair_rows.sort(
        key=lambda row: (
            row["observation_rmse"],
            -row["target_max_absolute_difference"],
        )
    )
    return {
        "candidate_count": len(rows),
        "near_identical_threshold_observation_rmse": 0.02,
        "material_target_difference_threshold": 0.5,
        "near_pair_count": len(pair_rows),
        "closest_materially_ambiguous_pairs": pair_rows[:20],
    }


def report_markdown(result: dict[str, Any]) -> str:
    comparison = result["observation_comparison"]
    action = result["action_comparison"]
    lines = [
        "# No-previous-action observation-domain diagnostic",
        "",
        f"Verdict: **{result['conclusion']['observation_domain_verdict']}**",
        "",
        (
            "This is a read-only diagnosis. No optimizer, reward calculation, physics "
            "step, or checkpoint mutation occurred."
        ),
        "",
        "## Group discrepancies",
        "",
        "| Group | Mean absolute | RMSE (normalized contract units) | Max | Largest field |",
        "|---|---:|---:|---:|---|",
    ]
    for name, metrics in comparison["groups"].items():
        lines.append(
            f"| {name} | {metrics['mean_absolute']:.6f} | "
            f"{metrics['rmse_observation_contract_units']:.6f} | "
            f"{metrics['maximum_absolute']:.6f} | `{metrics['largest_field']}` |"
        )
    lines.extend(
        [
            "",
            "## Human-pipeline versus matched-native deterministic actions",
            "",
            f"Complete-action RMSE: `{action['complete_action_rmse']:.6f}`; "
            f"mean absolute difference: `{action['complete_action_mean_absolute']:.6f}`; "
            f"maximum absolute difference: `{action['maximum_absolute']:.6f}`.",
            "",
            "## Five native RivalSim kickoff outputs",
            "",
            "| Layout | throttle | steer | pitch | yaw | roll | jump | boost | handbrake |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["native_kickoff_outputs"]:
        output = row["controller_output"]
        lines.append(
            f"| {row['layout']} {row['layout_name']} | "
            + " | ".join(f"{output[name]:.6f}" for name in ACTION_NAMES)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            result["conclusion"]["summary"],
            "",
            (
                "The complete field table, per-state inputs/actions, provenance, and "
                "integrity checks are in `diagnostic.json`."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.collision_root.is_dir():
        raise FileNotFoundError(args.collision_root)
    source_root = args.human_source_root
    session_dir = source_root / SESSION_UUID
    reader = SessionReader(session_dir)
    validation = reader.validate()
    if not validation.container_valid or validation.frame_count != FRAME_COUNT:
        raise RuntimeError("reviewed human gameplay source failed validation")

    checkpoint_before = file_sha256(CHECKPOINT)
    model, checkpoint_identity = load_policy(args.device)
    human_tensor, target_tensor, source_identity = stage1.load_gameplay(
        source_root,
        device=args.device,
        neutralize_previous_action=True,
    )
    source_identity = dict(source_identity)
    source_identity.pop("sequences", None)
    source_identity.pop("physics_frames", None)
    source_identity["complete_sequence_and_physics_lists_validated_but_not_embedded"] = True
    human_observations = human_tensor.numpy().astype(np.float32)
    targets = target_tensor.numpy().astype(np.float32)
    frames = list(SessionReader(session_dir).iter_frames())
    if len(frames) != FRAME_COUNT:
        raise RuntimeError("human frame count changed while running diagnostic")
    if np.count_nonzero(human_observations[:, list(PREVIOUS_ACTION_OBSERVATION_INDICES)]):
        raise RuntimeError("human Stage-1 observations contain nonzero previous-action fields")

    selected, episodes = select_representatives(frames, targets)
    selected_indices = [int(row["frame_index"]) for row in selected]
    human_selected = human_observations[selected_indices]
    target_selected = targets[selected_indices]
    native_selected, construction = construct_matched_native_observations(
        frames,
        selected,
        human_observations,
        args.collision_root,
        args.device,
    )
    if np.count_nonzero(native_selected[:, list(PREVIOUS_ACTION_OBSERVATION_INDICES)]):
        raise RuntimeError("matched native observations contain nonzero previous-action fields")

    human_actions = policy_actions(model, human_selected)
    native_actions = policy_actions(model, native_selected)
    observation_comparison = compare_observations(human_selected, native_selected)
    action_comparison = compare_actions(human_actions, native_actions)
    temporal = temporal_ambiguity_probe(human_observations, targets, frames, episodes)
    kickoff_outputs = native_kickoff_probe(model, args.collision_root, args.device)

    group_rmse = {
        name: float(metrics["rmse_observation_contract_units"])
        for name, metrics in observation_comparison["groups"].items()
    }
    material_groups = sorted(
        (name for name, value in group_rmse.items() if value >= 0.05),
        key=lambda name: -group_rmse[name],
    )
    domain_mismatch = bool(material_groups and action_comparison["complete_action_rmse"] >= 0.025)
    temporal_evidence = int(temporal.get("near_pair_count", 0)) > 0
    if domain_mismatch and temporal_evidence:
        category = "both observation-domain mismatch and temporal/action ambiguity"
    elif domain_mismatch:
        category = "observation-domain problem; temporal memory is not established by this probe"
    elif temporal_evidence:
        category = "temporal/action ambiguity; no material observation-domain mismatch"
    else:
        category = "neither conclusively established by the bounded probe"
    sensible_kickoff = all(
        float(row["controller_output"]["throttle"]) >= 0.75
        and (
            float(row["controller_output"]["boost"]) >= 0.5
            or float(row["controller_output"]["jump"]) >= 0.5
        )
        for row in kickoff_outputs
    )

    selected_rows: list[dict[str, Any]] = []
    for row_index, selected_row in enumerate(selected):
        selected_rows.append(
            {
                **selected_row,
                "demonstrated_action": {
                    name: float(target_selected[row_index, index])
                    for index, name in enumerate(ACTION_NAMES)
                },
                "human_pipeline_policy_action": {
                    name: float(human_actions[row_index, index])
                    for index, name in enumerate(ACTION_NAMES)
                },
                "matched_native_policy_action": {
                    name: float(native_actions[row_index, index])
                    for index, name in enumerate(ACTION_NAMES)
                },
                "observation_rmse": float(
                    np.sqrt(
                        np.mean(
                            np.square(human_selected[row_index] - native_selected[row_index])
                        )
                    )
                ),
                "action_rmse": float(
                    np.sqrt(
                        np.mean(
                            np.square(human_actions[row_index] - native_actions[row_index])
                        )
                    )
                ),
            }
        )

    checkpoint_after = file_sha256(CHECKPOINT)
    if checkpoint_after != checkpoint_before:
        raise RuntimeError("selected checkpoint changed during diagnostic")
    result = {
        "format": FORMAT,
        "scope": {
            "diagnosis_only": True,
            "training_process_found": False,
            "optimizer_steps": 0,
            "physics_steps": 0,
            "reward_evaluations": 0,
            "checkpoint_modified": False,
            "dataset_modified": False,
        },
        "contracts": {
            "observation": RIVAL2_OBS_V2_120HZ_VERSION,
            "action": RIVAL2_ACTION_V2_120HZ_VERSION,
            "observation_dimensions": OBS_DIM,
            "previous_action_indices": list(PREVIOUS_ACTION_OBSERVATION_INDICES),
            "previous_action_human_all_zero": True,
            "previous_action_native_all_zero": True,
        },
        "checkpoint": checkpoint_identity,
        "checkpoint_sha256_before_and_after_identical": checkpoint_before == checkpoint_after,
        "human_source": source_identity,
        "session_validation": {
            "container_valid": validation.container_valid,
            "frame_count": validation.frame_count,
            "sequence_gap_count": validation.sequence_gap_count,
            "missing_physics_frame_count": validation.missing_physics_frame_count,
        },
        "representative_selection": {
            "method": (
                "first settled reset per standard layout; settled reference, clear action "
                "onset, physical movement onset, and movement+30 phases"
            ),
            "selected_count": len(selected),
            "kickoff_episode_count": len(episodes),
            "rows": selected_rows,
        },
        "matched_native_construction": construction,
        "observation_comparison": observation_comparison,
        "action_comparison": action_comparison,
        "temporal_action_ambiguity_probe": temporal,
        "native_kickoff_outputs": kickoff_outputs,
        "conclusion": {
            "observation_domain_verdict": (
                "MISMATCH" if domain_mismatch else "MATCH_WITHIN_BOUNDED_THRESHOLDS"
            ),
            "material_group_threshold_rmse": 0.05,
            "material_action_threshold_rmse": 0.025,
            "material_groups": material_groups,
            "native_kickoff_outputs_sensible_by_throttle_plus_boost_or_jump_criterion": (
                sensible_kickoff
            ),
            "category": category,
            "summary": (
                f"The bounded evidence indicates {category}. "
                f"Material observation groups: {material_groups or ['none']}. "
                f"The five initial native kickoff outputs {'do' if sensible_kickoff else 'do not'} "
                "issue a strong throttle-plus-boost/jump kickoff command."
            ),
        },
    }
    write_json(OUTPUT_JSON, result)
    OUTPUT_REPORT.write_text(report_markdown(result), encoding="utf-8", newline="\n")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--collision-root", type=Path, default=DEFAULT_COLLISION_ROOT)
    parser.add_argument(
        "--human-source-root",
        type=Path,
        default=(
            Path(os.environ["APPDATA"])
            / "bakkesmod/bakkesmod/data/rival2/human_demos"
        ),
    )
    return parser.parse_args()


def main() -> int:
    result = run(parse_args())
    print(
        json.dumps(
            {
                "verdict": result["conclusion"]["observation_domain_verdict"],
                "category": result["conclusion"]["category"],
                "material_groups": result["conclusion"]["material_groups"],
                "action_rmse": result["action_comparison"]["complete_action_rmse"],
                "native_kickoff_outputs": result["native_kickoff_outputs"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
