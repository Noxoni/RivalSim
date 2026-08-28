#!/usr/bin/env python3
"""Physical Gameplay V3 classifier calibration and correction evidence.

This harness is intentionally validation-only.  It state-injects small,
deterministic scenario batches into the authoritative RivalSim 120 Hz physics
path, copies bounded traces after each simulator tick, and derives thresholds
only from the derivation split.  Held-out generation is a separate command
which requires an already-written frozen derivation artifact.
"""

# ruff: noqa: E402 -- direct execution prepends the repository root.

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import warp as wp

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmarks.run_rival2_mechanics_calibration import (
    MUSTY_OFFSETS,
    _base_state,
    _empty_musty_features,
    _musty_contact_features,
    _quat_from_euler,
    _quat_rotate,
    _run,
    _touch_onsets,
    _unit,
)
from rivalsim import CompleteWorldSim, StateSnapshot
from rivalsim.arena import ArenaGeometry, WarpArenaMeshes
from rivalsim.controls import ControlBatch
from rivalsim.rival2_contracts import (
    ACTION_CONTRACT_HASH,
    EPISODE_CONTRACT_HASH,
    OBSERVATION_SCHEMA_HASH,
    REWARD_GAMEPLAY_V3_CONTRACT,
    REWARD_GAMEPLAY_V3_CONTRACT_HASH,
    RIVAL2_REWARD_GAMEPLAY_V3_VERSION,
)
from rivalsim.rival2_env import Rival2Env

STARTING_HEAD = "7e6356fc2ebeffdec4d76bf458df840de71ead34"
PRIOR_ACCEPTED_EVIDENCE = "00a4865400291a5ff0a34925a966c0963f55d963"
SEED = 2026082801
DERIVATION_PER_CLASS = 16
HELDOUT_PER_CLASS = 8
CLASS_NAMES = ("positive", "near_miss", "ordinary_control")
CLASSIFIERS = ("contest", "power_contact", "controlled_flick")
OUTPUT_DIR = Path("results/rival2/gameplay_v3_validation_correction_v1")
CORPUS_FORMAT = "RIVAL2_GAMEPLAY_V3_PHYSICAL_CLASSIFIER_TRACE_V1"
DERIVATION_FORMAT = "RIVAL2_GAMEPLAY_V3_CLASSIFIER_DERIVATION_V1"


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest().upper()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def _provenance() -> dict[str, Any]:
    source_paths = (
        "rivalsim/gameplay_v3.py",
        "rivalsim/kernels/car_ball.py",
        "rivalsim/kernels/vehicle.py",
        "rivalsim/kernels/integrate.py",
        "rivalsim/state.py",
    )
    return {
        "source_commit": _git("rev-parse", "HEAD"),
        "starting_handoff_commit": STARTING_HEAD,
        "prior_accepted_evidence_commit": PRIOR_ACCEPTED_EVIDENCE,
        "simulator_source_sha256": {
            path: _sha256(REPOSITORY_ROOT / path) for path in source_paths
        },
        "contract_hashes": {
            "RIVAL2_OBS_V1": OBSERVATION_SCHEMA_HASH,
            "RIVAL2_ACTION_V1": ACTION_CONTRACT_HASH,
            "RIVAL2_REWARD_GAMEPLAY_V3": REWARD_GAMEPLAY_V3_CONTRACT_HASH,
            "RIVAL2_EPISODE_V1": EPISODE_CONTRACT_HASH,
        },
        "physics_hz": 120,
        "simulator": "CompleteWorldSim authoritative CUDA Soccar path",
    }


def _rows(classifier: str, split: str) -> list[dict[str, Any]]:
    if split not in ("derivation", "heldout"):
        raise ValueError(split)
    count = DERIVATION_PER_CLASS if split == "derivation" else HELDOUT_PER_CLASS
    offset = 0 if split == "derivation" else DERIVATION_PER_CLASS
    classifier_index = CLASSIFIERS.index(classifier)
    rows: list[dict[str, Any]] = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        for local_index in range(count):
            global_index = offset + local_index
            seed = SEED + classifier_index * 10000 + class_index * 1000 + global_index
            rows.append(
                {
                    "format": CORPUS_FORMAT,
                    "classifier": classifier,
                    "class": class_name,
                    "split": split,
                    "class_index": global_index,
                    "scenario_id": (
                        f"{classifier}-{class_name}-{split[0].upper()}{global_index:02d}"
                    ),
                    "seed": seed,
                    "initial_state_generator": (
                        f"run_rival2_gameplay_v3_validation_correction.py::"
                        f"_{classifier}_cases/v1"
                    ),
                    "action_sequence_identity": "pending",
                    "scenario": {},
                    "measured_contact_ticks": {},
                    "features": {},
                }
            )
    return rows


class DualTrace:
    """Bounded calibration-only host trace including both cars."""

    def __init__(self) -> None:
        self.car_pos: list[np.ndarray] = []
        self.car_vel: list[np.ndarray] = []
        self.car_quat: list[np.ndarray] = []
        self.car_ang: list[np.ndarray] = []
        self.on_ground: list[np.ndarray] = []
        self.has_jumped: list[np.ndarray] = []
        self.has_double_jumped: list[np.ndarray] = []
        self.has_flipped: list[np.ndarray] = []
        self.is_flipping: list[np.ndarray] = []
        self.flip_torque: list[np.ndarray] = []
        self.ball_pos: list[np.ndarray] = []
        self.ball_vel: list[np.ndarray] = []
        self.hit: list[np.ndarray] = []
        self.contact_normal: list[np.ndarray] = []
        self.contact_point: list[np.ndarray] = []
        self.pre_car_vel: list[np.ndarray] = []
        self.pre_car_ang: list[np.ndarray] = []
        self.pre_ball_vel: list[np.ndarray] = []
        self.ball_delta_v: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []

    def append(self, sim: CompleteWorldSim, controls: ControlBatch) -> None:
        snap = sim.snapshot()
        hit = np.stack(
            (
                np.asarray(sim.car_ball.hit_this_tick.numpy()),
                np.asarray(sim.car_ball_b.hit_this_tick.numpy()),
            ),
            axis=1,
        )
        self.car_pos.append(snap.car_pos.copy())
        self.car_vel.append(snap.car_vel.copy())
        self.car_quat.append(snap.car_quat.copy())
        self.car_ang.append(snap.car_ang_vel.copy())
        self.on_ground.append(snap.on_ground.copy())
        self.has_jumped.append(snap.has_jumped.copy())
        self.has_double_jumped.append(snap.has_double_jumped.copy())
        self.has_flipped.append(snap.has_flipped.copy())
        self.is_flipping.append(snap.is_flipping.copy())
        self.flip_torque.append(snap.flip_rel_torque.copy())
        self.ball_pos.append(snap.ball_pos.copy())
        self.ball_vel.append(snap.ball_vel.copy())
        self.hit.append(hit)
        self.contact_normal.append(
            np.stack(
                (
                    np.asarray(sim.car_ball.contact_normal.numpy()),
                    np.asarray(sim.car_ball_b.contact_normal.numpy()),
                ),
                axis=1,
            )
        )
        self.contact_point.append(
            np.stack(
                (
                    np.asarray(sim.car_ball.contact_point_a_bt.numpy()),
                    np.asarray(sim.car_ball_b.contact_point_a_bt.numpy()),
                ),
                axis=1,
            )
        )
        self.pre_car_vel.append(
            np.stack(
                (
                    np.asarray(sim.car_ball.pre_car_velocity_bt.numpy()),
                    np.asarray(sim.car_ball_b.pre_car_velocity_bt.numpy()),
                ),
                axis=1,
            )
        )
        self.pre_car_ang.append(
            np.stack(
                (
                    np.asarray(sim.car_ball.pre_car_angular_velocity.numpy()),
                    np.asarray(sim.car_ball_b.pre_car_angular_velocity.numpy()),
                ),
                axis=1,
            )
        )
        self.pre_ball_vel.append(
            np.stack(
                (
                    np.asarray(sim.car_ball.pre_ball_velocity_bt.numpy()),
                    np.asarray(sim.car_ball_b.pre_ball_velocity_bt.numpy()),
                ),
                axis=1,
            )
        )
        self.ball_delta_v.append(
            np.stack(
                (
                    np.asarray(sim.car_ball.extra_hit_velocity_uu.numpy()),
                    np.asarray(sim.car_ball_b.extra_hit_velocity_uu.numpy()),
                ),
                axis=1,
            )
        )
        self.actions.append(
            np.stack(
                (
                    controls.throttle,
                    controls.steer,
                    controls.pitch,
                    controls.yaw,
                    controls.roll,
                    controls.jump,
                    controls.boost,
                    controls.handbrake,
                ),
                axis=2,
            )
        )

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            name: np.asarray(value)
            for name, value in vars(self).items()
            if isinstance(value, list)
        }


def _run_dual(
    state: StateSnapshot,
    ticks: int,
    collision_root: str,
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    controller: Callable[[int, ControlBatch], None],
) -> dict[str, np.ndarray]:
    sim = CompleteWorldSim(
        state.num_envs,
        collision_root,
        initial=state,
        geometry=geometry,
        meshes=meshes,
        auto_kickoff=False,
        car_visitation_order="a_then_b",
    )
    trace = DualTrace()
    controls = ControlBatch.zeros(state.num_envs)
    trace.append(sim, controls)
    for tick in range(ticks):
        controls = ControlBatch.zeros(state.num_envs)
        controller(tick, controls)
        sim.set_controls(controls)
        sim.step(1, synchronize=True)
        trace.append(sim, controls)
    return trace.arrays()


def _onsets(mask: np.ndarray) -> np.ndarray:
    active = mask != 0
    return np.flatnonzero(
        active & ~np.concatenate((np.zeros(1, dtype=bool), active[:-1]))
    )


def _contest_cases(
    rows: list[dict[str, Any]],
    collision_root: str,
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
) -> list[dict[str, Any]]:
    count = len(rows)
    state = _base_state(count)
    state.ball_pos[:] = (0.0, 0.0, 105.0)
    state.ball_vel[:] = (0.0, 0.0, 0.0)
    state.on_ground[:, :] = 0
    state.has_jumped[:, :] = 1
    state.air_time[:, :] = 0.12
    state.air_time_since_jump[:, :] = 0.12
    dodge_pitch = np.full(count, -1.0, dtype=np.float32)
    for index, row in enumerate(rows):
        local = int(row["class_index"])
        rng = np.random.default_rng(int(row["seed"]))
        jitter = float(rng.uniform(-3.0, 3.0))
        state.car_pos[index, 0] = (-310.0 - jitter, 0.0, 48.0)
        state.car_vel[index, 0] = (720.0 + 7.0 * (local % 5), 0.0, 0.0)
        state.car_quat[index, 0] = _quat_from_euler(0.0, 0.0, 0.0)
        kind = ""
        if row["class"] == "positive":
            kind = (
                "simultaneous_two_car_50",
                "opponent_contact_just_before_self",
                "opponent_contact_just_after_self",
                "genuine_converging_challenge",
            )[local % 4]
            if kind == "simultaneous_two_car_50":
                opponent_x, opponent_speed, opponent_y = 310.0 + jitter, -720.0, 0.0
            elif kind == "opponent_contact_just_before_self":
                opponent_x, opponent_speed, opponent_y = 292.0 + jitter, -720.0, 0.0
            elif kind == "opponent_contact_just_after_self":
                opponent_x, opponent_speed, opponent_y = 330.0 + jitter, -720.0, 0.0
            else:
                opponent_x, opponent_speed, opponent_y = 430.0 + jitter, -980.0, 35.0
        elif row["class"] == "near_miss":
            kind = (
                "distant_opponent",
                "nearby_opponent_moving_away",
                "nearby_nonconverging_opponent",
                "opponent_behind_play",
                "delayed_unrelated_opponent_contact",
                "uncontested_loose_ball_flip_through",
                "self_not_converging",
                "mismatched_arrival_times",
            )[local % 8]
            if kind == "distant_opponent":
                opponent_x, opponent_speed, opponent_y = 900.0 + jitter, -250.0, 0.0
            elif kind == "nearby_opponent_moving_away":
                opponent_x, opponent_speed, opponent_y = 350.0 + jitter, 450.0, 0.0
            elif kind == "nearby_nonconverging_opponent":
                opponent_x, opponent_speed, opponent_y = 330.0 + jitter, 0.0, 310.0
            elif kind == "opponent_behind_play":
                opponent_x, opponent_speed, opponent_y = -620.0 + jitter, 350.0, 0.0
            elif kind == "delayed_unrelated_opponent_contact":
                opponent_x, opponent_speed, opponent_y = 390.0 + jitter, -720.0, 0.0
            elif kind == "uncontested_loose_ball_flip_through":
                opponent_x, opponent_speed, opponent_y = 1800.0, 0.0, 1000.0
            elif kind == "self_not_converging":
                state.ball_vel[index] = (800.0, 0.0, 0.0)
                opponent_x, opponent_speed, opponent_y = 0.0, 0.0, 350.0
            else:
                opponent_x, opponent_speed, opponent_y = 0.0, 0.0, 250.0
        else:
            kind = (
                "ordinary_uncontested_forward_flip",
                "ordinary_side_of_play_opponent",
                "ordinary_parked_opponent",
                "ordinary_trailing_opponent",
            )[local % 4]
            if kind == "ordinary_side_of_play_opponent":
                opponent_x, opponent_speed, opponent_y = 100.0, 0.0, 700.0
            elif kind == "ordinary_parked_opponent":
                opponent_x, opponent_speed, opponent_y = 650.0, 0.0, 0.0
            elif kind == "ordinary_trailing_opponent":
                opponent_x, opponent_speed, opponent_y = -850.0, 500.0, 0.0
            else:
                opponent_x, opponent_speed, opponent_y = 2200.0, 0.0, -1000.0
        state.car_pos[index, 1] = (opponent_x, opponent_y, 48.0)
        state.car_vel[index, 1] = (opponent_speed, 0.0, 0.0)
        if kind == "nearby_nonconverging_opponent":
            state.car_vel[index, 1] = (0.0, opponent_y, 0.0)
        elif kind == "self_not_converging":
            state.car_vel[index, 1] = (0.0, -900.0, 0.0)
        elif kind == "mismatched_arrival_times":
            state.car_vel[index, 1] = (0.0, -300.0, 0.0)
        state.car_quat[index, 1] = _quat_from_euler(0.0, 0.0, math.pi)
        row["scenario"] = {
            "kind": kind,
            "initial_self_position": state.car_pos[index, 0].tolist(),
            "initial_self_velocity": state.car_vel[index, 0].tolist(),
            "initial_opponent_position": state.car_pos[index, 1].tolist(),
            "initial_opponent_velocity": state.car_vel[index, 1].tolist(),
            "initial_ball_position": state.ball_pos[index].tolist(),
            "initial_ball_velocity": state.ball_vel[index].tolist(),
        }
        row["action_sequence_identity"] = "self_forward_directional_dodge_at_tick_0"

    def controller(tick: int, controls: ControlBatch) -> None:
        if tick == 0:
            controls.jump[:, 0] = 1
            controls.pitch[:, 0] = dodge_pitch

    trace = _run_dual(state, 90, collision_root, geometry, meshes, controller)
    for index, row in enumerate(rows):
        self_ticks = _onsets(trace["hit"][:, index, 0])
        opponent_ticks = _onsets(trace["hit"][:, index, 1])
        self_tick = int(self_ticks[0]) if self_ticks.size else -1
        if self_tick >= 0:
            opponent_after = opponent_ticks[opponent_ticks >= self_tick]
            opponent_before = opponent_ticks[opponent_ticks < self_tick]
            candidates = []
            if opponent_after.size:
                candidates.append(int(opponent_after[0]))
            if opponent_before.size:
                candidates.append(int(opponent_before[-1]))
            opponent_tick = (
                min(candidates, key=lambda value: abs(value - self_tick))
                if candidates
                else -1
            )
            # Convergence is a pre-contact identity.  Post-contact velocity is
            # already changed by the very collision being classified and can
            # report a genuine challenger as moving away.  These are the
            # authoritative immediately preceding simulator samples; the
            # production correction consumes the collision pair's pre-car
            # velocity and an explicit previous-ball sample equivalently.
            ball = trace["ball_pos"][self_tick, index]
            ball_velocity = trace["pre_ball_vel"][self_tick, index, 0] * 50.0
            self_position = trace["car_pos"][self_tick, index, 0]
            opponent_position = trace["car_pos"][self_tick, index, 1]
            self_velocity = trace["pre_car_vel"][self_tick, index, 0] * 50.0
            opponent_velocity = trace["pre_car_vel"][self_tick, index, 1] * 50.0
            self_direction = _unit(ball - self_position)
            opponent_direction = _unit(ball - opponent_position)
            self_distance = float(np.linalg.norm(ball - self_position))
            opponent_distance = float(np.linalg.norm(ball - opponent_position))
            self_closing = float(np.dot(self_velocity - ball_velocity, self_direction))
            opponent_closing = float(
                np.dot(opponent_velocity - ball_velocity, opponent_direction)
            )
            self_ttb = self_distance / max(self_closing, 1.0e-6)
            opponent_ttb = opponent_distance / max(opponent_closing, 1.0e-6)
            separation = abs(opponent_tick - self_tick) if opponent_tick >= 0 else 999
            displacement = (
                float(
                    np.linalg.norm(
                        trace["ball_pos"][opponent_tick, index]
                        - trace["ball_pos"][self_tick, index]
                    )
                )
                if opponent_tick >= 0
                else 9999.0
            )
            active_dodge = bool(
                trace["is_flipping"][self_tick, index, 0]
                and trace["has_flipped"][self_tick, index, 0]
                and np.linalg.norm(trace["flip_torque"][self_tick, index, 0]) > 0.25
            )
        else:
            opponent_tick = -1
            self_distance = opponent_distance = 9999.0
            self_closing = opponent_closing = -9999.0
            self_ttb = opponent_ttb = 1.0e12
            separation = 999
            displacement = 9999.0
            active_dodge = False
        row["measured_contact_ticks"] = {
            "self_onsets": self_ticks.astype(int).tolist(),
            "opponent_onsets": opponent_ticks.astype(int).tolist(),
            "selected_self": self_tick,
            "selected_opponent": opponent_tick,
        }
        row["features"] = {
            "legitimate_self_contact": float(self_tick >= 0),
            "active_directional_dodge": float(active_dodge),
            "self_ball_distance": self_distance,
            "opponent_ball_distance": opponent_distance,
            "self_closing_speed": self_closing,
            "opponent_closing_speed": opponent_closing,
            "self_time_to_ball": self_ttb,
            "opponent_time_to_ball": opponent_ttb,
            "time_to_ball_delta": abs(self_ttb - opponent_ttb),
            "adjacent_contact_separation_ticks": float(separation),
            "adjacent_ball_displacement": displacement,
        }
    return rows


def _dodge_contact_cases(
    classifier: str,
    rows: list[dict[str, Any]],
    collision_root: str,
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
) -> list[dict[str, Any]]:
    count = len(rows)
    state = _base_state(count)
    state.on_ground[:, 0] = 0
    state.has_jumped[:, 0] = 1
    state.air_time[:, 0] = 0.2
    state.air_time_since_jump[:, 0] = 0.2
    dodge_pitch = np.zeros(count, dtype=np.float32)
    dodge_yaw = np.zeros(count, dtype=np.float32)
    dodge_tick = np.zeros(count, dtype=np.int32)
    for index, row in enumerate(rows):
        local = int(row["class_index"])
        rng = np.random.default_rng(int(row["seed"]))
        phase = float(rng.uniform(-1.0, 1.0))
        origin = local % 4
        yaw = (-0.45, -0.12, 0.28, 0.55)[origin] + phase * 0.004
        quaternion = _quat_from_euler(0.0, 0.0, yaw)
        height = (620.0, 980.0, 1320.0, 860.0)[origin] + phase * 2.0
        local_speed = (320.0, 520.0, 690.0, 430.0)[origin] + phase * 2.0
        car_velocity = _quat_rotate(
            quaternion,
            np.asarray((local_speed, ((local % 3) - 1) * 18.0, 0.0), dtype=np.float32),
        )
        state.car_pos[index, 0] = (0.0, 0.0, height)
        state.car_quat[index, 0] = quaternion
        state.car_vel[index, 0] = car_velocity
        if row["class"] == "positive":
            if classifier == "controlled_flick":
                kind = (
                    "front_controlled_release",
                    "diagonal_left_controlled_release",
                    "diagonal_right_controlled_release",
                    "side_left_controlled_release",
                    "side_right_controlled_release",
                    "stable_roof_control_origin",
                    "stable_rear_control_origin",
                    "stable_nose_control_origin",
                )[local % 8]
                dodge_tick[index] = 8 + local % 5
                if kind == "front_controlled_release":
                    dodge_pitch[index], dodge_yaw[index] = -1.0, 0.0
                    offset = np.asarray((132.0, 0.0, 150.0), dtype=np.float32)
                elif kind == "diagonal_left_controlled_release":
                    dodge_pitch[index], dodge_yaw[index] = -0.72, -0.72
                    offset = np.asarray((92.0, -92.0, 145.0), dtype=np.float32)
                elif kind == "diagonal_right_controlled_release":
                    dodge_pitch[index], dodge_yaw[index] = -0.72, 0.72
                    offset = np.asarray((92.0, 92.0, 145.0), dtype=np.float32)
                elif kind == "side_left_controlled_release":
                    dodge_pitch[index], dodge_yaw[index] = 0.0, -1.0
                    offset = np.asarray((0.0, -132.0, 145.0), dtype=np.float32)
                elif kind == "side_right_controlled_release":
                    dodge_pitch[index], dodge_yaw[index] = 0.0, 1.0
                    offset = np.asarray((0.0, 132.0, 145.0), dtype=np.float32)
                elif kind == "stable_nose_control_origin":
                    dodge_pitch[index], dodge_yaw[index] = -1.0, 0.0
                    offset = np.asarray((108.0, 0.0, 142.0), dtype=np.float32)
                else:
                    dodge_pitch[index] = 1.0
                    offset = np.asarray(
                        (-110.0 if kind == "stable_rear_control_origin" else -70.0, 0.0, 150.0),
                        dtype=np.float32,
                    )
                relative_velocity = np.asarray(
                    ((local % 3 - 1) * 4.0, (local % 5 - 2) * 2.0, 0.0),
                    dtype=np.float32,
                )
            else:
                dodge_pitch[index] = 1.0
                dodge_yaw[index] = (0.0, -0.22, 0.22)[local % 3]
                offset = np.asarray(MUSTY_OFFSETS[1 + local % 5], dtype=np.float32)
                offset[1] += (0.0, -10.0, 10.0)[local % 3]
                relative_velocity = np.asarray(
                    ((local % 4 - 1.5) * 8.0, (local % 3 - 1) * 5.0, 0.0),
                    dtype=np.float32,
                )
                kind = (
                    "offensive_dodge_powered_shot",
                    "defensive_dodge_powered_clear",
                    "rear_roof_contact_geometry",
                    "diagonal_contact_geometry",
                    "weak_real_rotational_power",
                )[local % 5]
        elif row["class"] == "near_miss":
            kinds = (
                (
                    "loose_ball_flip_through",
                    "kickoff_or_50_flip_contact",
                    "brief_near_car_relation",
                    "chase_contact",
                    "controlled_looking_no_dodge_release",
                    "directional_dodge_with_no_release",
                )
                if classifier == "controlled_flick"
                else (
                    "translation_dominated_high_speed_flip_hit",
                    "weak_ordinary_flip_touch",
                    "already_fast_ball_negligible_dodge",
                    "pre_sweep_contact",
                    "incoming_roof_slap",
                    "lateral_loose_ball_flip_hit",
                )
            )
            kind = kinds[local % 6]
            dodge_pitch[index] = 1.0
            if classifier == "controlled_flick":
                if kind == "loose_ball_flip_through":
                    offset = np.asarray((-520.0, 0.0, 135.0), dtype=np.float32)
                    relative_velocity = np.asarray((1700.0, 0.0, 0.0), dtype=np.float32)
                elif kind == "kickoff_or_50_flip_contact":
                    offset = np.asarray((-140.0, 0.0, 130.0), dtype=np.float32)
                    relative_velocity = np.asarray((0.0, 0.0, 0.0), dtype=np.float32)
                    state.car_pos[index, 1] = state.car_pos[index, 0] + _quat_rotate(
                        quaternion, np.asarray((-350.0, 0.0, 0.0), dtype=np.float32)
                    )
                    state.car_vel[index, 1] = car_velocity + _quat_rotate(
                        quaternion, np.asarray((800.0, 0.0, 0.0), dtype=np.float32)
                    )
                elif kind == "brief_near_car_relation":
                    dodge_tick[index] = 2
                    offset = np.asarray((-110.0, 0.0, 150.0), dtype=np.float32)
                    relative_velocity = np.asarray((0.0, 0.0, 0.0), dtype=np.float32)
                elif kind == "chase_contact":
                    offset = np.asarray((260.0, 0.0, 110.0), dtype=np.float32)
                    relative_velocity = np.asarray((-850.0, 0.0, 0.0), dtype=np.float32)
                elif kind == "controlled_looking_no_dodge_release":
                    dodge_pitch[index] = 0.0
                    dodge_tick[index] = 10
                    offset = np.asarray((-110.0, 0.0, 150.0), dtype=np.float32)
                    relative_velocity = np.asarray((0.0, 0.0, 0.0), dtype=np.float32)
                else:
                    dodge_tick[index] = 10
                    offset = np.asarray((-320.0, 0.0, 160.0), dtype=np.float32)
                    relative_velocity = np.asarray((0.0, 0.0, 0.0), dtype=np.float32)
            elif kind == "translation_dominated_high_speed_flip_hit":
                offset = np.asarray((-520.0, 0.0, 135.0), dtype=np.float32)
                relative_velocity = np.asarray((1700.0, 0.0, 0.0), dtype=np.float32)
            elif kind == "weak_ordinary_flip_touch":
                offset = np.asarray((180.0, 0.0, 108.0), dtype=np.float32)
                relative_velocity = np.asarray((-950.0, 0.0, 0.0), dtype=np.float32)
            elif kind == "already_fast_ball_negligible_dodge":
                offset = np.asarray((300.0, 0.0, 105.0), dtype=np.float32)
                relative_velocity = np.asarray((-1050.0, 0.0, 0.0), dtype=np.float32)
            elif kind == "pre_sweep_contact":
                offset = np.asarray((28.0 + local % 4, 0.0, 105.0), dtype=np.float32)
                relative_velocity = np.asarray((-250.0, 0.0, 0.0), dtype=np.float32)
            elif kind == "incoming_roof_slap":
                offset = np.asarray((0.0, 150.0, 150.0), dtype=np.float32)
                relative_velocity = np.asarray((0.0, -400.0, -400.0), dtype=np.float32)
            else:
                side = -1.0 if local % 2 else 1.0
                offset = np.asarray((-140.0, side * 150.0, 125.0), dtype=np.float32)
                relative_velocity = np.asarray((0.0, -side * 620.0, 0.0), dtype=np.float32)
        else:
            kind = (
                (
                    "ordinary_loose_ball_contact",
                    "ordinary_forward_flip_drive_through",
                    "ordinary_random_tumble_contact",
                    "ordinary_no_release_control_relation",
                )[local % 4]
                if classifier == "controlled_flick"
                else (
                    "normal_drive_through_contact",
                    "ordinary_forward_flip_drive_through",
                    "random_tumble_contact",
                )[local % 3]
            )
            offset = np.asarray((260.0, ((local % 5) - 2) * 16.0, 110.0), dtype=np.float32)
            relative_velocity = np.asarray((-850.0, 0.0, 0.0), dtype=np.float32)
            if kind == "ordinary_forward_flip_drive_through":
                dodge_pitch[index] = -1.0
            elif kind in ("random_tumble_contact", "ordinary_random_tumble_contact"):
                state.car_ang_vel[index, 0] = _quat_rotate(
                    quaternion, np.asarray((0.8, 2.0, 1.2), dtype=np.float32)
                )
            elif kind == "ordinary_no_release_control_relation":
                offset = np.asarray((-110.0, 0.0, 150.0), dtype=np.float32)
                relative_velocity = np.asarray((0.0, 0.0, 0.0), dtype=np.float32)
                dodge_tick[index] = 10
        state.ball_pos[index] = state.car_pos[index, 0] + _quat_rotate(quaternion, offset)
        state.ball_vel[index] = car_velocity + _quat_rotate(quaternion, relative_velocity)
        row["scenario"] = {
            "kind": kind,
            "control_origin": ("roof", "rear", "nose", "diagonal")[origin],
            "dodge_axis": [float(dodge_pitch[index]), float(dodge_yaw[index])],
            "dodge_tick": int(dodge_tick[index]),
            "initial_car_position": state.car_pos[index, 0].tolist(),
            "initial_car_velocity": state.car_vel[index, 0].tolist(),
            "initial_car_quaternion": state.car_quat[index, 0].tolist(),
            "initial_ball_position": state.ball_pos[index].tolist(),
            "initial_ball_velocity": state.ball_vel[index].tolist(),
            "ball_local_offset": offset.tolist(),
            "ball_local_relative_velocity": relative_velocity.tolist(),
        }
        row["action_sequence_identity"] = (
            f"directional_dodge_tick_{dodge_tick[index]}_pitch_{dodge_pitch[index]:+.2f}_"
            f"yaw_{dodge_yaw[index]:+.2f}"
            if dodge_pitch[index] != 0.0 or dodge_yaw[index] != 0.0
            else "zero_controls_all_ticks"
        )

    def controller(tick: int, controls: ControlBatch) -> None:
        active = ((dodge_pitch != 0.0) | (dodge_yaw != 0.0)) & (dodge_tick == tick)
        controls.jump[active, 0] = 1
        controls.pitch[active, 0] = dodge_pitch[active]
        controls.yaw[active, 0] = dodge_yaw[active]

    trace = _run(state, 70, collision_root, geometry, meshes, controller)
    touch_onsets = _touch_onsets(trace["hit"])
    for index, row in enumerate(rows):
        touches = touch_onsets[index]
        tick = int(touches[0]) if touches.size else -1
        if tick >= 0:
            power = _musty_contact_features(
                trace, index, tick, dodge_tick=int(dodge_tick[index])
            )
            contact_world_array = trace["contact_point"][tick, index] * 50.0
            contact_offset = contact_world_array - trace["car_pos"][tick, index]
            direction = _unit(
                trace["ball_pos"][tick, index] - trace["car_pos"][tick, index]
            )
            pre_velocity_array = trace["pre_car_vel"][tick, index] * 50.0
            pre_ball_velocity_array = trace["pre_ball_vel"][tick, index] * 50.0
            rotational_velocity = np.cross(
                trace["pre_car_ang"][tick, index], contact_offset
            )
            rotational = max(float(np.dot(rotational_velocity, direction)), 0.0)
            translational = max(
                float(np.dot(pre_velocity_array - pre_ball_velocity_array, direction)),
                0.0,
            )
            total = rotational + translational
            control_stop = min(int(dodge_tick[index]), tick)
            distances = np.linalg.norm(
                trace["ball_pos"][: control_stop + 1, index]
                - trace["car_pos"][: control_stop + 1, index],
                axis=1,
            )
            relative_speeds = np.linalg.norm(
                trace["ball_vel"][: control_stop + 1, index]
                - trace["car_vel"][: control_stop + 1, index],
                axis=1,
            )
            release_tick = min(tick + 2, trace["ball_pos"].shape[0] - 1)
            release_distance = float(
                np.linalg.norm(
                    trace["ball_pos"][release_tick, index]
                    - trace["car_pos"][release_tick, index]
                )
            )
            active_dodge = bool(
                trace["is_flipping"][tick, index]
                and trace["has_flipped"][tick, index]
                and np.linalg.norm(trace["flip_torque"][tick, index]) > 0.25
            )
            contact_point = contact_world_array.tolist()
            contact_normal = trace["contact_normal"][tick, index].tolist()
            v_linear = pre_velocity_array.tolist()
            omega_cross_r = rotational_velocity.tolist()
        else:
            power = _empty_musty_features()
            rotational = translational = total = 0.0
            distances = np.asarray((9999.0,), dtype=np.float32)
            relative_speeds = np.asarray((9999.0,), dtype=np.float32)
            release_tick = -1
            release_distance = 0.0
            active_dodge = False
            contact_point = contact_normal = v_linear = omega_cross_r = [0.0, 0.0, 0.0]
        row["measured_contact_ticks"] = {
            "self_onsets": touches.astype(int).tolist(),
            "selected_contact": tick,
            "release_measurement": release_tick,
        }
        row["features"] = {
            "legitimate_contact": power["legitimate_contact"],
            "active_directional_dodge": float(active_dodge),
            "total_closing_speed": total,
            "translational_closing_contribution": translational,
            "rotational_closing_contribution": rotational,
            "rotational_share": rotational / max(total, 1.0e-8),
            "ball_delta_v": power["ball_delta_v"],
            "contact_point_world": contact_point,
            "contact_normal": contact_normal,
            "v_linear": v_linear,
            "omega_cross_r": omega_cross_r,
            "precontact_observed_ticks": float(len(distances)),
            "precontact_max_distance": float(np.max(distances)),
            "precontact_max_relative_speed": float(np.max(relative_speeds)),
            "precontact_distance_series": distances.astype(float).tolist(),
            "precontact_relative_speed_series": relative_speeds.astype(float).tolist(),
            "release_distance": release_distance,
            "release_ball_delta_v": power["ball_delta_v"],
        }
    return rows


def _generate_split(
    split: str, collision_root: str, provenance: dict[str, Any]
) -> list[dict[str, Any]]:
    geometry = ArenaGeometry.load_soccar(collision_root)
    meshes = WarpArenaMeshes(geometry)
    generated: list[dict[str, Any]] = []
    for classifier in CLASSIFIERS:
        rows = _rows(classifier, split)
        if classifier == "contest":
            rows = _contest_cases(rows, collision_root, geometry, meshes)
        else:
            rows = _dodge_contact_cases(classifier, rows, collision_root, geometry, meshes)
        for row in rows:
            row["provenance"] = provenance
        generated.extend(rows)
    return generated


def _feature_values(
    rows: list[dict[str, Any]], feature: str, *, kinds: tuple[str, ...] | None = None
) -> list[float]:
    selected = rows
    if kinds is not None:
        selected = [row for row in rows if row["scenario"]["kind"] in kinds]
    return [float(row["features"][feature]) for row in selected]


def _boundary(
    *,
    name: str,
    feature: str,
    direction: str,
    positives: list[dict[str, Any]],
    negatives: list[dict[str, Any]],
    positive_kinds: tuple[str, ...] | None = None,
    negative_kinds: tuple[str, ...] | None = None,
    integer: bool = False,
) -> dict[str, Any]:
    positive_values = _feature_values(positives, feature, kinds=positive_kinds)
    negative_values = _feature_values(negatives, feature, kinds=negative_kinds)
    if not positive_values or not negative_values:
        raise RuntimeError(f"{name}: empty prospective boundary scope")
    if direction == "min":
        positive_edge = min(positive_values)
        negative_edge = max(negative_values)
        margin = positive_edge - negative_edge
    elif direction == "max":
        positive_edge = max(positive_values)
        negative_edge = min(negative_values)
        margin = negative_edge - positive_edge
    else:
        raise ValueError(direction)
    if not math.isfinite(margin) or margin <= 0.0:
        raise RuntimeError(
            f"{name}: no clean derivation separation; positive_edge={positive_edge}, "
            f"negative_edge={negative_edge}"
        )
    midpoint = (positive_edge + negative_edge) * 0.5
    threshold: float | int = midpoint
    if integer:
        threshold = math.floor(midpoint) if direction == "max" else math.ceil(midpoint)
        if direction == "max" and not (positive_edge <= threshold < negative_edge):
            raise RuntimeError(f"{name}: no clean integer max threshold")
        if direction == "min" and not (negative_edge < threshold <= positive_edge):
            raise RuntimeError(f"{name}: no clean integer min threshold")
    return {
        "name": name,
        "feature": feature,
        "direction": direction,
        "positive_scope": list(positive_kinds or ("all_positive",)),
        "negative_scope": list(negative_kinds or ("all_negative",)),
        "positive_values": positive_values,
        "negative_values": negative_values,
        "positive_edge": positive_edge,
        "negative_edge": negative_edge,
        "margin": margin,
        "midpoint": midpoint,
        "selected_threshold": threshold,
        "integer_threshold": integer,
    }


def _derive(rows: list[dict[str, Any]], provenance: dict[str, Any]) -> dict[str, Any]:
    by_classifier = {
        name: [row for row in rows if row["classifier"] == name]
        for name in CLASSIFIERS
    }
    payload: dict[str, Any] = {
        "format": DERIVATION_FORMAT,
        "created_utc": _utc_now(),
        "split": "derivation_only",
        "heldout_generated_or_inspected": False,
        "thresholds_frozen_before_heldout": True,
        "threshold_selection": (
            "midpoint of prospectively scoped nearest clean positive and negative edges"
        ),
        "provenance": provenance,
        "classifiers": {},
    }

    contest = by_classifier["contest"]
    contest_positive = [row for row in contest if row["class"] == "positive"]
    contest_negative = [row for row in contest if row["class"] != "positive"]
    adjacent_positive = (
        "simultaneous_two_car_50",
        "opponent_contact_just_before_self",
        "opponent_contact_just_after_self",
    )
    convergence_positive = ("genuine_converging_challenge",)
    delayed_negative = ("delayed_unrelated_opponent_contact",)
    contest_boundaries = [
        _boundary(
            name="CONTEST_CONTACT_WINDOW_TICKS",
            feature="adjacent_contact_separation_ticks",
            direction="max",
            positives=contest_positive,
            negatives=contest_negative,
            positive_kinds=adjacent_positive,
            negative_kinds=delayed_negative,
            integer=True,
        ),
        _boundary(
            name="CONTEST_ASSOCIATION_BALL_DISPLACEMENT_MAX",
            feature="adjacent_ball_displacement",
            direction="max",
            positives=contest_positive,
            negatives=contest_negative,
            positive_kinds=adjacent_positive,
            negative_kinds=delayed_negative,
        ),
        _boundary(
            name="CONTEST_OPPONENT_DISTANCE_MAX",
            feature="opponent_ball_distance",
            direction="max",
            positives=contest_positive,
            negatives=contest_negative,
            positive_kinds=convergence_positive,
            negative_kinds=("distant_opponent", "opponent_behind_play"),
        ),
        _boundary(
            name="CONTEST_SELF_CLOSING_SPEED_MIN",
            feature="self_closing_speed",
            direction="min",
            positives=contest_positive,
            negatives=contest_negative,
            positive_kinds=convergence_positive,
            negative_kinds=("self_not_converging",),
        ),
        _boundary(
            name="CONTEST_OPPONENT_CLOSING_SPEED_MIN",
            feature="opponent_closing_speed",
            direction="min",
            positives=contest_positive,
            negatives=contest_negative,
            positive_kinds=convergence_positive,
            negative_kinds=(
                "nearby_opponent_moving_away",
                "nearby_nonconverging_opponent",
                "mismatched_arrival_times",
            ),
        ),
        _boundary(
            name="CONTEST_TIME_TO_BALL_DELTA_MAX",
            feature="time_to_ball_delta",
            direction="max",
            positives=contest_positive,
            negatives=contest_negative,
            positive_kinds=convergence_positive,
            negative_kinds=(
                "delayed_unrelated_opponent_contact",
                "mismatched_arrival_times",
            ),
        ),
    ]

    power = by_classifier["power_contact"]
    power_positive = [row for row in power if row["class"] == "positive"]
    power_negative = [row for row in power if row["class"] != "positive"]
    power_boundaries = [
        _boundary(
            name="POWER_TOTAL_CLOSING_SPEED_MIN",
            feature="total_closing_speed",
            direction="min",
            positives=power_positive,
            negatives=power_negative,
            negative_kinds=("pre_sweep_contact",),
        ),
        _boundary(
            name="POWER_ROTATIONAL_CLOSING_SPEED_MIN",
            feature="rotational_closing_contribution",
            direction="min",
            positives=power_positive,
            negatives=power_negative,
            negative_kinds=("weak_ordinary_flip_touch",),
        ),
        _boundary(
            name="POWER_ROTATIONAL_SHARE_MIN",
            feature="rotational_share",
            direction="min",
            positives=power_positive,
            negatives=power_negative,
            negative_kinds=(
                "translation_dominated_high_speed_flip_hit",
                "incoming_roof_slap",
                "ordinary_forward_flip_drive_through",
            ),
        ),
        _boundary(
            name="POWER_BALL_DELTA_V_MIN",
            feature="ball_delta_v",
            direction="min",
            positives=power_positive,
            negatives=power_negative,
            negative_kinds=("weak_ordinary_flip_touch",),
        ),
    ]

    controlled = by_classifier["controlled_flick"]
    controlled_positive = [row for row in controlled if row["class"] == "positive"]
    controlled_negative = [row for row in controlled if row["class"] != "positive"]
    controlled_boundaries = [
        _boundary(
            name="CONTROL_HISTORY_TICKS_MIN",
            feature="precontact_observed_ticks",
            direction="min",
            positives=controlled_positive,
            negatives=controlled_negative,
            negative_kinds=(
                "brief_near_car_relation",
                "kickoff_or_50_flip_contact",
                "ordinary_forward_flip_drive_through",
            ),
            integer=True,
        ),
        _boundary(
            name="CONTROL_DISTANCE_MAX",
            feature="precontact_max_distance",
            direction="max",
            positives=controlled_positive,
            negatives=controlled_negative,
            negative_kinds=(
                "loose_ball_flip_through",
                "directional_dodge_with_no_release",
                "ordinary_loose_ball_contact",
            ),
        ),
        _boundary(
            name="CONTROL_RELATIVE_SPEED_MAX",
            feature="precontact_max_relative_speed",
            direction="max",
            positives=controlled_positive,
            negatives=controlled_negative,
            negative_kinds=(
                "loose_ball_flip_through",
                "ordinary_loose_ball_contact",
                "ordinary_forward_flip_drive_through",
            ),
        ),
        _boundary(
            name="CONTROL_RELEASE_DISTANCE_MIN",
            feature="release_distance",
            direction="min",
            positives=controlled_positive,
            negatives=controlled_negative,
            negative_kinds=(
                "controlled_looking_no_dodge_release",
                "ordinary_no_release_control_relation",
            ),
        ),
        _boundary(
            name="CONTROL_RELEASE_BALL_DELTA_V_MIN",
            feature="release_ball_delta_v",
            direction="min",
            positives=controlled_positive,
            negatives=controlled_negative,
            negative_kinds=(
                "controlled_looking_no_dodge_release",
                "ordinary_no_release_control_relation",
            ),
        ),
    ]

    for classifier, boundaries in (
        ("contest", contest_boundaries),
        ("power_contact", power_boundaries),
        ("controlled_flick", controlled_boundaries),
    ):
        classifier_rows = by_classifier[classifier]
        feature_names = sorted(
            {
                boundary["feature"]
                for boundary in boundaries
            }
        )
        payload["classifiers"][classifier] = {
            "derivation_counts": {
                name: sum(row["class"] == name for row in classifier_rows)
                for name in CLASS_NAMES
            },
            "boundaries": boundaries,
            "all_measured_derivation_values": {
                feature: {
                    name: _feature_values(
                        [row for row in classifier_rows if row["class"] == name], feature
                    )
                    for name in CLASS_NAMES
                }
                for feature in feature_names
            },
        }
    payload["thresholds"] = {
        boundary["name"]: boundary["selected_threshold"]
        for classifier in payload["classifiers"].values()
        for boundary in classifier["boundaries"]
    }
    payload["frozen_thresholds_sha256"] = _canonical_sha256(payload["thresholds"])
    return payload


def _classify(row: dict[str, Any], thresholds: dict[str, float | int]) -> bool:
    features = row["features"]
    legitimate = features.get("legitimate_contact", features.get("legitimate_self_contact", 0.0))
    if not legitimate or not features["active_directional_dodge"]:
        return False
    if row["classifier"] == "contest":
        adjacent = (
            features["adjacent_contact_separation_ticks"]
            <= thresholds["CONTEST_CONTACT_WINDOW_TICKS"]
            and features["adjacent_ball_displacement"]
            <= thresholds["CONTEST_ASSOCIATION_BALL_DISPLACEMENT_MAX"]
        )
        convergence = (
            features["opponent_ball_distance"]
            <= thresholds["CONTEST_OPPONENT_DISTANCE_MAX"]
            and features["self_closing_speed"]
            >= thresholds["CONTEST_SELF_CLOSING_SPEED_MIN"]
            and features["opponent_closing_speed"]
            >= thresholds["CONTEST_OPPONENT_CLOSING_SPEED_MIN"]
            and features["time_to_ball_delta"]
            <= thresholds["CONTEST_TIME_TO_BALL_DELTA_MAX"]
        )
        return bool(adjacent or convergence)
    if row["classifier"] == "power_contact":
        return bool(
            features["total_closing_speed"]
            >= thresholds["POWER_TOTAL_CLOSING_SPEED_MIN"]
            and features["rotational_closing_contribution"]
            >= thresholds["POWER_ROTATIONAL_CLOSING_SPEED_MIN"]
            and features["rotational_share"] >= thresholds["POWER_ROTATIONAL_SHARE_MIN"]
            and features["ball_delta_v"] >= thresholds["POWER_BALL_DELTA_V_MIN"]
        )
    return bool(
        features["precontact_observed_ticks"] >= thresholds["CONTROL_HISTORY_TICKS_MIN"]
        and features["precontact_max_distance"] <= thresholds["CONTROL_DISTANCE_MAX"]
        and features["precontact_max_relative_speed"]
        <= thresholds["CONTROL_RELATIVE_SPEED_MAX"]
        and features["release_distance"] >= thresholds["CONTROL_RELEASE_DISTANCE_MIN"]
        and features["release_ball_delta_v"]
        >= thresholds["CONTROL_RELEASE_BALL_DELTA_V_MIN"]
    )


def _confusion(
    rows: list[dict[str, Any]], thresholds: dict[str, float | int]
) -> dict[str, Any]:
    predictions = []
    tp = tn = fp = fn = 0
    for row in rows:
        predicted = _classify(row, thresholds)
        expected = row["class"] == "positive"
        row["classified_positive"] = predicted
        row["expected_positive"] = expected
        predictions.append(
            {
                "scenario_id": row["scenario_id"],
                "class": row["class"],
                "scenario_kind": row["scenario"]["kind"],
                "expected_positive": expected,
                "classified_positive": predicted,
            }
        )
        tp += int(expected and predicted)
        tn += int(not expected and not predicted)
        fp += int(not expected and predicted)
        fn += int(expected and not predicted)
    return {
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "predictions": predictions,
        "pass": fp == 0 and fn == 0,
    }


def _derive_phase(collision_root: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = _provenance()
    rows = _generate_split("derivation", collision_root, provenance)
    derivation = _derive(rows, provenance)
    thresholds = derivation["thresholds"]
    derivation["derivation_confusion"] = {
        classifier: _confusion(
            [row for row in rows if row["classifier"] == classifier], thresholds
        )
        for classifier in CLASSIFIERS
    }
    failed = [
        name
        for name, result in derivation["derivation_confusion"].items()
        if not result["pass"]
    ]
    if failed:
        raise RuntimeError(f"derivation classification failed: {failed}")
    _write_jsonl(output_dir / "classifier_trace_corpus.derivation.jsonl", rows)
    _write_json(output_dir / "classifier_threshold_derivation.json", derivation)
    print(json.dumps(derivation["thresholds"], indent=2, sort_keys=True))


def _heldout_phase(collision_root: str, output_dir: Path) -> None:
    derivation_path = output_dir / "classifier_threshold_derivation.json"
    derivation_rows_path = output_dir / "classifier_trace_corpus.derivation.jsonl"
    if not derivation_path.is_file() or not derivation_rows_path.is_file():
        raise RuntimeError("heldout requires completed frozen derivation artifacts")
    derivation_bytes_sha = _sha256(derivation_path)
    derivation = json.loads(derivation_path.read_text(encoding="utf-8"))
    if not derivation.get("thresholds_frozen_before_heldout"):
        raise RuntimeError("derivation artifact is not prospectively frozen")
    thresholds = derivation["thresholds"]
    if _canonical_sha256(thresholds) != derivation["frozen_thresholds_sha256"]:
        raise RuntimeError("frozen threshold digest mismatch")
    provenance = _provenance()
    if provenance != derivation["provenance"]:
        raise RuntimeError("source identity changed after derivation freeze")
    rows = _generate_split("heldout", collision_root, provenance)
    heldout = {
        "format": "RIVAL2_GAMEPLAY_V3_CLASSIFIER_HELDOUT_V1",
        "created_utc": _utc_now(),
        "thresholds_frozen_before_heldout": True,
        "heldout_retuning": False,
        "frozen_thresholds_sha256": derivation["frozen_thresholds_sha256"],
        "derivation_artifact_sha256_before_heldout": derivation_bytes_sha,
        "confusion": {
            classifier: _confusion(
                [row for row in rows if row["classifier"] == classifier], thresholds
            )
            for classifier in CLASSIFIERS
        },
        "provenance": provenance,
    }
    failed = [
        name for name, result in heldout["confusion"].items() if not result["pass"]
    ]
    if failed:
        _write_json(output_dir / "classifier_heldout.json", heldout)
        raise RuntimeError(f"untouched heldout classification failed: {failed}")
    derivation_rows = [
        json.loads(line)
        for line in derivation_rows_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    _write_jsonl(output_dir / "classifier_trace_corpus.jsonl", derivation_rows + rows)
    _write_json(output_dir / "classifier_heldout.json", heldout)
    print(json.dumps({key: value["pass"] for key, value in heldout["confusion"].items()}))


class _SourceExactHarness:
    """Small authoritative-array harness for the production V3 kernel."""

    def __init__(self, collision_root: str):
        self.env = Rival2Env(
            1,
            collision_root,
            reward_version=RIVAL2_REWARD_GAMEPLAY_V3_VERSION,
        )
        self.world = self.env.world
        self.v3 = self.world.gameplay_v3
        if self.v3 is None:
            raise RuntimeError("Gameplay V3 state was not constructed")
        self._clear_inputs()

    @staticmethod
    def _torch(array: wp.array) -> torch.Tensor:
        return wp.to_torch(array)

    def _clear_inputs(self) -> None:
        state = self.world.state
        for array in (
            state.on_ground,
            state.has_jumped,
            state.has_double_jumped,
            state.has_flipped,
            state.is_flipping,
        ):
            self._torch(array).zero_()
        self._torch(state.air_time).zero_()
        self._torch(state.flip_rel_torque).zero_()
        self._torch(state.car_vel).zero_()
        self._torch(self.world.vehicle.wheel_contact).zero_()
        self._torch(self.world.vehicle.wheel_hit_normal).zero_()
        self._torch(self.world.vehicle.wheel_hit_face).fill_(-1)
        self._torch(self.world.car_ball.hit_this_tick).zero_()
        self._torch(self.world.car_ball_b.hit_this_tick).zero_()
        torch.cuda.synchronize(self.env.device)

    def prime_dash(
        self,
        *,
        air_ticks: int,
        pre_speed: float,
        previous_has_jumped: int = 1,
    ) -> None:
        self._torch(self.v3.dash_previous_on_ground)[0] = 0
        self._torch(self.v3.dash_previous_has_jumped)[0] = previous_has_jumped
        self._torch(self.v3.dash_previous_has_flipped)[0] = 0
        self._torch(self.v3.dash_previous_air_time)[0] = air_ticks / 120.0
        self._torch(self.v3.dash_previous_wheel_mask)[0] = 0
        self._torch(self.v3.dash_previous_velocity)[0] = torch.tensor(
            (pre_speed, 0.0, 0.0), device=self.env.device
        )

    def launch(
        self,
        tick: int,
        *,
        velocity: float = 0.0,
        has_jumped: int = 0,
        has_double_jumped: int = 0,
        has_flipped: int = 0,
        is_flipping: int = 0,
        air_time: float = 0.0,
        torque: tuple[float, float, float] = (0.0, 0.0, 0.0),
        wheel_faces: tuple[int, ...] = (),
    ) -> None:
        self._clear_inputs()
        self._torch(self.world.rival2.episode_ticks)[0] = tick - 1
        self._torch(self.world.state.car_vel)[0] = torch.tensor(
            (velocity, 0.0, 0.0), device=self.env.device
        )
        self._torch(self.world.state.has_jumped)[0] = has_jumped
        self._torch(self.world.state.has_double_jumped)[0] = has_double_jumped
        self._torch(self.world.state.has_flipped)[0] = has_flipped
        self._torch(self.world.state.is_flipping)[0] = is_flipping
        self._torch(self.world.state.air_time)[0] = air_time
        self._torch(self.world.state.flip_rel_torque)[0] = torch.tensor(
            torque, device=self.env.device
        )
        wheel_contact = self._torch(self.world.vehicle.wheel_contact)
        wheel_normal = self._torch(self.world.vehicle.wheel_hit_normal)
        wheel_face = self._torch(self.world.vehicle.wheel_hit_face)
        for wheel, face in enumerate(wheel_faces):
            wheel_contact[wheel] = 1
            wheel_normal[wheel] = torch.tensor((0.0, 0.0, 1.0), device=self.env.device)
            wheel_face[wheel] = face
        self.v3.launch_tick()
        torch.cuda.synchronize(self.env.device)

    def value(self, name: str, index: int = 0) -> int | float:
        value = self._torch(getattr(self.v3, name))[index].item()
        return int(value) if isinstance(value, int) else float(value)


def _dash_case(
    collision_root: str,
    *,
    air_ticks: int,
    landing_delay: int,
    gain: float,
    fresh_jump: bool = False,
) -> dict[str, Any]:
    harness = _SourceExactHarness(collision_root)
    harness.prime_dash(
        air_ticks=air_ticks,
        pre_speed=100.0,
        previous_has_jumped=0 if fresh_jump else 1,
    )
    harness.launch(
        10,
        velocity=100.0,
        has_jumped=1,
        has_flipped=1,
        is_flipping=1,
        air_time=air_ticks / 120.0,
        torque=(0.0, 1.0, 0.0),
    )
    harness.launch(
        10 + landing_delay,
        velocity=100.0 + gain,
        has_jumped=1,
        has_flipped=1,
        wheel_faces=(0, 0, 0, 0),
    )
    return {
        "dash_success_total": harness.value("dash_success_total"),
        "interval_requested": harness.value("interval_requested"),
        "pending_flip_tick": harness.value("dash_pending_flip_tick"),
    }


def _reset_case(
    collision_root: str,
    *,
    face: int,
    support_wheels: int,
    previous_untimed: int = 0,
    previous_has_flipped: int = 0,
) -> tuple[_SourceExactHarness, dict[str, Any]]:
    harness = _SourceExactHarness(collision_root)
    harness._torch(harness.v3.reset_previous_untimed)[0] = previous_untimed
    harness._torch(harness.v3.reset_previous_has_flipped)[0] = previous_has_flipped
    harness._torch(harness.v3.reset_armed)[0] = 1
    harness.launch(10, wheel_faces=tuple(face for _ in range(support_wheels)))
    pending_after_support = harness.value("reset_pending_body")
    harness.launch(11)
    result = {
        "pending_after_support": pending_after_support,
        "reset_completion_total": harness.value("reset_completion_total"),
        "interval_requested": harness.value("interval_requested"),
        "chain_reset_total": harness.value("chain_reset_total"),
        "preflip_reset_total": harness.value("preflip_reset_total"),
    }
    return harness, result


def run_dash_reset_source_exact(collision_root: str) -> dict[str, Any]:
    """Execute every required dash/reset rule against production state arrays."""

    cases: dict[str, dict[str, Any]] = {}

    valid = _dash_case(collision_root, air_ticks=42, landing_delay=24, gain=1.01)
    cases["dash_gain_strictly_greater_than_one_accepts"] = {
        "observed": valid,
        "pass": valid["dash_success_total"] == 1 and valid["interval_requested"] == 1,
    }
    near = _dash_case(collision_root, air_ticks=42, landing_delay=24, gain=1.0)
    cases["dash_gain_at_most_one_rejects"] = {
        "observed": near,
        "pass": near["dash_success_total"] == 0 and near["interval_requested"] == 0,
    }
    long_air = _dash_case(collision_root, air_ticks=43, landing_delay=1, gain=20.0)
    cases["dash_air_over_42_ticks_rejects"] = {
        "observed": long_air,
        "pass": long_air["dash_success_total"] == 0,
    }
    late = _dash_case(collision_root, air_ticks=42, landing_delay=25, gain=20.0)
    cases["dash_landing_over_24_ticks_rejects"] = {
        "observed": late,
        "pass": late["dash_success_total"] == 0,
    }
    fresh = _dash_case(
        collision_root,
        air_ticks=42,
        landing_delay=1,
        gain=2.0,
        fresh_jump=True,
    )
    cases["dash_has_no_fresh_jump_prohibition"] = {
        "observed": fresh,
        "pass": fresh["dash_success_total"] == 1,
    }

    double = _SourceExactHarness(collision_root)
    double.prime_dash(air_ticks=20, pre_speed=100.0)
    double.launch(
        10,
        velocity=100.0,
        has_jumped=1,
        has_flipped=1,
        is_flipping=1,
        air_time=20 / 120.0,
        torque=(0.0, 1.0, 0.0),
    )
    double.launch(11, velocity=102.0, has_jumped=1, has_flipped=1, wheel_faces=(0,) * 4)
    double.launch(20, velocity=102.0, has_jumped=1, has_flipped=0, air_time=0.1)
    double.launch(
        21,
        velocity=102.0,
        has_jumped=1,
        has_flipped=1,
        is_flipping=1,
        air_time=0.1,
        torque=(0.0, 1.0, 0.0),
    )
    double.launch(22, velocity=104.0, has_jumped=1, has_flipped=1, wheel_faces=(0,) * 4)
    double_observed = {
        "dash_success_total": double.value("dash_success_total"),
        "double_dash_total": double.value("double_dash_total"),
        "interval_requested": double.value("interval_requested"),
        "successful_dash_detected": double.value("interval_detected", 7),
    }
    cases["two_dashes_pay_twice_double_label_has_no_third_payout"] = {
        "observed": double_observed,
        "pass": double_observed
        == {
            "dash_success_total": 2,
            "double_dash_total": 1,
            "interval_requested": 2,
            "successful_dash_detected": 2,
        },
    }

    _, ball = _reset_case(collision_root, face=-6, support_wheels=3)
    cases["ball_reset_three_support_wheels_and_resource_transition"] = {
        "observed": ball,
        "pass": ball["pending_after_support"] == 1
        and ball["reset_completion_total"] == 1
        and ball["interval_requested"] == 1,
    }
    _, car = _reset_case(collision_root, face=-7, support_wheels=3)
    cases["car_reset_three_other_car_support_wheels"] = {
        "observed": car,
        "pass": car["pending_after_support"] == 2
        and car["reset_completion_total"] == 1,
    }
    _, too_few = _reset_case(collision_root, face=-6, support_wheels=2)
    cases["reset_two_support_wheels_rejects"] = {
        "observed": too_few,
        "pass": too_few["pending_after_support"] == 0
        and too_few["reset_completion_total"] == 0,
    }
    _, unchanged = _reset_case(
        collision_root, face=-6, support_wheels=3, previous_untimed=1
    )
    cases["unchanged_untimed_resource_rejects"] = {
        "observed": unchanged,
        "pass": unchanged["pending_after_support"] == 0
        and unchanged["reset_completion_total"] == 0,
    }

    chain, first = _reset_case(collision_root, face=-6, support_wheels=3)
    chain.launch(12, wheel_faces=(-6, -6, -6))
    chain.launch(13)
    before_consumption = chain.value("reset_completion_total")
    chain.launch(20, has_flipped=1, is_flipping=1, torque=(0.0, 1.0, 0.0))
    chain.launch(21, wheel_faces=(-6, -6, -6))
    chain.launch(22)
    chain_observed = {
        "first_completion": first["reset_completion_total"],
        "before_consumption": before_consumption,
        "after_distinct_reacquisition": chain.value("reset_completion_total"),
        "chain_reset_total": chain.value("chain_reset_total"),
        "interval_requested": chain.value("interval_requested"),
    }
    cases["chain_requires_loss_then_distinct_reacquisition"] = {
        "observed": chain_observed,
        "pass": chain_observed
        == {
            "first_completion": 1,
            "before_consumption": 1,
            "after_distinct_reacquisition": 2,
            "chain_reset_total": 1,
            "interval_requested": 2,
        },
    }

    preflip_harness, preflip = _reset_case(
        collision_root,
        face=-6,
        support_wheels=3,
        previous_has_flipped=1,
    )
    preflip["ball_reset_detected"] = preflip_harness.value("interval_detected", 8)
    cases["preflip_subtype_adds_zero_extra_payout"] = {
        "observed": preflip,
        "pass": preflip["preflip_reset_total"] == 1
        and preflip["interval_requested"] == 1
        and preflip["ball_reset_detected"] == 1,
    }

    return {
        "format": "RIVAL2_GAMEPLAY_V3_DASH_RESET_SOURCE_EXACT_V1",
        "created_utc": _utc_now(),
        "production_kernel": "rivalsim.gameplay_v3.gameplay_v3_track_tick",
        "authoritative_arrays": True,
        "bounded_full_simulator_benchmark": False,
        "cases": cases,
        "case_count": len(cases),
        "passed": sum(bool(case["pass"]) for case in cases.values()),
        "failed": [name for name, case in cases.items() if not case["pass"]],
        "verdict": "PASS" if all(case["pass"] for case in cases.values()) else "FAIL",
    }


def _source_exact_phase(collision_root: str, output_dir: Path) -> None:
    evidence = run_dash_reset_source_exact(collision_root)
    _write_json(output_dir / "dash_reset_source_exact.json", evidence)
    if evidence["verdict"] != "PASS":
        raise RuntimeError(f"source-exact dash/reset failures: {evidence['failed']}")
    print(json.dumps({"passed": evidence["passed"], "failed": evidence["failed"]}))


def _reward_phase(collision_root: str, output_dir: Path) -> None:
    derivation = json.loads(
        (output_dir / "classifier_threshold_derivation.json").read_text(encoding="utf-8")
    )
    thresholds = derivation["thresholds"]
    from rivalsim import gameplay_v3 as gameplay_v3_module

    runtime_thresholds = {
        name: getattr(gameplay_v3_module, name) for name in sorted(thresholds)
    }
    runtime_parity = {
        name: float(runtime_thresholds[name]) == float(value)
        for name, value in thresholds.items()
    }
    env = Rival2Env(
        64,
        collision_root,
        reward_version=RIVAL2_REWARD_GAMEPLAY_V3_VERSION,
    )
    generator = torch.Generator(device=env.device).manual_seed(2026082802)
    maxima = {"blue_reconstruction": 0.0, "orange_zero_sum": 0.0}
    samples = 0
    for _ in range(64):
        action = torch.rand((64, 2, 8), device=env.device, generator=generator) * 2.0 - 1.0
        step = env.step(action)
        blue = sum(
            env.bridge.views[name]
            for name in (
                "rival2.v1_goal_component",
                "rival2.v1_progress_component",
                "rival2.v1_touch_component",
                "rival2.v1_demo_component",
                "rival2.speed_component",
                "rival2.supersonic_component",
                "rival2.boost_use_component",
                "rival2.boost_pickup_component",
                "rival2.save_component",
                "gameplay_v3.mechanics_component",
                "gameplay_v3.bad_flip_component",
            )
        )
        maxima["blue_reconstruction"] = max(
            maxima["blue_reconstruction"],
            float((step.reward[:, 0] - blue).abs().max().item()),
        )
        maxima["orange_zero_sum"] = max(
            maxima["orange_zero_sum"],
            float((step.reward[:, 1] + step.reward[:, 0]).abs().max().item()),
        )
        samples += 64
    torch.cuda.synchronize(env.device)
    reward = {
        "schema_version": 1,
        "created_utc": _utc_now(),
        "decisions": samples,
        "max_abs_error": maxima,
        "touch_component_exact_zero": bool(
            (env.bridge.views["rival2.v1_touch_component"] == 0).all()
        ),
        "runtime_threshold_parity": runtime_parity,
        "frozen_thresholds_sha256": derivation["frozen_thresholds_sha256"],
        "tolerance": 1.0e-6,
    }
    reward["verdict"] = (
        "PASS"
        if max(maxima.values()) <= 1.0e-6
        and reward["touch_component_exact_zero"]
        and all(runtime_parity.values())
        else "BLOCKED"
    )
    _write_json(output_dir / "reward_reconstruction.json", reward)
    contract = {
        "schema_version": 1,
        "created_utc": _utc_now(),
        "version": RIVAL2_REWARD_GAMEPLAY_V3_VERSION,
        "sha256": REWARD_GAMEPLAY_V3_CONTRACT_HASH,
        "contract": REWARD_GAMEPLAY_V3_CONTRACT,
        "contract_hashes": _provenance()["contract_hashes"],
        "runtime_threshold_parity": runtime_parity,
        "verdict": reward["verdict"],
    }
    _write_json(output_dir / "contract.json", contract)
    if reward["verdict"] != "PASS":
        raise RuntimeError("reward reconstruction or runtime threshold parity failed")
    print(json.dumps({"reward": reward["verdict"], "contract": contract["sha256"]}))


def _probe(collision_root: str) -> None:
    provenance = _provenance()
    rows = _generate_split("derivation", collision_root, provenance)
    for classifier in CLASSIFIERS:
        print(classifier)
        selected = [row for row in rows if row["classifier"] == classifier]
        for row in selected:
            features = row["features"]
            if classifier == "contest":
                compact = {
                    key: round(float(features[key]), 3)
                    for key in (
                        "opponent_ball_distance",
                        "self_closing_speed",
                        "opponent_closing_speed",
                        "time_to_ball_delta",
                        "adjacent_contact_separation_ticks",
                        "adjacent_ball_displacement",
                        "active_directional_dodge",
                    )
                }
            else:
                compact = {
                    key: round(float(features[key]), 3)
                    for key in (
                        "total_closing_speed",
                        "rotational_closing_contribution",
                        "rotational_share",
                        "ball_delta_v",
                        "precontact_observed_ticks",
                        "precontact_max_distance",
                        "precontact_max_relative_speed",
                        "release_distance",
                        "active_directional_dodge",
                    )
                }
            print(row["scenario_id"], row["scenario"]["kind"], compact)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collision-root",
        default=os.environ.get(
            "RIVALSIM_COLLISION_DIR", r"G:\dev\RLBot-Rival\bot\collision_meshes"
        ),
    )
    parser.add_argument("--probe", action="store_true")
    parser.add_argument(
        "--phase", choices=("derive", "heldout", "source-exact", "reward")
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if args.probe:
        _probe(args.collision_root)
        return 0
    if args.phase == "derive":
        _derive_phase(args.collision_root, args.output_dir)
        return 0
    if args.phase == "heldout":
        _heldout_phase(args.collision_root, args.output_dir)
        return 0
    if args.phase == "source-exact":
        _source_exact_phase(args.collision_root, args.output_dir)
        return 0
    if args.phase == "reward":
        _reward_phase(args.collision_root, args.output_dir)
        return 0
    raise RuntimeError("choose --phase derive or --phase heldout")


if __name__ == "__main__":
    raise SystemExit(main())
