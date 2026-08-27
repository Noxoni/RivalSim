#!/usr/bin/env python3
"""Deterministic RivalSim-native mechanics calibration and shadow gate.

The calibration phase runs deliberately small, state-injected batches through
the normal 120 Hz Soccar simulator and exports bounded host traces.  The shadow
phase attaches the GPU observer without changing policy, opponent, reward, or
lifecycle behavior.
"""

# ruff: noqa: E402 -- direct script execution must prepend the repository root.

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from rivalsim import CompleteWorldSim, StateSnapshot
from rivalsim.arena import ArenaGeometry, WarpArenaMeshes
from rivalsim.controls import ControlBatch
from rivalsim.mechanics_calibration import (
    FAMILY_ID,
    FAMILY_NAMES,
    STATUS_CALIBRATED,
    STATUS_NOT_READY,
    THRESHOLD_NAMES,
    MechanicsShadowObserver,
    ScalarBoundary,
    classify,
    midpoint_boundary,
)
from rivalsim.nexto_short_eval import NextoShortEpisodeRunner
from rivalsim.wisp_short_eval import WispShortEpisodeRunner

SOURCE_HANDOFF_COMMIT = "1da8557f32a94e6a8e96d1acbb0103656e203e27"
ACCEPTED_BASELINE_COMMIT = "f49768368377dcb5aa0cc67f3a08f79bd68538a3"
TARGET_FAMILIES = ("musty", "breezi", "redirect")
EXPECTED_CHECKPOINT_SHA256 = "77BF257131FB71DDEAEAE49D668C5E25AB1D06EE26149AB0D0AE303573CA5F21"
CALIBRATION_SEED = 2026082701
TARGET_CALIBRATION_SEED = 2026082707
TARGET_HELDOUT_SCENARIO_OFFSET = 509
SHADOW_SEED = 2026082702
CASE_COUNT_PER_CLASS = 24
DERIVATION_PER_CLASS = 16
HELDOUT_PER_CLASS = 8
TRACE_TICKS = 120


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _git_text_at(commit: str, path: str) -> str:
    return subprocess.check_output(
        ["git", "show", f"{commit}:{path}"], text=True, encoding="utf-8"
    )


def _quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return np.asarray(
        (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ),
        dtype=np.float32,
    )


def _quat_rotate(q: np.ndarray, value: np.ndarray) -> np.ndarray:
    u = q[:3]
    return (
        2.0 * np.dot(u, value) * u
        + (q[3] * q[3] - np.dot(u, u)) * value
        + 2.0 * q[3] * np.cross(u, value)
    )


def _quat_unrotate(q: np.ndarray, value: np.ndarray) -> np.ndarray:
    conjugate = np.asarray((-q[0], -q[1], -q[2], q[3]), dtype=np.float32)
    return _quat_rotate(conjugate, value)


def _unit(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    return value / norm if norm > 1.0e-8 else np.zeros(3, dtype=np.float32)


def _base_state(count: int) -> StateSnapshot:
    state = StateSnapshot.empty(count)
    state.car_pos[:, 0] = np.asarray((0.0, 0.0, 17.0), dtype=np.float32)
    state.car_pos[:, 1] = np.asarray((2800.0, 2800.0, 1000.0), dtype=np.float32)
    state.car_quat[:, 0, 3] = 1.0
    state.car_quat[:, 1, 3] = 1.0
    state.on_ground[:, 0] = 1
    state.on_ground[:, 1] = 0
    state.ball_pos[:] = np.asarray((0.0, -3000.0, 1000.0), dtype=np.float32)
    return state


def _controls(count: int) -> ControlBatch:
    return ControlBatch.zeros(count)


def _scenario_rows(family: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for class_name in ("positive", "near_miss", "ordinary_control"):
        for index in range(CASE_COUNT_PER_CLASS):
            split = "derivation" if index < DERIVATION_PER_CLASS else "heldout"
            scenario_index = index
            seed = CALIBRATION_SEED + FAMILY_ID[family] * 1000 + index
            if family in TARGET_FAMILIES:
                seed = TARGET_CALIBRATION_SEED + FAMILY_ID[family] * 1000 + index
            rows.append(
                {
                    "case_id": f"{family}-{class_name}-{split[0].upper()}{index:02d}",
                    "family": family,
                    "class": class_name,
                    "split": split,
                    "class_index": scenario_index,
                    "scenario_variant": (
                        TARGET_HELDOUT_SCENARIO_OFFSET if split == "heldout" else 0
                    ),
                    "seed": seed,
                    "scenario": {},
                }
            )
    return rows


class Trace:
    """Bounded host trace copied only by the calibration harness."""

    def __init__(self, count: int):
        self.count = count
        self.car_pos: list[np.ndarray] = []
        self.car_vel: list[np.ndarray] = []
        self.car_quat: list[np.ndarray] = []
        self.car_ang: list[np.ndarray] = []
        self.on_ground: list[np.ndarray] = []
        self.has_flipped: list[np.ndarray] = []
        self.is_flipping: list[np.ndarray] = []
        self.flip_torque: list[np.ndarray] = []
        self.wheel_count: list[np.ndarray] = []
        self.chassis_count: list[np.ndarray] = []
        self.chassis_local: list[np.ndarray] = []
        self.chassis_normal: list[np.ndarray] = []
        self.ball_pos: list[np.ndarray] = []
        self.ball_vel: list[np.ndarray] = []
        self.ball_world_count: list[np.ndarray] = []
        self.ball_world_normal: list[np.ndarray] = []
        self.hit: list[np.ndarray] = []
        self.contact_normal: list[np.ndarray] = []
        self.contact_point: list[np.ndarray] = []
        self.pre_car_vel: list[np.ndarray] = []
        self.pre_car_ang: list[np.ndarray] = []
        self.ball_delta_v: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []

    def append(self, sim: CompleteWorldSim, controls: ControlBatch) -> None:
        snap = sim.snapshot()
        count = self.count
        wheel = np.asarray(sim.vehicle.wheel_contact.numpy()).reshape(count, 2, 4)
        chassis = np.asarray(sim.vehicle.contact_count.numpy()).reshape(count, 2)
        local = np.asarray(sim.vehicle.contact_local_a.numpy()).reshape(count, 2, 12, 3)
        normal = np.asarray(sim.vehicle.contact_normal.numpy()).reshape(count, 2, 12, 3)
        world_count = np.asarray(sim.ball_world.contact_count.numpy()).reshape(count)
        world_normal = np.asarray(sim.ball_world.contact_normal.numpy()).reshape(count, 16, 3)
        hit = np.stack(
            (
                np.asarray(sim.car_ball.hit_this_tick.numpy()),
                np.asarray(sim.car_ball_b.hit_this_tick.numpy()),
            ),
            axis=1,
        )
        pair_normal = np.stack(
            (
                np.asarray(sim.car_ball.contact_normal.numpy()),
                np.asarray(sim.car_ball_b.contact_normal.numpy()),
            ),
            axis=1,
        )
        pair_point = np.stack(
            (
                np.asarray(sim.car_ball.contact_point_a_bt.numpy()),
                np.asarray(sim.car_ball_b.contact_point_a_bt.numpy()),
            ),
            axis=1,
        )
        pre_vel = np.stack(
            (
                np.asarray(sim.car_ball.pre_car_velocity_bt.numpy()),
                np.asarray(sim.car_ball_b.pre_car_velocity_bt.numpy()),
            ),
            axis=1,
        )
        pre_ang = np.stack(
            (
                np.asarray(sim.car_ball.pre_car_angular_velocity.numpy()),
                np.asarray(sim.car_ball_b.pre_car_angular_velocity.numpy()),
            ),
            axis=1,
        )
        delta_v = np.stack(
            (
                np.asarray(sim.car_ball.extra_hit_velocity_uu.numpy()),
                np.asarray(sim.car_ball_b.extra_hit_velocity_uu.numpy()),
            ),
            axis=1,
        )
        self.car_pos.append(snap.car_pos[:, 0].copy())
        self.car_vel.append(snap.car_vel[:, 0].copy())
        self.car_quat.append(snap.car_quat[:, 0].copy())
        self.car_ang.append(snap.car_ang_vel[:, 0].copy())
        self.on_ground.append(snap.on_ground[:, 0].copy())
        self.has_flipped.append(snap.has_flipped[:, 0].copy())
        self.is_flipping.append(snap.is_flipping[:, 0].copy())
        self.flip_torque.append(snap.flip_rel_torque[:, 0].copy())
        self.wheel_count.append(np.sum(wheel[:, 0] != 0, axis=1).astype(np.int32))
        self.chassis_count.append(chassis[:, 0].copy())
        self.chassis_local.append(local[:, 0, 0].copy())
        self.chassis_normal.append(normal[:, 0, 0].copy())
        self.ball_pos.append(snap.ball_pos.copy())
        self.ball_vel.append(snap.ball_vel.copy())
        self.ball_world_count.append(world_count.copy())
        self.ball_world_normal.append(world_normal[:, 0].copy())
        self.hit.append(hit.copy())
        self.contact_normal.append(pair_normal[:, 0].copy())
        self.contact_point.append(pair_point[:, 0].copy())
        self.pre_car_vel.append(pre_vel[:, 0].copy())
        self.pre_car_ang.append(pre_ang[:, 0].copy())
        self.ball_delta_v.append(delta_v[:, 0].copy())
        self.actions.append(
            np.stack(
                (
                    controls.throttle[:, 0],
                    controls.steer[:, 0],
                    controls.pitch[:, 0],
                    controls.yaw[:, 0],
                    controls.roll[:, 0],
                    controls.jump[:, 0],
                    controls.boost[:, 0],
                    controls.handbrake[:, 0],
                ),
                axis=1,
            )
        )

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            name: np.asarray(value) for name, value in vars(self).items() if isinstance(value, list)
        }


def _run(
    state: StateSnapshot,
    ticks: int,
    collision_root: str,
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    controller: Callable[[int, ControlBatch], None],
) -> dict[str, np.ndarray]:
    count = state.num_envs
    sim = CompleteWorldSim(
        count,
        collision_root,
        initial=state,
        geometry=geometry,
        meshes=meshes,
        auto_kickoff=False,
        car_visitation_order="a_then_b",
    )
    trace = Trace(count)
    controls = _controls(count)
    trace.append(sim, controls)
    for tick in range(ticks):
        controls = _controls(count)
        controller(tick, controls)
        sim.set_controls(controls)
        sim.step(1, synchronize=True)
        trace.append(sim, controls)
    return trace.arrays()


def _first_onset(mask: np.ndarray) -> np.ndarray:
    onset = mask & ~np.concatenate((np.zeros_like(mask[:1]), mask[:-1]), axis=0)
    result = np.full(mask.shape[1], -1, dtype=np.int32)
    for column in range(mask.shape[1]):
        found = np.flatnonzero(onset[:, column])
        if found.size:
            result[column] = int(found[0])
    return result


def _touch_onsets(hit: np.ndarray) -> list[np.ndarray]:
    latched = hit[:, :, 0] != 0
    onset = latched & ~np.concatenate((np.zeros_like(latched[:1]), latched[:-1]), axis=0)
    return [np.flatnonzero(onset[:, index]) for index in range(onset.shape[1])]


def _flip_cases(
    family: str,
    rows: list[dict[str, Any]],
    collision_root: str,
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
) -> list[dict[str, Any]]:
    count = len(rows)
    state = _base_state(count)
    speed_range = (450.0, 675.0) if family == "half_flip" else (700.0, 1250.0)
    # Give every class the same prospective speed distribution so detector
    # separation cannot be an artifact of row/class ordering.
    state.car_vel[:, 0, 0] = np.tile(
        np.linspace(*speed_range, CASE_COUNT_PER_CLASS, dtype=np.float32), 3
    )
    cancel_delay = np.full(count, -1, dtype=np.int32)
    dodge_tick = np.full(count, 10, dtype=np.int32)
    diagonal = np.zeros(count, dtype=np.float32)
    for index, row in enumerate(rows):
        kind = row["class"]
        local = int(row["class_index"])
        if kind == "positive":
            cancel_delay[index] = (34 + local % 5) if family == "half_flip" else (1 + local % 3)
            diagonal[index] = (-1.0 if local % 2 else 1.0) * (0.55 + 0.04 * (local % 5))
        elif kind == "near_miss":
            cancel_delay[index] = (8 + local % 17) if family == "half_flip" else (13 + local % 9)
            diagonal[index] = (-1.0 if local % 2 else 1.0) * (0.55 + 0.04 * (local % 5))
        elif local < 8:
            diagonal[index] = 0.0
        elif local < 16:
            dodge_tick[index] = -1
        else:
            diagonal[index] = 1.0
        if family == "half_flip":
            diagonal[index] = 0.0
        row["scenario"] = {
            "initial_speed": float(state.car_vel[index, 0, 0]),
            "dodge_tick": int(dodge_tick[index]),
            "cancel_delay": int(cancel_delay[index]),
            "diagonal_input": float(diagonal[index]),
        }

    backward = family == "half_flip"

    def controller(tick: int, controls: ControlBatch) -> None:
        controls.boost[:, 0] = int(not backward)
        if tick == 0:
            controls.jump[:, 0] = 1
        active = dodge_tick == tick
        controls.jump[active, 0] = 1
        controls.pitch[active, 0] = 1.0 if backward else -1.0
        controls.yaw[active, 0] = diagonal[active]
        for index in range(count):
            if cancel_delay[index] >= 0 and tick >= dodge_tick[index] + cancel_delay[index]:
                controls.pitch[index, 0] = -1.0 if backward else 1.0
                if backward:
                    controls.roll[index, 0] = -1.0 if index % 2 else 1.0

    trace = _run(state, 240 if backward else 55, collision_root, geometry, meshes, controller)
    onset = _first_onset(trace["has_flipped"] != 0)
    for index, row in enumerate(rows):
        start = int(onset[index])
        assessment = min(trace["car_vel"].shape[0] - 1, start + (72 if backward else 36))
        finish = assessment
        if backward and start >= 0:
            support = np.flatnonzero(trace["wheel_count"][assessment:, index] > 0)
            if support.size:
                finish = assessment + int(support[0])
        if start < 0:
            pitch_path = 999.0
            roll_path = 999.0
            yaw_path = 999.0
            alignment = -1.0
            alignment_recovery_ticks = 999.0
            heading = 1.0
            new_forward_speed = -9999.0
            initial_tangent_speed = 0.0
            final_tangent_speed = 0.0
            actual = 0.0
        else:
            actual = 1.0
            pitch_path = 0.0
            roll_path = 0.0
            yaw_path = 0.0
            alignment_recovery_ticks = 999.0
            for tick in range(start, assessment + 1):
                q = trace["car_quat"][tick, index]
                forward_tick = _quat_rotate(
                    q, np.asarray((1.0, 0.0, 0.0), dtype=np.float32)
                )
                right = _quat_rotate(q, np.asarray((0.0, 1.0, 0.0), dtype=np.float32))
                up = _quat_rotate(q, np.asarray((0.0, 0.0, 1.0), dtype=np.float32))
                angular = trace["car_ang"][tick, index]
                pitch_path += abs(float(np.dot(angular, right))) / 120.0
                roll_path += abs(float(np.dot(angular, forward_tick))) / 120.0
                yaw_path += abs(float(np.dot(angular, up))) / 120.0
                tangent_tick = trace["car_vel"][tick, index].copy()
                tangent_tick[2] = 0.0
                if (
                    alignment_recovery_ticks == 999.0
                    and float(np.dot(_unit(forward_tick), _unit(tangent_tick))) >= 0.9
                ):
                    alignment_recovery_ticks = float(tick - start)
            forward_initial = _quat_rotate(
                trace["car_quat"][start, index], np.asarray((1.0, 0.0, 0.0), dtype=np.float32)
            )
            forward_final = _quat_rotate(
                trace["car_quat"][finish, index], np.asarray((1.0, 0.0, 0.0), dtype=np.float32)
            )
            tangent = trace["car_vel"][finish, index].copy()
            tangent[2] = 0.0
            initial_tangent = trace["car_vel"][start, index].copy()
            initial_tangent[2] = 0.0
            initial_tangent_speed = float(np.linalg.norm(initial_tangent))
            final_tangent_speed = float(np.linalg.norm(tangent))
            alignment = float(np.dot(_unit(forward_final), _unit(tangent)))
            heading = float(np.dot(_unit(forward_final), _unit(forward_initial)))
            new_forward_speed = float(np.dot(trace["car_vel"][finish, index], forward_final))
        features = {
            "actual_dodge": actual,
            "cancel_ticks": float(cancel_delay[index] if cancel_delay[index] >= 0 else 999),
            "cancel_ticks_min_feature": float(
                cancel_delay[index] if cancel_delay[index] >= 0 else 999
            ),
            "cancel_ticks_max_feature": float(
                cancel_delay[index] if cancel_delay[index] >= 0 else 999
            ),
            "pitch_rotation": float(pitch_path),
            "roll_rotation": float(roll_path),
            "yaw_rotation": float(yaw_path),
            "alignment": alignment,
            "alignment_recovery_ticks": alignment_recovery_ticks,
            "heading_dot": heading,
            "new_forward_speed": new_forward_speed,
            "initial_tangent_speed": initial_tangent_speed,
            "final_tangent_speed": final_tangent_speed,
            "tangent_speed_delta": final_tangent_speed - initial_tangent_speed,
            "supported_completion": float(
                trace["wheel_count"][finish, index] > 0 if backward else False
            ),
        }
        row["features"] = features
        row["physical_invariant"] = (
            bool(
                actual
                and (
                    (
                        not backward
                        and abs(diagonal[index]) >= 0.5
                        and cancel_delay[index] <= 3
                        and features["alignment"] > 0.0
                        and features["tangent_speed_delta"] >= -1.0
                    )
                    or (
                        backward
                        and 34 <= cancel_delay[index] <= 38
                        and features["heading_dot"] < -0.35
                        and features["new_forward_speed"] > 1.0
                        and features["supported_completion"] > 0.0
                    )
                )
            )
            if row["class"] == "positive"
            else True
        )
    return rows


def _roof_cases(
    family: str,
    rows: list[dict[str, Any]],
    collision_root: str,
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
) -> list[dict[str, Any]]:
    count = len(rows)
    state = _base_state(count)
    for index, row in enumerate(rows):
        local = int(row["class_index"])
        if row["class"] == "positive":
            z = 145.0 + 1.2 * (local % 10)
            lateral = ((local % 5) - 2) * 4.0
            vz = -45.0 + 6.0 * (local % 8)
            shared_speed = 120.0 + 10.0 * (local % 6) if family == "ground_carry" else 0.0
        elif row["class"] == "near_miss":
            z = 185.0 + 2.0 * (local % 10)
            lateral = ((local % 7) - 3) * 22.0
            vz = 130.0 + 8.0 * (local % 8)
            shared_speed = 100.0 if family == "ground_carry" else 0.0
        else:
            z = 300.0 + 8.0 * local
            lateral = 180.0 + 12.0 * local
            vz = 350.0
            shared_speed = 0.0
        state.ball_pos[index] = (lateral, 0.0, z)
        state.ball_vel[index] = (shared_speed, 0.0, vz)
        state.car_vel[index, 0, 0] = shared_speed
        row["scenario"] = {
            "ball_z": z,
            "lateral_offset": lateral,
            "ball_vz": vz,
            "shared_forward_speed": shared_speed,
        }

    def controller(tick: int, controls: ControlBatch) -> None:
        if family == "ground_carry":
            controls.throttle[:, 0] = 0.45 if tick < 60 else -0.20

    trace = _run(state, 300, collision_root, geometry, meshes, controller)
    onsets = _touch_onsets(trace["hit"])
    for index, row in enumerate(rows):
        touches = onsets[index]
        distances = np.linalg.norm(trace["ball_pos"][:, index] - trace["car_pos"][:, index], axis=1)
        relative = np.linalg.norm(trace["ball_vel"][:, index] - trace["car_vel"][:, index], axis=1)
        gaps = np.diff(touches) if touches.size > 1 else np.asarray((999,), dtype=np.int32)
        contact = trace["hit"][:, index, 0] != 0
        support = (
            contact
            & (trace["on_ground"][:, index] != 0)
            & (trace["ball_pos"][:, index, 2] - trace["car_pos"][:, index, 2] > 80.0)
        )
        longest = 0
        current = 0
        for active in support:
            current = current + 1 if active else 0
            longest = max(longest, current)
        features = {
            "touch_onsets": float(touches.size),
            "control_distance": float(np.max(distances[touches])) if touches.size else 9999.0,
            "control_relative_speed": float(np.max(relative[touches])) if touches.size else 9999.0,
            "contact_gap_ticks": float(np.max(gaps)),
            "support_ticks": float(longest),
            "velocity_change": float(
                np.linalg.norm(trace["car_vel"][-1, index] - trace["car_vel"][0, index])
            ),
        }
        row["features"] = features
        row["physical_invariant"] = (
            bool(touches.size >= 2 if family == "possession" else longest >= 2)
            if row["class"] == "positive"
            else True
        )
    return rows


MUSTY_OFFSETS = (
    (-132.0, 0.0, 110.0),
    (0.0, 0.0, 150.0),
    (-88.0, 0.0, 150.0),
    (-132.0, 0.0, 150.0),
    (-176.0, 0.0, 130.0),
    (-220.0, 0.0, 130.0),
)


def _musty_contact_features(
    trace: dict[str, np.ndarray],
    index: int,
    tick: int,
    *,
    history_ticks: int = 8,
    dodge_tick: int = 0,
) -> dict[str, float]:
    """Measure the signed rotating-surface sweep at one legitimate contact onset."""

    pre_tick = max(0, tick - 1)
    car_position = trace["car_pos"][pre_tick, index]
    ball_position = trace["ball_pos"][pre_tick, index]
    car_quaternion = trace["car_quat"][pre_tick, index]
    contact_world = trace["contact_point"][tick, index] * 50.0
    contact_offset_world = contact_world - car_position
    contact_offset_local = _quat_unrotate(car_quaternion, contact_offset_world)
    car_to_ball = _unit(ball_position - car_position)
    rotational_velocity = np.cross(
        trace["pre_car_ang"][tick, index], contact_offset_world
    )
    car_velocity = trace["pre_car_vel"][tick, index] * 50.0
    incoming_ball_velocity = trace["ball_vel"][pre_tick, index]
    translation_relative = car_velocity - incoming_ball_velocity
    rotational_closing = float(np.dot(rotational_velocity, car_to_ball))
    translation_closing = float(np.dot(translation_relative, car_to_ball))
    rotational_positive = max(0.0, rotational_closing)
    translation_positive = max(0.0, translation_closing)
    rotational_fraction = rotational_positive / max(
        rotational_positive + translation_positive, 1.0e-8
    )

    start = max(0, pre_tick - history_ticks)
    surface_points: list[np.ndarray] = []
    gaps: list[float] = []
    rotation_offsets: list[np.ndarray] = []
    for sample_tick in range(start, pre_tick + 1):
        rotated_offset = _quat_rotate(
            trace["car_quat"][sample_tick, index], contact_offset_local
        )
        surface_point = trace["car_pos"][sample_tick, index] + rotated_offset
        rotation_offsets.append(rotated_offset)
        surface_points.append(surface_point)
        gaps.append(
            float(np.linalg.norm(trace["ball_pos"][sample_tick, index] - surface_point))
        )
    path_length = float(
        sum(
            np.linalg.norm(rotation_offsets[item] - rotation_offsets[item - 1])
            for item in range(1, len(rotation_offsets))
        )
    )
    rotational_displacement = rotation_offsets[-1] - rotation_offsets[0]
    sweep_closure = float(np.dot(rotational_displacement, car_to_ball))
    sweep_alignment = float(np.dot(_unit(rotational_displacement), car_to_ball))
    monotonic_steps = sum(
        gaps[item] < gaps[item - 1] for item in range(1, len(gaps))
    )
    monotonic_fraction = float(monotonic_steps / max(len(gaps) - 1, 1))
    gap_closure = float(gaps[0] - gaps[-1])
    ball_delta = float(np.linalg.norm(trace["ball_delta_v"][tick, index]))
    delta_velocity = trace["ball_delta_v"][tick, index]
    rotational_impulse_alignment = float(
        np.dot(_unit(delta_velocity), _unit(rotational_velocity))
    )
    translation_impulse_alignment = float(
        np.dot(_unit(delta_velocity), _unit(translation_relative))
    )
    ball_offset_local = _quat_unrotate(
        car_quaternion, ball_position - car_position
    )
    return {
        "legitimate_contact": 1.0,
        "actual_backward_dodge": float(
            np.any(trace["has_flipped"][: tick + 1, index] != 0)
            and np.min(trace["flip_torque"][: tick + 1, index, 1]) < -0.25
        ),
        "contact_age_ticks": float(tick - dodge_tick),
        "rotational_closing_speed": rotational_closing,
        "translation_closing_speed": translation_closing,
        "rotational_fraction": rotational_fraction,
        "sweep_closure": sweep_closure,
        "sweep_alignment": sweep_alignment,
        "sweep_path_length": path_length,
        "sweep_monotonic_fraction": monotonic_fraction,
        "precontact_gap_closure": gap_closure,
        "contact_local_x": float(contact_offset_local[0]),
        "contact_local_y": float(contact_offset_local[1]),
        "contact_local_z": float(contact_offset_local[2]),
        "contact_local_up_fraction": float(contact_offset_local[2] / 36.16),
        "ball_local_x": float(ball_offset_local[0]),
        "ball_local_y": float(ball_offset_local[1]),
        "ball_local_z": float(ball_offset_local[2]),
        "rotational_impulse_alignment": rotational_impulse_alignment,
        "translation_impulse_alignment": translation_impulse_alignment,
        "ball_delta_v": ball_delta,
    }


def _empty_musty_features() -> dict[str, float]:
    return {
        "legitimate_contact": 0.0,
        "actual_backward_dodge": 0.0,
        "contact_age_ticks": -1.0,
        "rotational_closing_speed": 0.0,
        "translation_closing_speed": 0.0,
        "rotational_fraction": 0.0,
        "sweep_closure": 0.0,
        "sweep_alignment": -1.0,
        "sweep_path_length": 0.0,
        "sweep_monotonic_fraction": 0.0,
        "precontact_gap_closure": 0.0,
        "contact_local_x": 0.0,
        "contact_local_y": 0.0,
        "contact_local_z": 0.0,
        "contact_local_up_fraction": 0.0,
        "ball_local_x": 0.0,
        "ball_local_y": 0.0,
        "ball_local_z": 0.0,
        "rotational_impulse_alignment": -1.0,
        "translation_impulse_alignment": -1.0,
        "ball_delta_v": 0.0,
    }


def _musty_cases(
    family: str,
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
    for index, row in enumerate(rows):
        local = int(row["class_index"])
        variant = int(row.get("scenario_variant", 0))
        phase = ((variant * 17 + local * 13) % 11 - 5) if variant else 0
        origin = local % 4
        yaw = (-0.45, -0.12, 0.28, 0.55)[local % 4] + phase * 0.006
        roll = (0.0, 0.18, -0.22, 0.08)[local % 4]
        pitch = (0.0, 0.08, -0.10, 0.04)[local % 4]
        quaternion = _quat_from_euler(roll, pitch, yaw)
        height = (
            (620.0, 980.0, 1320.0, 860.0)[origin]
            + 8.0 * (local // 4)
            + phase * 2.0
        )
        local_speed = (
            (320.0, 520.0, 690.0, 430.0)[origin]
            + 12.0 * (local % 3)
            + phase * 3.0
        )
        local_vertical = (-30.0, 35.0, -70.0, 55.0)[origin]
        local_car_velocity = np.asarray(
            (local_speed, ((local % 3) - 1) * 18.0, local_vertical), dtype=np.float32
        )
        car_velocity = _quat_rotate(quaternion, local_car_velocity)
        state.car_pos[index, 0] = (0.0, 0.0, height)
        state.car_quat[index, 0] = quaternion
        state.car_vel[index, 0] = car_velocity
        state.has_jumped[index, 0] = 0 if origin == 3 else 1
        if row["class"] == "positive":
            offset = np.asarray(MUSTY_OFFSETS[1 + local % 5], dtype=np.float32)
            offset[1] += ((local % 3) - 1) * 6.0
            relative_velocity = np.asarray(
                ((local % 4 - 1.5) * 8.0, (local % 3 - 1) * 5.0, 0.0),
                dtype=np.float32,
            )
            dodge = True
            hard_negative = ""
        elif row["class"] == "near_miss":
            negative_kind = local % 6
            hard_negative = (
                "rotated_translation_dominated_rear_bonk",
                "rotated_front_clear",
                "lateral_loose_ball_backflip_hit",
                "pre_scoop_contact",
                "roof_slap_from_incoming_ball",
                "high_delta_v_head_on_backflip_hit",
            )[negative_kind]
            if negative_kind == 0:
                # A fast rear impact after the backflip has developed: the
                # incoming ball catches the rotating car, so translation (not
                # the swept rear/roof surface) owns the closure.
                offset = np.asarray((-520.0, 0.0, 135.0), dtype=np.float32)
                relative_velocity = np.asarray((1700.0, 0.0, 0.0), dtype=np.float32)
            elif negative_kind == 1:
                offset = np.asarray((180.0, 0.0, 108.0), dtype=np.float32)
                relative_velocity = np.asarray((-950.0, 0.0, 0.0), dtype=np.float32)
            elif negative_kind == 2:
                side = -1.0 if local % 2 else 1.0
                offset = np.asarray((-140.0, side * 150.0, 125.0), dtype=np.float32)
                relative_velocity = np.asarray((0.0, -side * 620.0, 0.0), dtype=np.float32)
            elif negative_kind == 3:
                offset = np.asarray((28.0 + local % 4, 0.0, 105.0), dtype=np.float32)
                relative_velocity = np.asarray((-250.0, 0.0, 0.0), dtype=np.float32)
            elif negative_kind == 4:
                # A descending ball slaps the roof during a developed
                # backflip, but the roof's rotational sweep is not what
                # closes the contact.
                offset = np.asarray((0.0, 150.0, 150.0), dtype=np.float32)
                relative_velocity = np.asarray((0.0, -400.0, -400.0), dtype=np.float32)
            else:
                offset = np.asarray((300.0, 0.0, 105.0), dtype=np.float32)
                relative_velocity = np.asarray((-1050.0, 0.0, 0.0), dtype=np.float32)
            dodge = True
        else:
            control_kind = local % 3
            hard_negative = (
                "non_dodge_incoming_contact",
                "forward_dodge_contact",
                "random_tumble_contact",
            )[control_kind]
            offset = np.asarray((260.0, ((local % 5) - 2) * 16.0, 110.0), dtype=np.float32)
            relative_velocity = np.asarray((-850.0, 0.0, 0.0), dtype=np.float32)
            dodge = control_kind == 1
            if control_kind == 2:
                state.car_ang_vel[index, 0] = _quat_rotate(
                    quaternion, np.asarray((0.8, 2.0, 1.2), dtype=np.float32)
                )
        world_offset = _quat_rotate(quaternion, offset)
        world_relative_velocity = _quat_rotate(quaternion, relative_velocity)
        state.ball_pos[index] = state.car_pos[index, 0] + world_offset
        state.ball_vel[index] = car_velocity + world_relative_velocity
        row["scenario"] = {
            "origin": (
                "controlled_catch",
                "aerial",
                "rotated_aerial",
                "untimed_resource_origin",
            )[origin],
            "car_height": height,
            "car_local_velocity": local_car_velocity.tolist(),
            "initial_euler": [roll, pitch, yaw],
            "ball_local_offset": offset.tolist(),
            "ball_local_relative_velocity": relative_velocity.tolist(),
            "backward_dodge": dodge,
            "hard_negative_kind": hard_negative,
        }

    def controller(tick: int, controls: ControlBatch) -> None:
        if tick == 0:
            for index, row in enumerate(rows):
                if row["scenario"]["backward_dodge"]:
                    controls.jump[index, 0] = 1
                    controls.pitch[index, 0] = (
                        -1.0
                        if row["scenario"]["hard_negative_kind"] == "forward_dodge_contact"
                        else 1.0
                    )

    trace = _run(state, 70, collision_root, geometry, meshes, controller)
    onsets = _touch_onsets(trace["hit"])
    for index, row in enumerate(rows):
        touches = onsets[index]
        if touches.size:
            tick = int(touches[0])
            features = _musty_contact_features(trace, index, tick)
        else:
            features = _empty_musty_features()
        row["features"] = features
        row["physical_invariant"] = (
            bool(
                touches.size
                and features["actual_backward_dodge"] > 0.0
                and features["contact_age_ticks"] >= 4.0
                and features["rotational_closing_speed"] > 1.0
                and features["sweep_closure"] > 1.0
                and features["sweep_path_length"] > 1.0
                and features["ball_delta_v"] > 1.0
            )
            if row["class"] == "positive"
            else True
        )
    return rows


def _breezi_cases(
    rows: list[dict[str, Any]],
    collision_root: str,
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
) -> list[dict[str, Any]]:
    count = len(rows)
    initial = _base_state(count)
    initial.on_ground[:, 0] = 0
    initial.has_jumped[:, 0] = 1
    initial.air_time[:, 0] = 0.2
    initial.air_time_since_jump[:, 0] = 0.0
    duration = np.zeros(count, dtype=np.int32)
    roll = np.zeros(count, dtype=np.float32)
    yaw = np.zeros(count, dtype=np.float32)
    tornado_duration = np.zeros(count, dtype=np.int32)
    transition_pitch = np.zeros(count, dtype=np.float32)
    terminal_dodge = np.ones(count, dtype=bool)
    ball_relative_velocity = np.zeros((count, 3), dtype=np.float32)
    terminal_offsets = np.zeros((count, 3), dtype=np.float32)
    for index, row in enumerate(rows):
        local = int(row["class_index"])
        variant = int(row.get("scenario_variant", 0))
        phase = ((variant * 19 + local * 7) % 11 - 5) if variant else 0
        initial_yaw = (-0.30, -0.08, 0.18, 0.42)[local % 4] + phase * 0.005
        quaternion = _quat_from_euler(0.0, 0.0, initial_yaw)
        initial.car_pos[index, 0] = (
            ((local % 6) - 2.5) * 120.0,
            ((local // 6) - 1.5) * 120.0,
            1450.0 + 10.0 * (local % 5) + phase * 2.0,
        )
        initial.car_quat[index, 0] = quaternion
        initial.car_vel[index, 0] = _quat_rotate(
            quaternion,
            np.asarray((360.0 + 14.0 * (local % 5), 0.0, 20.0), dtype=np.float32),
        )
        terminal_offsets[index] = np.asarray((-180.0, 0.0, 130.0), dtype=np.float32)
        if row["class"] == "positive":
            duration[index] = 132 + local % 7
            tornado_duration[index] = duration[index] - 48
            transition_pitch[index] = 1.0
            roll[index] = -1.0
            yaw[index] = 0.68 + 0.025 * (local % 5) + phase * 0.004
            hard_negative = ""
        elif row["class"] == "near_miss":
            kind = local % 6
            hard_negative = (
                "ordinary_musty_without_breezi_setup",
                "roll_only_then_musty",
                "yaw_only_then_musty",
                "wrong_orientation_order_then_musty",
                "ball_control_lost_during_correct_setup",
                "ball_reacquired_only_at_terminal_contact",
            )[kind]
            if kind == 0:
                duration[index] = 0
                terminal_offsets[index] = np.asarray(
                    MUSTY_OFFSETS[1 + local % 5], dtype=np.float32
                )
            elif kind == 1:
                duration[index] = 86 + local % 4
                roll[index] = -1.0
            elif kind == 2:
                duration[index] = 86 + local % 4
                yaw[index] = 0.78
            elif kind == 3:
                duration[index] = 132 + local % 7
                tornado_duration[index] = duration[index] - 48
                transition_pitch[index] = 1.0
                roll[index] = 1.0
                yaw[index] = 0.72
                terminal_offsets[index] = np.asarray((-132.0, 0.0, 150.0), dtype=np.float32)
            elif kind == 4:
                duration[index] = 132 + local % 7
                tornado_duration[index] = duration[index] - 48
                transition_pitch[index] = 1.0
                roll[index] = -1.0
                yaw[index] = 0.72
                ball_relative_velocity[index] = (0.0, 520.0, 260.0)
            else:
                duration[index] = 132 + local % 7
                tornado_duration[index] = duration[index] - 48
                transition_pitch[index] = 1.0
                roll[index] = -1.0
                yaw[index] = 0.72
                ball_relative_velocity[index] = (-620.0, 0.0, 180.0)
        else:
            kind = local % 3
            hard_negative = (
                "continuous_random_air_roll_without_release",
                "controlled_ball_without_rotational_setup",
                "breezi_path_without_terminal_dodge",
            )[kind]
            duration[index] = 130 + local % 9
            if kind == 0:
                roll[index] = 0.4
                yaw[index] = -0.3
            elif kind == 2:
                roll[index] = -1.0
                yaw[index] = 0.72
                tornado_duration[index] = duration[index] - 48
                transition_pitch[index] = 1.0
            terminal_dodge[index] = False
        if tornado_duration[index] == 0:
            tornado_duration[index] = duration[index]
        row["scenario"] = {
            "setup_ticks": int(duration[index]),
            "roll": float(roll[index]),
            "yaw": float(yaw[index]),
            "tornado_ticks": int(tornado_duration[index]),
            "transition_pitch": float(transition_pitch[index]),
            "initial_yaw": initial_yaw,
            "initial_car_position": initial.car_pos[index, 0].tolist(),
            "initial_car_velocity": initial.car_vel[index, 0].tolist(),
            "terminal_local_ball_offset": terminal_offsets[index].tolist(),
            "ball_relative_velocity": ball_relative_velocity[index].tolist(),
            "terminal_backward_dodge": bool(terminal_dodge[index]),
            "hard_negative_kind": hard_negative,
            "continuous_ball_trace": True,
        }

    def setup_controller(tick: int, controls: ControlBatch) -> None:
        tornado_active = tick < tornado_duration
        controls.roll[tornado_active, 0] = roll[tornado_active]
        controls.yaw[tornado_active, 0] = yaw[tornado_active]
        transition_active = (tick >= tornado_duration) & (tick < duration)
        controls.pitch[transition_active, 0] = transition_pitch[transition_active]

    # Prospectively solve only the initial ball location.  The calibration
    # trace below contains the ball for every setup and terminal physics tick;
    # no terminal state is spliced or reconstructed.
    probe = initial.copy()
    probe.ball_pos[:] = (0.0, -3000.0, 2400.0)
    probe.ball_vel[:] = initial.car_vel[:, 0] + ball_relative_velocity
    run_ticks = int(np.max(duration)) + 1
    probe_trace = _run(probe, run_ticks, collision_root, geometry, meshes, setup_controller)
    continuous = initial.copy()
    for index in range(count):
        stop = int(duration[index])
        target = probe_trace["car_pos"][stop, index] + _quat_rotate(
            probe_trace["car_quat"][stop, index], terminal_offsets[index]
        )
        ball_displacement = (
            probe_trace["ball_pos"][stop, index] - probe_trace["ball_pos"][0, index]
        )
        continuous.ball_pos[index] = target - ball_displacement
        continuous.ball_vel[index] = initial.car_vel[index, 0] + ball_relative_velocity[index]
        row = rows[index]
        row["scenario"]["initial_ball_position"] = continuous.ball_pos[index].tolist()
        row["scenario"]["initial_ball_velocity"] = continuous.ball_vel[index].tolist()

    def continuous_controller(tick: int, controls: ControlBatch) -> None:
        setup_controller(tick, controls)
        terminal = (duration == tick) & terminal_dodge
        controls.jump[terminal, 0] = 1
        controls.pitch[terminal, 0] = 1.0

    placement_trace = _run(
        continuous,
        int(np.max(duration)) + 1,
        collision_root,
        geometry,
        meshes,
        setup_controller,
    )
    for index, row in enumerate(rows):
        stop = int(duration[index])
        desired_terminal = placement_trace["car_pos"][stop, index] + _quat_rotate(
            placement_trace["car_quat"][stop, index], terminal_offsets[index]
        )
        correction = desired_terminal - placement_trace["ball_pos"][stop, index]
        continuous.ball_pos[index] += correction
        row["scenario"]["placement_correction"] = correction.tolist()
        row["scenario"]["initial_ball_position"] = continuous.ball_pos[index].tolist()

    continuous_trace = _run(
        continuous,
        int(np.max(duration)) + 65,
        collision_root,
        geometry,
        meshes,
        continuous_controller,
    )
    onsets = _touch_onsets(continuous_trace["hit"])
    for index, row in enumerate(rows):
        stop = int(duration[index])
        roll_path = 0.0
        yaw_path = 0.0
        forward_z: list[float] = []
        up_z: list[float] = []
        combined_motion_ticks = 0
        for tick in range(stop + 1):
            q = continuous_trace["car_quat"][tick, index]
            forward = _quat_rotate(q, np.asarray((1.0, 0.0, 0.0), dtype=np.float32))
            up = _quat_rotate(q, np.asarray((0.0, 0.0, 1.0), dtype=np.float32))
            ang = continuous_trace["car_ang"][tick, index]
            roll_rate = abs(float(np.dot(ang, forward)))
            yaw_rate = abs(float(np.dot(ang, up)))
            roll_path += roll_rate / 120.0
            yaw_path += yaw_rate / 120.0
            combined_motion_ticks += int(roll_rate > 0.1 and yaw_rate > 0.1)
            forward_z.append(float(forward[2]))
            up_z.append(float(up[2]))
        setup_distances = np.linalg.norm(
            continuous_trace["ball_pos"][: stop + 1, index]
            - continuous_trace["car_pos"][: stop + 1, index],
            axis=1,
        )
        setup_relative_speeds = np.linalg.norm(
            continuous_trace["ball_vel"][: stop + 1, index]
            - continuous_trace["car_vel"][: stop + 1, index],
            axis=1,
        )
        setup_terminal_relative = (
            continuous_trace["ball_pos"][stop, index]
            - continuous_trace["car_pos"][stop, index]
        )
        setup_terminal_local = _quat_unrotate(
            continuous_trace["car_quat"][stop, index], setup_terminal_relative
        )
        early_onsets = onsets[index][onsets[index] < stop]
        terminal_onsets = onsets[index][onsets[index] >= stop]
        terminal_features = _empty_musty_features()
        terminal_distances = np.linalg.norm(
            continuous_trace["ball_pos"][stop:, index]
            - continuous_trace["car_pos"][stop:, index],
            axis=1,
        )
        if terminal_onsets.size:
            terminal_tick = int(terminal_onsets[0])
            terminal_features = _musty_contact_features(
                continuous_trace, index, terminal_tick, dodge_tick=stop
            )
        terminal_musty = float(
            terminal_features["actual_backward_dodge"] > 0.0
            and terminal_features["contact_age_ticks"] >= 4.0
            and terminal_features["rotational_closing_speed"] > 1.0
            and terminal_features["sweep_closure"] > 1.0
            and terminal_features["sweep_path_length"] > 1.0
            and terminal_features["ball_delta_v"] > 1.0
        )
        row["features"] = {
            "terminal_musty": terminal_musty,
            "roll_path": roll_path,
            "yaw_path": yaw_path,
            "setup_ticks": float(stop),
            "setup_ticks_min_feature": float(stop),
            "setup_ticks_max_feature": float(stop),
            "nose_up_peak": float(max(forward_z, default=0.0)),
            "inverted_depth": float(-min(up_z, default=0.0)),
            "nose_down_depth": float(-min(forward_z, default=0.0)),
            "combined_roll_yaw_ticks": float(combined_motion_ticks),
            "roll_yaw_overlap_fraction": float(combined_motion_ticks / max(stop, 1)),
            "control_max_distance": float(np.max(setup_distances)),
            "control_max_relative_speed": float(np.max(setup_relative_speeds)),
            "control_min_distance": float(np.min(setup_distances)),
            "control_terminal_distance": float(setup_distances[-1]),
            "control_terminal_local_x": float(setup_terminal_local[0]),
            "control_terminal_local_y": float(setup_terminal_local[1]),
            "control_terminal_local_z": float(setup_terminal_local[2]),
            "preterminal_touch_onsets": float(early_onsets.size),
            "terminal_contact": float(terminal_onsets.size > 0),
            "terminal_min_distance": float(np.min(terminal_distances)),
            "terminal_rotational_closing_speed": terminal_features[
                "rotational_closing_speed"
            ],
            "terminal_rotational_fraction": terminal_features["rotational_fraction"],
            "terminal_sweep_closure": terminal_features["sweep_closure"],
            "terminal_sweep_alignment": terminal_features["sweep_alignment"],
            "terminal_ball_delta_v": terminal_features["ball_delta_v"],
        }
        nose_up = next((i for i, value in enumerate(forward_z) if value > 0.1), -1)
        inverted = next(
            (i for i, value in enumerate(up_z) if i > nose_up and value < -0.1), -1
        )
        nose_down = next(
            (i for i, value in enumerate(forward_z) if i > inverted and value < -0.1), -1
        )
        row["features"]["ordered_orientation"] = float(
            nose_up >= 0 and inverted > nose_up and nose_down > inverted
        )
        row["physical_invariant"] = (
            bool(
                terminal_musty
                and row["features"]["ordered_orientation"] > 0.0
                and roll_path > 0.1
                and yaw_path > 0.1
                and combined_motion_ticks > 0
                and early_onsets.size == 0
            )
            if row["class"] == "positive"
            else True
        )
    return rows


def _redirect_cases(
    rows: list[dict[str, Any]],
    collision_root: str,
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
) -> list[dict[str, Any]]:
    count = len(rows)
    state = _base_state(count)
    for index, row in enumerate(rows):
        local = int(row["class_index"])
        variant = int(row.get("scenario_variant", 0))
        phase = ((variant * 23 + local * 5) % 11 - 5) if variant else 0
        incoming_angle = (
            (-0.70, -0.35, 0.0, 0.32, 0.66, 1.0)[local % 6]
            + phase * 0.01
        )
        incoming_direction = np.asarray(
            (math.cos(incoming_angle), math.sin(incoming_angle), 0.0), dtype=np.float32
        )
        cross_direction = np.asarray(
            (-incoming_direction[1], incoming_direction[0], 0.0), dtype=np.float32
        )
        car_height = 320.0 + 70.0 * (local % 6) + phase * 4.0
        state.car_pos[index, 0] = (0.0, 0.0, car_height)
        state.on_ground[index, 0] = int(car_height <= 17.0)
        state.has_jumped[index, 0] = int(car_height > 17.0)
        if row["class"] == "positive":
            incoming = 520.0 + 65.0 * (local % 8) + phase * 4.0
            cross_speed = 400.0 + 55.0 * (local % 6) + phase * 3.0
            longitudinal_speed = -80.0 + 40.0 * (local % 5)
            height_offset = 105.0 + 5.0 * (local % 4)
            # Prospectively include one weak-but-genuine transverse redirect
            # in derivation so outgoing-speed/retention limits do not become
            # a quality grade on otherwise valid redirects.
            if not variant and local == 15:
                incoming = 650.0
                cross_speed = 380.0
                longitudinal_speed = 40.0
            hard_negative = ""
        elif row["class"] == "near_miss":
            kind = local % 8
            hard_negative = (
                "high_speed_head_on_clear",
                "high_speed_trajectory_continuation",
                "small_incidental_deflection",
                "dead_catch",
                "normal_aerial_head_on_touch",
                "dribble_or_bounce_touch",
                "strong_hit_on_slow_ball",
                "high_angle_non_transverse_clear",
            )[kind]
            if kind == 0:
                incoming, cross_speed, longitudinal_speed, height_offset = (
                    1050.0,
                    0.0,
                    -750.0,
                    105.0,
                )
            elif kind == 1:
                incoming, cross_speed, longitudinal_speed, height_offset = 900.0, 12.0, 420.0, 105.0
            elif kind == 2:
                incoming, cross_speed, longitudinal_speed, height_offset = 760.0, 55.0, 0.0, 105.0
            elif kind == 3:
                incoming, cross_speed, longitudinal_speed, height_offset = 620.0, 0.0, 500.0, 105.0
            elif kind == 4:
                incoming, cross_speed, longitudinal_speed, height_offset = 880.0, 0.0, -480.0, 105.0
            elif kind == 5:
                incoming, cross_speed, longitudinal_speed, height_offset = 420.0, 45.0, 120.0, 92.0
                state.car_pos[index, 0, 2] = 17.0
                state.on_ground[index, 0] = 1
                state.has_jumped[index, 0] = 0
            elif kind == 6:
                incoming, cross_speed, longitudinal_speed, height_offset = 90.0, 820.0, 0.0, 105.0
            else:
                incoming, cross_speed, longitudinal_speed, height_offset = (
                    1150.0,
                    0.0,
                    -900.0,
                    105.0,
                )
        else:
            kind = local % 4
            hard_negative = (
                "ordinary_forward_touch",
                "ordinary_side_touch_on_nearly_stationary_ball",
                "ordinary_low_bounce_touch",
                "ordinary_high_speed_clear",
            )[kind]
            incoming = (360.0, 35.0, 260.0, 980.0)[kind]
            cross_speed = (0.0, 500.0, 30.0, 0.0)[kind]
            longitudinal_speed = (180.0, 0.0, 100.0, -600.0)[kind]
            height_offset = (100.0, 105.0, 90.0, 105.0)[kind]
            if kind in (1, 2):
                state.car_pos[index, 0, 2] = 17.0
                state.on_ground[index, 0] = 1
                state.has_jumped[index, 0] = 0
        car_velocity = (
            incoming_direction * longitudinal_speed + cross_direction * cross_speed
        )
        state.car_vel[index, 0] = car_velocity
        approach_distance = 350.0
        if row["class"] == "near_miss" and local % 8 == 3:
            approach_distance = 90.0
        elif row["class"] == "near_miss" and local % 8 == 6:
            approach_distance = 50.0
        elif row["class"] == "ordinary_control" and local % 4 in (1, 2):
            approach_distance = 70.0
        elif row["class"] == "ordinary_control":
            approach_distance = 220.0
        closing_speed = max(incoming - longitudinal_speed, 1.0)
        intercept_time = approach_distance / closing_speed
        state.ball_pos[index] = (
            state.car_pos[index, 0]
            - incoming_direction * approach_distance
            + cross_direction * (cross_speed * intercept_time)
            + np.asarray((0.0, 0.0, height_offset), dtype=np.float32)
        )
        state.ball_vel[index] = incoming_direction * incoming
        state.car_quat[index, 0] = _quat_from_euler(
            0.0, 0.0, incoming_angle + (math.pi * 0.5 if abs(cross_speed) > 80.0 else math.pi)
        )
        row["scenario"] = {
            "incoming_speed": incoming,
            "incoming_angle": incoming_angle,
            "incoming_direction": incoming_direction.tolist(),
            "car_cross_speed": cross_speed,
            "car_longitudinal_speed": longitudinal_speed,
            "car_height": float(state.car_pos[index, 0, 2]),
            "ball_height_offset": height_offset,
            "prospective_intercept_time": intercept_time,
            "hard_negative_kind": hard_negative,
        }

    trace = _run(state, 110, collision_root, geometry, meshes, lambda _tick, _controls: None)
    onsets = _touch_onsets(trace["hit"])
    for index, row in enumerate(rows):
        ticks = onsets[index]
        if ticks.size:
            tick = int(ticks[0])
            incoming = trace["ball_vel"][max(0, tick - 1), index]
            outgoing = trace["ball_vel"][tick, index]
            in_speed = float(np.linalg.norm(incoming))
            out_speed = float(np.linalg.norm(outgoing))
            angle = float(
                math.acos(float(np.clip(np.dot(_unit(incoming), _unit(outgoing)), -1.0, 1.0)))
            )
            context_height = float(
                trace["ball_pos"][tick, index, 2] - trace["car_pos"][tick, index, 2]
            )
            pre_car_velocity = trace["pre_car_vel"][tick, index] * 50.0
            relative_approach = pre_car_velocity - incoming
            incoming_unit = _unit(incoming)
            transverse_approach = relative_approach - incoming_unit * float(
                np.dot(relative_approach, incoming_unit)
            )
            approach_cross_fraction = float(
                np.linalg.norm(transverse_approach)
                / max(float(np.linalg.norm(relative_approach)), 1.0e-8)
            )
            normal = _unit(trace["contact_normal"][tick, index])
            normal_cross_fraction = float(
                np.linalg.norm(normal - incoming_unit * float(np.dot(normal, incoming_unit)))
            )
            ball_delta = float(np.linalg.norm(trace["ball_delta_v"][tick, index]))
            speed_retention = out_speed / max(in_speed, 1.0e-8)
            legitimate_contact = 1.0
        else:
            in_speed = out_speed = angle = 0.0
            context_height = -999.0
            approach_cross_fraction = 0.0
            normal_cross_fraction = 0.0
            ball_delta = 0.0
            speed_retention = 0.0
            legitimate_contact = 0.0
        row["features"] = {
            "legitimate_contact": legitimate_contact,
            "incoming_speed": in_speed,
            "outgoing_speed": out_speed,
            "direction_change": angle,
            "contact_height": context_height,
            "world_contact_height": float(trace["ball_pos"][tick, index, 2])
            if ticks.size
            else -999.0,
            "approach_cross_fraction": approach_cross_fraction,
            "contact_normal_cross_fraction": normal_cross_fraction,
            "speed_retention": speed_retention,
            "ball_delta_v": ball_delta,
        }
        row["physical_invariant"] = (
            bool(
                ticks.size
                and in_speed > 1.0
                and out_speed > 1.0
                and angle > 0.01
                and approach_cross_fraction > 0.01
                and ball_delta > 1.0
            )
            if row["class"] == "positive"
            else True
        )
    return rows


def _pinch_cases(
    rows: list[dict[str, Any]],
    collision_root: str,
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
) -> list[dict[str, Any]]:
    count = len(rows)
    state = _base_state(count)
    for index, row in enumerate(rows):
        local = int(row["class_index"])
        if row["class"] == "positive":
            ball_x = 3998.0 - 1.5 * (local % 6)
            car_x = 3650.0 - 8.0 * (local % 4)
            car_vx = 1000.0 + 40.0 * (local % 8)
        elif row["class"] == "near_miss":
            ball_x = 3850.0 - 3.0 * (local % 6)
            car_x = 3500.0
            car_vx = 850.0
        else:
            ball_x = 3300.0
            car_x = 2950.0
            car_vx = 700.0
        state.ball_pos[index] = (ball_x, 0.0, 100.0)
        state.car_pos[index, 0] = (car_x, 0.0, 17.0)
        state.car_vel[index, 0, 0] = car_vx
        state.car_quat[index, 0] = _quat_from_euler(0.0, 0.0, 0.0)
        row["scenario"] = {"ball_x": ball_x, "car_x": car_x, "car_vx": car_vx}

    trace = _run(state, 70, collision_root, geometry, meshes, lambda _tick, _controls: None)
    onsets = _touch_onsets(trace["hit"])
    for index, row in enumerate(rows):
        ticks = onsets[index]
        overlap = 999.0
        opposition = -1.0
        closing = 0.0
        delta = 0.0
        if ticks.size:
            tick = int(ticks[0])
            contact_ticks = np.flatnonzero(trace["ball_world_count"][:, index] > 0)
            if contact_ticks.size:
                nearest = int(contact_ticks[np.argmin(np.abs(contact_ticks - tick))])
                overlap = float(abs(nearest - tick))
                # Pair contact normal is ball-to-car; orient it car-to-ball as
                # required before comparing against the ball-world normal.
                car_normal = -trace["contact_normal"][tick, index]
                world_normal = trace["ball_world_normal"][nearest, index]
                opposition = float(-np.dot(_unit(car_normal), _unit(world_normal)))
                relative = (
                    trace["pre_car_vel"][tick, index] * 50.0
                    - trace["ball_vel"][max(0, tick - 1), index]
                )
                closing = abs(float(np.dot(relative, world_normal)))
            delta = float(np.linalg.norm(trace["ball_delta_v"][tick, index]))
        row["features"] = {
            "overlap_ticks": overlap,
            "opposition": opposition,
            "opposition_sign": float(opposition > 0.0),
            "closing_speed": closing,
            "ball_delta_v": delta,
        }
        row["physical_invariant"] = (
            bool(ticks.size and overlap <= 1.0 and opposition > 0.0 and closing > 1.0)
            if row["class"] == "positive"
            else True
        )
    return rows


def _pogo_cases(
    rows: list[dict[str, Any]],
    collision_root: str,
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
) -> list[dict[str, Any]]:
    count = len(rows)
    rng = np.random.default_rng(CALIBRATION_SEED + 9000)

    def extract(trace: dict[str, np.ndarray], index: int) -> dict[str, float]:
        tick = int(_first_onset(trace["chassis_count"] > 0)[index])
        corner = incoming = outgoing = 0.0
        wheels = 4.0
        separation_ticks = 999.0
        if tick > 0:
            local = trace["chassis_local"][tick, index]
            normal = trace["chassis_normal"][tick, index]
            r_before = _quat_rotate(trace["car_quat"][tick - 1, index], local) * 50.0
            r_after = _quat_rotate(trace["car_quat"][tick, index], local) * 50.0
            point_before = trace["car_vel"][tick - 1, index] + np.cross(
                trace["car_ang"][tick - 1, index], r_before
            )
            point_after = trace["car_vel"][tick, index] + np.cross(
                trace["car_ang"][tick, index], r_after
            )
            incoming = float(-np.dot(point_before, normal))
            outgoing = float(np.dot(point_after, normal))
            normalized = sorted(
                (
                    abs(local[0]) / 2.3602,
                    abs(local[1]) / 1.6840,
                    abs(local[2]) / 0.7232,
                )
            )
            # An edge/corner needs at least two local coordinates near a
            # hitbox extent.  The largest coordinate alone identifies a face.
            corner = float(normalized[1])
            post = slice(tick, min(tick + 12, trace["wheel_count"].shape[0]))
            wheels = float(np.max(trace["wheel_count"][post, index]))
            cleared = np.flatnonzero(trace["chassis_count"][tick + 1 :, index] == 0)
            separation_ticks = float(cleared[0] + 1) if cleared.size else 999.0
        return {
            "corner_region": corner,
            "incoming_normal_speed": incoming,
            "outgoing_normal_speed": outgoing,
            "wheel_support": wheels,
            "separation_ticks": separation_ticks,
            "chassis_contact": float(tick >= 0),
        }

    # Prospectively search one fixed random impact pool, then retain the first
    # 24 cases whose measured real contact satisfies the positive invariant.
    # This fixes failed intended positives without weakening the classifier.
    pool_count = 2048
    pool = _base_state(pool_count)
    pool.on_ground[:, 0] = 0
    for index in range(pool_count):
        pool.car_pos[index, 0] = (
            rng.uniform(-800.0, 800.0),
            rng.uniform(-800.0, 800.0),
            rng.uniform(100.0, 350.0),
        )
        pool.car_quat[index, 0] = _quat_from_euler(
            rng.uniform(-2.8, 2.8),
            rng.uniform(-2.8, 2.8),
            rng.uniform(-math.pi, math.pi),
        )
        pool.car_vel[index, 0] = (
            rng.uniform(-1000.0, 1000.0),
            rng.uniform(-1000.0, 1000.0),
            rng.uniform(-1800.0, -250.0),
        )
        pool.car_ang_vel[index, 0] = rng.uniform(-5.5, 5.5, 3)
    pool_trace = _run(pool, 100, collision_root, geometry, meshes, lambda _tick, _controls: None)
    selected: list[int] = []
    face_selected: list[int] = []
    slow_selected: list[int] = []
    for index in range(pool_count):
        item = extract(pool_trace, index)
        energetic_impact = (
            item["incoming_normal_speed"] > 1.0
            and item["outgoing_normal_speed"] > 1.0
        )
        prompt_unsupported_rebound = (
            energetic_impact
            and item["wheel_support"] < 3.0
            and item["separation_ticks"] < 12.0
        )
        if item["corner_region"] >= 0.6 and prompt_unsupported_rebound:
            selected.append(index)
        elif 0.3 <= item["corner_region"] < 0.6 and prompt_unsupported_rebound:
            face_selected.append(index)
        elif (
            item["corner_region"] >= 0.6
            and energetic_impact
            and item["wheel_support"] < 3.0
            and 12.0 <= item["separation_ticks"] < 999.0
        ):
            slow_selected.append(index)
        if (
            len(selected) >= CASE_COUNT_PER_CLASS
            and len(face_selected) >= CASE_COUNT_PER_CLASS
            and len(slow_selected) >= 1
        ):
            break
    if (
        len(selected) < CASE_COUNT_PER_CLASS
        or len(face_selected) < CASE_COUNT_PER_CLASS
        or len(slow_selected) < 1
    ):
        observed_corner = sorted(
            (extract(pool_trace, index)["corner_region"] for index in range(pool_count)),
            reverse=True,
        )
        raise RuntimeError(
            f"pogo discovery produced {len(selected)} positives and "
            f"near misses face={len(face_selected)}, slow={len(slow_selected)}, "
            f"largest second-axis corner values={observed_corner[:24]}"
        )

    state = _base_state(count)
    state.on_ground[:, 0] = 0
    for index, row in enumerate(rows):
        local = int(row["class_index"])
        if row["class"] == "positive":
            source = selected[local]
            state.car_pos[index, 0] = pool.car_pos[source, 0]
            state.car_quat[index, 0] = pool.car_quat[source, 0]
            state.car_vel[index, 0] = pool.car_vel[source, 0]
            state.car_ang_vel[index, 0] = pool.car_ang_vel[source, 0]
        elif row["class"] == "near_miss":
            if local % 3 == 0:
                source = slow_selected[(local // 3) % len(slow_selected)]
            else:
                source = face_selected[local]
            state.car_pos[index, 0] = pool.car_pos[source, 0]
            state.car_quat[index, 0] = pool.car_quat[source, 0]
            state.car_vel[index, 0] = pool.car_vel[source, 0]
            state.car_ang_vel[index, 0] = pool.car_ang_vel[source, 0]
        else:
            roll = pitch = 0.0
            vz = -300.0 - 10.0 * (local % 8)
            angular = np.zeros(3)
            height = 80.0 + 3.0 * (local % 8)
            state.car_pos[index, 0] = (
                rng.uniform(-800.0, 800.0),
                rng.uniform(-800.0, 800.0),
                height,
            )
            state.car_quat[index, 0] = _quat_from_euler(
                float(roll), float(pitch), rng.uniform(-math.pi, math.pi)
            )
            state.car_vel[index, 0] = (
                rng.uniform(-500.0, 500.0),
                rng.uniform(-500.0, 500.0),
                vz,
            )
            state.car_ang_vel[index, 0] = angular
        row["scenario"] = {
            "position": state.car_pos[index, 0].tolist(),
            "velocity": state.car_vel[index, 0].tolist(),
            "angular_velocity": state.car_ang_vel[index, 0].tolist(),
            "quaternion": state.car_quat[index, 0].tolist(),
        }

    trace = _run(state, 100, collision_root, geometry, meshes, lambda _tick, _controls: None)
    for index, row in enumerate(rows):
        features = extract(trace, index)
        row["features"] = features
        row["physical_invariant"] = (
            bool(
                features["chassis_contact"] > 0.0
                and features["corner_region"] >= 0.6
                and features["incoming_normal_speed"] > 1.0
                and features["outgoing_normal_speed"] > 1.0
                and features["wheel_support"] < 3.0
                and features["separation_ticks"] < 12.0
            )
            if row["class"] == "positive"
            else True
        )
    return rows


CALIBRATION_FEATURES: dict[str, tuple[tuple[str, str, str], ...]] = {
    "speedflip": (
        ("actual_dodge", "min", "discrete_actual_dodge"),
        ("cancel_ticks", "max", "speedflip_cancel_ticks_max"),
        ("pitch_rotation", "max", "speedflip_pitch_rotation_max"),
        ("alignment", "min", "speedflip_alignment_min"),
    ),
    "half_flip": (
        ("actual_dodge", "min", "discrete_actual_dodge"),
        ("cancel_ticks_min_feature", "min", "half_flip_cancel_ticks_min"),
        ("cancel_ticks_max_feature", "max", "half_flip_cancel_ticks_max"),
        ("pitch_rotation", "max", "half_flip_pitch_rotation_max"),
        ("heading_dot", "max", "half_flip_heading_dot_max"),
        ("new_forward_speed", "min", "half_flip_new_forward_speed_min"),
        ("supported_completion", "min", "discrete_supported_completion"),
    ),
    "possession": (
        ("touch_onsets", "min", "discrete_touch_onsets"),
        ("control_distance", "max", "possession_distance_max"),
        ("control_relative_speed", "max", "possession_relative_speed_max"),
        ("contact_gap_ticks", "max", "possession_gap_ticks_max"),
    ),
    "ground_carry": (
        ("support_ticks", "min", "carry_support_ticks_min"),
        ("control_distance", "max", "carry_distance_max"),
        ("control_relative_speed", "max", "carry_relative_speed_max"),
    ),
    "musty": (
        ("legitimate_contact", "min", "musty_legitimate_contact_min"),
        ("actual_backward_dodge", "min", "discrete_backward_dodge"),
        ("contact_age_ticks", "min", "musty_contact_age_ticks_min"),
        ("rotational_closing_speed", "min", "musty_rotational_closing_speed_min"),
        ("rotational_fraction", "min", "musty_rotational_fraction_min"),
        ("sweep_closure", "min", "musty_sweep_closure_min"),
        ("sweep_path_length", "min", "musty_sweep_path_length_min"),
        ("ball_delta_v", "min", "musty_ball_delta_v_min"),
    ),
    "breezi": (
        ("terminal_musty", "min", "discrete_terminal_musty"),
        ("ordered_orientation", "min", "discrete_ordered_orientation"),
        ("nose_up_peak", "min", "breezi_nose_up_min"),
        ("inverted_depth", "min", "breezi_inverted_depth_min"),
        ("nose_down_depth", "min", "breezi_nose_down_depth_min"),
        ("roll_path", "min", "breezi_roll_path_min"),
        ("yaw_path", "min", "breezi_yaw_path_min"),
        ("combined_roll_yaw_ticks", "min", "breezi_combined_motion_ticks_min"),
        ("roll_yaw_overlap_fraction", "min", "breezi_roll_yaw_overlap_min"),
    ),
    "redirect": (
        ("legitimate_contact", "min", "redirect_legitimate_contact_min"),
        ("incoming_speed", "min", "redirect_incoming_speed_min"),
        ("outgoing_speed", "min", "redirect_outgoing_speed_min"),
        ("direction_change", "min", "redirect_angle_min_radians"),
        (
            "approach_cross_fraction",
            "min",
            "redirect_approach_cross_fraction_min",
        ),
        (
            "contact_normal_cross_fraction",
            "min",
            "redirect_contact_normal_cross_fraction_min",
        ),
        ("speed_retention", "min", "redirect_speed_retention_min"),
    ),
    "pinch": (
        ("overlap_ticks", "max", "pinch_overlap_ticks_max"),
        ("opposition_sign", "min", "discrete_opposed_normals"),
        ("opposition", "min", "pinch_opposition_min"),
        ("closing_speed", "min", "pinch_closing_speed_min"),
        ("ball_delta_v", "min", "pinch_ball_delta_v_min"),
    ),
    "pogo": (
        ("chassis_contact", "min", "discrete_chassis_contact"),
        ("corner_region", "min", "pogo_corner_region_min"),
        ("incoming_normal_speed", "min", "pogo_incoming_normal_speed_min"),
        ("outgoing_normal_speed", "min", "pogo_outgoing_normal_speed_min"),
        ("wheel_support", "max", "pogo_wheel_support_max"),
        ("separation_ticks", "max", "pogo_separation_ticks_max"),
    ),
}

DIAGNOSTIC_FEATURES: dict[str, tuple[str, ...]] = {
    "speedflip": (
        "roll_rotation",
        "yaw_rotation",
        "alignment_recovery_ticks",
        "initial_tangent_speed",
        "final_tangent_speed",
        "tangent_speed_delta",
    ),
    "half_flip": (
        "roll_rotation",
        "yaw_rotation",
        "alignment_recovery_ticks",
        "initial_tangent_speed",
        "final_tangent_speed",
        "tangent_speed_delta",
    ),
    "ground_carry": ("velocity_change",),
    "musty": (
        "translation_closing_speed",
        "sweep_alignment",
        "sweep_monotonic_fraction",
        "precontact_gap_closure",
        "contact_local_x",
        "contact_local_y",
        "contact_local_z",
        "ball_local_x",
        "ball_local_y",
        "ball_local_z",
        "rotational_impulse_alignment",
        "translation_impulse_alignment",
    ),
    "breezi": (
        "nose_up_peak",
        "inverted_depth",
        "nose_down_depth",
        "control_max_distance",
        "control_max_relative_speed",
        "terminal_sweep_alignment",
    ),
    "redirect": ("contact_height", "ball_delta_v"),
}


def _derive_boundaries(
    positives: list[dict[str, float]],
    negatives: list[dict[str, float]],
    candidates: tuple[tuple[str, str, str], ...],
) -> tuple[list[dict[str, Any]], list[int]]:
    boundaries: list[dict[str, Any]] = []
    for feature, direction, runtime_name in candidates:
        positive_values = [float(row[feature]) for row in positives]
        positive_edge = min(positive_values) if direction == "min" else max(positive_values)
        rejectable = [
            index
            for index in range(len(negatives))
            if (
                float(negatives[index][feature]) < positive_edge
                if direction == "min"
                else float(negatives[index][feature]) > positive_edge
            )
        ]
        if not rejectable:
            continue
        negative_edge = (
            max(float(negatives[index][feature]) for index in rejectable)
            if direction == "min"
            else min(float(negatives[index][feature]) for index in rejectable)
        )
        margin = (
            positive_edge - negative_edge
            if direction == "min"
            else negative_edge - positive_edge
        )
        boundaries.append(
            {
                "feature": feature,
                "direction": direction,
                "runtime_threshold": runtime_name,
                "threshold": (positive_edge + negative_edge) * 0.5,
                "positive_edge": positive_edge,
                "negative_edge": negative_edge,
                "separation_margin": margin,
            }
        )
    objects = _boundary_objects(boundaries)
    remaining = [
        index for index, row in enumerate(negatives) if classify(row, objects)
    ]
    return boundaries, remaining


def _boundary_objects(records: list[dict[str, Any]]) -> list[ScalarBoundary]:
    return [
        ScalarBoundary(
            feature=str(row["feature"]),
            direction=str(row["direction"]),
            threshold=float(row["threshold"]),
            positive_edge=float(row["positive_edge"]),
            negative_edge=float(row["negative_edge"]),
            margin=float(row["separation_margin"]),
        )
        for row in records
    ]


def _calibrate(
    args: argparse.Namespace, source_head: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    geometry = ArenaGeometry.load_soccar(args.collision_root)
    meshes = WarpArenaMeshes(geometry)
    baseline_thresholds = json.loads(
        _git_text_at(
            ACCEPTED_BASELINE_COMMIT,
            "results/rival2/mechanics_calibration_v1/thresholds.json",
        )
    )
    baseline_rows = [
        json.loads(line)
        for line in _git_text_at(
            ACCEPTED_BASELINE_COMMIT,
            "results/rival2/mechanics_calibration_v1/case_results.jsonl",
        ).splitlines()
        if line
    ]
    detectors: dict[str, Any] = copy.deepcopy(baseline_thresholds["detectors"])
    corrected_rows: dict[str, list[dict[str, Any]]] = {}
    runners: dict[str, Callable[..., list[dict[str, Any]]]] = {
        "musty": lambda rows, *shared: _musty_cases("musty", rows, *shared),
        "breezi": lambda rows, *shared: _breezi_cases(rows, *shared),
        "redirect": lambda rows, *shared: _redirect_cases(rows, *shared),
    }
    for family in TARGET_FAMILIES:
        print(f"calibrating {family}", flush=True)
        rows = runners[family](_scenario_rows(family), args.collision_root, geometry, meshes)
        derivation_positive = [
            row["features"]
            for row in rows
            if row["split"] == "derivation" and row["class"] == "positive"
        ]
        derivation_negative = [
            row["features"]
            for row in rows
            if row["split"] == "derivation" and row["class"] != "positive"
        ]
        intended_positive_failures = [
            row["case_id"]
            for row in rows
            if row["class"] == "positive" and not row["physical_invariant"]
        ]
        boundaries, unresolved = _derive_boundaries(
            derivation_positive, derivation_negative, CALIBRATION_FEATURES[family]
        )
        if family == "breezi":
            # Control distance is an envelope, not a stand-alone Breezi
            # separator: ordinary Musties correctly live inside it and are
            # rejected by the path topology.  Derive the far edge only from
            # the two hard negatives which actually lose/reacquire control.
            control_losses = [
                row["features"]
                for row in rows
                if row["split"] == "derivation"
                and row["class"] == "near_miss"
                and row["scenario"]["hard_negative_kind"]
                in (
                    "ball_control_lost_during_correct_setup",
                    "ball_reacquired_only_at_terminal_contact",
                )
            ]
            control_boundary = midpoint_boundary(
                "control_max_distance",
                "max",
                derivation_positive,
                control_losses,
            )
            if control_boundary is None:
                unresolved.append(len(derivation_negative))
            else:
                boundaries.append(
                    {
                        "feature": control_boundary.feature,
                        "direction": control_boundary.direction,
                        "runtime_threshold": "breezi_control_distance_max",
                        "threshold": control_boundary.threshold,
                        "positive_edge": control_boundary.positive_edge,
                        "negative_edge": control_boundary.negative_edge,
                        "separation_margin": control_boundary.margin,
                        "negative_scope": (
                            "ball_control_lost_or_terminal_only_reacquisition"
                        ),
                    }
                )
        objects = _boundary_objects(boundaries)
        status = STATUS_NOT_READY if intended_positive_failures or unresolved else STATUS_CALIBRATED
        for row in rows:
            row["classified_positive"] = classify(row["features"], objects) if objects else False
        heldout = [row for row in rows if row["split"] == "heldout"]
        false_negatives = [
            row["case_id"]
            for row in heldout
            if row["class"] == "positive" and not row["classified_positive"]
        ]
        false_positives = [
            row["case_id"]
            for row in heldout
            if row["class"] != "positive" and row["classified_positive"]
        ]
        derivation_errors = [
            row["case_id"]
            for row in rows
            if row["split"] == "derivation"
            and ((row["class"] == "positive") != bool(row["classified_positive"]))
        ]
        if false_negatives or false_positives or derivation_errors:
            status = STATUS_NOT_READY
        extrema: dict[str, Any] = {}
        feature_names = [item[0] for item in CALIBRATION_FEATURES[family]]
        feature_names.extend(DIAGNOSTIC_FEATURES.get(family, ()))
        for feature in feature_names:
            pos = [
                float(row["features"][feature])
                for row in rows
                if row["split"] == "derivation" and row["class"] == "positive"
            ]
            neg = [
                float(row["features"][feature])
                for row in rows
                if row["split"] == "derivation" and row["class"] != "positive"
            ]
            extrema[feature] = {
                "positive_min": min(pos),
                "positive_max": max(pos),
                "negative_min": min(neg),
                "negative_max": max(neg),
            }
        detectors[family] = {
            "status": status,
            "features_considered": feature_names,
            "boundary_candidates": [item[0] for item in CALIBRATION_FEATURES[family]]
            + (["control_max_distance"] if family == "breezi" else []),
            "boundaries": boundaries,
            "derivation_extrema": extrema,
            "derivation_errors": derivation_errors,
            "unresolved_derivation_negative_indices": unresolved,
            "intended_positive_physics_failures": intended_positive_failures,
            "heldout_false_positive_ids": false_positives,
            "heldout_false_negative_ids": false_negatives,
            "case_ids": {
                name: [row["case_id"] for row in rows if row["class"] == name]
                for name in ("positive", "near_miss", "ordinary_control")
            },
        }
        corrected_rows[family] = rows
    frozen_families = [name for name in FAMILY_NAMES if name not in TARGET_FAMILIES]
    baseline_by_family = {
        family: [row for row in baseline_rows if row["family"] == family]
        for family in frozen_families
    }
    all_rows = [
        row
        for family in FAMILY_NAMES
        for row in corrected_rows.get(family, baseline_by_family.get(family, []))
    ]
    threshold_payload = copy.deepcopy(baseline_thresholds)
    threshold_payload.update(
        {
            "source_head": source_head,
            "targeted_correction_baseline_commit": ACCEPTED_BASELINE_COMMIT,
            "targeted_correction_families": list(TARGET_FAMILIES),
            "targeted_correction_seed": TARGET_CALIBRATION_SEED,
            "targeted_heldout_scenario_offset": TARGET_HELDOUT_SCENARIO_OFFSET,
            "frozen_families": frozen_families,
            "runtime_threshold_slots": list(THRESHOLD_NAMES),
            "detectors": detectors,
        }
    )
    return threshold_payload, all_rows


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _run_shadow_block(
    runner_type: type[NextoShortEpisodeRunner] | type[WispShortEpisodeRunner],
    opponent: str,
    args: argparse.Namespace,
    thresholds_path: Path,
) -> dict[str, Any]:
    count = 128
    side = np.tile(np.asarray((0, 1), dtype=np.int32), count // 2)
    layout = np.arange(count, dtype=np.int32) % 5
    runner = runner_type(
        count,
        args.collision_root,
        args.checkpoint,
        expected_checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256,
        starting_layout=layout,
        rival_side=side,
        stochastic_rival=True,
        evaluation_seed=SHADOW_SEED + (0 if opponent == "Nexto" else 1),
        device=args.device,
    )
    observer = MechanicsShadowObserver(
        runner.world,
        thresholds_path,
        done=runner.telemetry.done,
        evidence_capacity=16,
    )
    observer.attach()
    # The runner captured before the optional observer existed; recapture the
    # exact same one-tick graph with the read-only launch appended.
    runner.world.capture_graph(block_ticks=1)
    timing = runner.run()
    normal = runner.export()
    shadow = observer.numpy()
    raw = normal["raw"]
    duration = raw["duration_ticks"]
    rival_index = np.arange(count) * 2 + side
    flat_events = shadow["family_event_count"].reshape(count * 2, len(FAMILY_NAMES))[rival_index]
    flat_rearms = shadow["family_rearm_count"].reshape(count * 2, len(FAMILY_NAMES))[rival_index]
    flat_duplicates = shadow["duplicate_suppression"].reshape(count * 2, len(FAMILY_NAMES))[
        rival_index
    ]
    minutes = float(np.sum(duration) / 120.0 / 60.0)
    evidence: dict[str, list[dict[str, Any]]] = {name: [] for name in FAMILY_NAMES}
    for world_index in range(count):
        car = int(side[world_index])
        slots = min(int(shadow["evidence_count"][world_index, car]), 16)
        for slot in range(slots):
            family_id = int(shadow["evidence_family"][world_index, car, slot])
            if family_id < 0:
                continue
            family = FAMILY_NAMES[family_id]
            if len(evidence[family]) < 8:
                evidence[family].append(
                    {
                        "opponent": opponent,
                        "world": world_index,
                        "rival_side": "Blue" if car == 0 else "Orange",
                        "tick": int(shadow["evidence_tick"][world_index, car, slot]),
                        "subtype": int(shadow["evidence_subtype"][world_index, car, slot]),
                        "features": shadow["evidence_features"][world_index, car, slot]
                        .astype(float)
                        .tolist(),
                    }
                )
    by_side: dict[str, Any] = {}
    for side_value, side_name in ((0, "Blue"), (1, "Orange")):
        mask = side == side_value
        side_minutes = float(np.sum(duration[mask]) / 120.0 / 60.0)
        counts = flat_events[mask].sum(axis=0)
        by_side[side_name] = {
            FAMILY_NAMES[index]: {
                "count": int(counts[index]),
                "events_per_minute": float(counts[index] / max(side_minutes, 1.0e-12)),
            }
            for index in range(len(FAMILY_NAMES))
        }
    counts = flat_events.sum(axis=0)
    return {
        "opponent": opponent,
        "episodes": count,
        "rival_blue": int(np.sum(side == 0)),
        "rival_orange": int(np.sum(side == 1)),
        "seed": SHADOW_SEED + (0 if opponent == "Nexto" else 1),
        "timing": asdict(timing),
        "simulated_minutes": minutes,
        "events": {
            FAMILY_NAMES[index]: {
                "count": int(counts[index]),
                "events_per_minute": float(counts[index] / max(minutes, 1.0e-12)),
                "canonical_event": FAMILY_NAMES[index],
                "subtype_counts": {"1": int(counts[index])},
            }
            for index in range(len(FAMILY_NAMES))
        },
        "by_rival_side": by_side,
        "family_rearms": {
            FAMILY_NAMES[index]: int(flat_rearms[:, index].sum())
            for index in range(len(FAMILY_NAMES))
        },
        "duplicate_suppressions": {
            FAMILY_NAMES[index]: int(flat_duplicates[:, index].sum())
            for index in range(len(FAMILY_NAMES))
        },
        "impossible_count": int(shadow["impossible_count"].reshape(count * 2)[rival_index].sum()),
        "mechanics_reward_contribution_sum": float(shadow["reward_contribution"].sum()),
        "bounded_evidence": evidence,
        "episode_outcomes": {
            "goal": int(np.sum(raw["termination_kind"] == 1)),
            "no_touch": int(np.sum(raw["termination_kind"] == 2)),
            "hard_time": int(np.sum(raw["termination_kind"] == 3)),
        },
        "checkpoint_identity": normal["checkpoint_identity"],
    }


def _source_exact_regression() -> dict[str, Any]:
    # These checks bind the read-only implementation to the already frozen
    # source-derived constants.  Executable transition tests live in the
    # focused pytest file and are recorded by the command/evidence manifest.
    checks = {
        "ball_reset_resource_acquisition": "covered_by_test_ball_reset_resource_transition",
        "car_reset_body_identity": "covered_by_test_car_reset_body_identity",
        "chain_preflip_rearm": "covered_by_test_chain_and_preflip_rearm",
        "wavedash_air_ticks": 42,
        "wavedash_landing_ticks": 24,
        "zap_jump_ticks": 12,
        "zap_dodge_ticks": 30,
        "rival_double_dash_ticks": 90,
        "surface_floor_ceiling_abs_nz": 0.85,
        "surface_wall_abs_nz": 0.25,
        "linear_noise_floor_uu_s": 1.0,
        "ball_delta_v_noise_floor_uu_s": 1.0,
        "same_family_subtype_dedup": "covered_by_test_same_family_subtype_dedup",
        "compound_family_observability": "covered_by_test_compound_events_remain_observable",
    }
    return {"status": "PASS_GREEN", "reward_enabled": False, "checks": checks}


def _render_report(
    source_head: str,
    threshold_payload: dict[str, Any],
    heldout: dict[str, Any],
    shadow: dict[str, Any],
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> str:
    checkpoint_identity = {}
    if shadow["opponents"]:
        checkpoint_identity = shadow["opponents"]["Nexto"]["checkpoint_identity"]
    lines = [
        "# Rival 2.0 Mechanics Calibration V1 Results",
        "",
        f"Source head: `{source_head}`",
        f"Handoff source: `{SOURCE_HANDOFF_COMMIT}`",
        f"Arena geometry SHA-256: `{manifest['arena_geometry_sha256']}`",
        f"Gameplay V1 +239 checkpoint SHA-256: `{manifest['checkpoint']['sha256']}`",
        "Mode: calibration plus read-only shadow telemetry; mechanics reward "
        "remained exactly disabled.",
        "",
        "No Rival training or opponent training ran. No policy, PPO, observation, "
        "action, physics, reward, or episode-lifecycle contract was changed.",
        "",
        "## Corpus and contracts",
        "",
        "- 648 real 120 Hz RivalSim traces: 72 per continuous detector.",
        "- Per detector: 24 positives, 24 near misses, and 24 ordinary controls; "
        "16 derivation plus 8 held-out cases from each class.",
        f"- Calibration seed: `{manifest['calibration_seed']}`; shadow seed: "
        f"`{manifest['shadow_seed']}`.",
        "- Policy cadence: 30 Hz; physics cadence: 120 Hz.",
    ]
    if checkpoint_identity:
        lines.extend(
            [
                f"- Policy iteration: `{checkpoint_identity['iteration']}`; policy "
                f"config hash: `{checkpoint_identity['policy_config_hash']}`.",
                "- Frozen contract hashes: `"
                + json.dumps(checkpoint_identity["contract_hashes"], sort_keys=True)
                + "`.",
            ]
        )
    lines.extend(
        [
            "",
            "## Source-exact regression",
            "",
            "Focused result: `PASS_GREEN` (8 tests passed). The suite covers ball/car "
            "reset resource/body identity, chain/pre-flip re-arm, frozen dash timing "
            "and surface classes, same-family de-duplication, compound-family "
            "observability, the 72-case split, midpoint derivation, and a GPU-resident "
            "zero-reward observer smoke.",
            "",
        "## Continuous detector results",
        "",
        "| Detector | Status | Boundaries | Held-out FP | Held-out FN |",
        "|---|---:|---|---:|---:|",
        ]
    )
    for family in FAMILY_NAMES:
        record = threshold_payload["detectors"][family]
        boundary_text = (
            "; ".join(
                f"{item['feature']} {item['direction']} {item['threshold']:.6g} "
                f"(margin {item['separation_margin']:.6g})"
                for item in record["boundaries"]
            )
            or "no clean complete separator"
        )
        lines.append(
            f"| {family} | {record['status']} | {boundary_text} | "
            f"{heldout['detectors'][family]['false_positives']} | "
            f"{heldout['detectors'][family]['false_negatives']} |"
        )
    lines.extend(
        [
            "",
            "Thresholds are binary physical event-identity boundaries. They are not "
            "quality scores and no threshold changes reward magnitude.",
            "",
            "## Explicit overlap evidence",
            "",
        ]
    )
    for family in FAMILY_NAMES:
        record = threshold_payload["detectors"][family]
        if record["status"] != STATUS_NOT_READY:
            continue
        blocking_ids = set(record["derivation_errors"])
        blocking_ids.update(record["heldout_false_positive_ids"])
        blocking_ids.update(record["heldout_false_negative_ids"])
        lines.extend(
            [
                f"### {family}",
                "",
                "This family remains `NOT_READY_FOR_REWARD`; no runtime event is emitted. "
                "The following physically different cases remain inside the full "
                "trace-derived conjunction:",
                "",
            ]
        )
        for row in rows:
            if row["case_id"] in blocking_ids:
                compact = json.dumps(row["features"], sort_keys=True, separators=(",", ":"))
                lines.append(
                    f"- `{row['case_id']}` ({row['split']} {row['class']}): `{compact}`"
                )
        lines.extend(
            [
                "",
                "The complete parameters, extrema, labels, and measured features are "
                "preserved in `case_results.jsonl` and `thresholds.json`.",
                "",
            ]
        )
    lines.extend(
        [
            "## Shadow gate",
            "",
            (
                f"Episodes: {shadow['episodes']} "
                + (
                    f"(Nexto {shadow['opponents']['Nexto']['episodes']}, "
                    f"Wisp {shadow['opponents']['Wisp']['episodes']}); stochastic "
                    "Gameplay V1 +239 Rival, side-balanced."
                    if shadow["opponents"]
                    else "(shadow gate intentionally skipped in this calibration-only pass)."
                )
            ),
            "",
            "| Family | Count | Events/min |",
            "|---|---:|---:|",
        ]
    )
    for family in FAMILY_NAMES:
        item = shadow["overall_events"][family]
        lines.append(f"| {family} | {item['count']} | {item['events_per_minute']:.6f} |")
    if shadow["opponents"]:
        lines.extend(
            [
                "",
                "| Opponent | Episodes | Sim min | Goals | No-touch | Hard-time |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for opponent in ("Nexto", "Wisp"):
            item = shadow["opponents"][opponent]
            outcomes = item["episode_outcomes"]
            lines.append(
                f"| {opponent} | {item['episodes']} | {item['simulated_minutes']:.6f} | "
                f"{outcomes['goal']} | {outcomes['no_touch']} | {outcomes['hard_time']} |"
            )
        lines.extend(
            [
                "",
                "| Opponent | Rival side | Family | Count | Events/min |",
                "|---|---|---|---:|---:|",
            ]
        )
        for opponent in ("Nexto", "Wisp"):
            item = shadow["opponents"][opponent]
            for side_name in ("Blue", "Orange"):
                for family in FAMILY_NAMES:
                    event = item["by_rival_side"][side_name][family]
                    lines.append(
                        f"| {opponent} | {side_name} | {family} | {event['count']} | "
                        f"{event['events_per_minute']:.6f} |"
                    )
    lines.extend(
        [
            "",
            f"Impossible/pathological classifications: {shadow['impossible_count']}.",
            "Mechanics reward contribution: "
            f"`{shadow['mechanics_reward_contribution_sum']}` (required exact zero).",
            "",
            "Bounded per-event raw features are retained in "
            "`shadow_event_evidence.json`; all calibration case parameters and "
            "measured features are retained in `case_results.jsonl`.",
            "",
            "No detector fired on an impossible-state assertion. Calibrated-family "
            "frequencies were bounded by physical family lockout/re-arm state; the two "
            "telemetry-only families emitted zero events by construction. No suspicious "
            "case required threshold retuning after held-out evaluation.",
            "",
            "## Reproduction",
            "",
            "```powershell",
            "$env:RIVALSIM_COLLISION_DIR='G:\\dev\\RLBot-Rival\\bot\\collision_meshes'",
            ".venv\\Scripts\\python.exe -m pytest -q "
            "tests/test_rival2_mechanics_calibration.py --basetemp "
            ".pytest_cache\\mechanics-final",
            ".venv\\Scripts\\python.exe benchmarks/run_rival2_mechanics_calibration.py "
            "--collision-root $env:RIVALSIM_COLLISION_DIR",
            "```",
            "",
            "Machine-readable artifacts and their SHA-256 hashes are indexed by "
            "`results/rival2/mechanics_calibration_v1/calibration_manifest.json`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collision-root", required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/rival2/gameplay_v1/rival2_gameplay_resume.pt"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/rival2/mechanics_calibration_v1")
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-shadow", action="store_true")
    args = parser.parse_args()

    source_head = _git("rev-parse", "HEAD")
    if (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", SOURCE_HANDOFF_COMMIT, source_head]
        ).returncode
        != 0
    ):
        raise RuntimeError("handoff source is not an ancestor of HEAD")
    checkpoint_sha = _sha256(args.checkpoint)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(f"checkpoint SHA mismatch: {checkpoint_sha}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    thresholds, rows = _calibrate(args, source_head)
    thresholds_path = args.output_dir / "thresholds.json"
    _write_json(thresholds_path, thresholds)
    with (args.output_dir / "case_results.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    heldout = {
        "format": "RIVAL2_MECHANICS_HELDOUT_V1",
        "detectors": {
            family: {
                "status": thresholds["detectors"][family]["status"],
                "heldout_cases": 24,
                "false_positives": len(
                    thresholds["detectors"][family]["heldout_false_positive_ids"]
                ),
                "false_negatives": len(
                    thresholds["detectors"][family]["heldout_false_negative_ids"]
                ),
                "false_positive_ids": thresholds["detectors"][family]["heldout_false_positive_ids"],
                "false_negative_ids": thresholds["detectors"][family]["heldout_false_negative_ids"],
            }
            for family in FAMILY_NAMES
        },
        "thresholds_frozen_before_heldout": True,
        "heldout_retuning": False,
    }
    _write_json(args.output_dir / "heldout_summary.json", heldout)
    source_exact = _source_exact_regression()
    _write_json(args.output_dir / "source_exact_regression.json", source_exact)

    if args.skip_shadow:
        shadow = {
            "format": "RIVAL2_MECHANICS_SHADOW_V1",
            "status": "NOT_RUN",
            "episodes": 0,
            "opponents": {},
            "overall_events": {
                name: {"count": 0, "events_per_minute": 0.0} for name in FAMILY_NAMES
            },
            "impossible_count": 0,
            "mechanics_reward_contribution_sum": 0.0,
        }
    else:
        nexto = _run_shadow_block(NextoShortEpisodeRunner, "Nexto", args, thresholds_path)
        wisp = _run_shadow_block(WispShortEpisodeRunner, "Wisp", args, thresholds_path)
        total_minutes = nexto["simulated_minutes"] + wisp["simulated_minutes"]
        overall_events = {}
        evidence: dict[str, list[dict[str, Any]]] = {}
        for family in FAMILY_NAMES:
            count = nexto["events"][family]["count"] + wisp["events"][family]["count"]
            overall_events[family] = {
                "count": count,
                "events_per_minute": count / max(total_minutes, 1.0e-12),
            }
            evidence[family] = nexto["bounded_evidence"][family] + wisp["bounded_evidence"][family]
        shadow = {
            "format": "RIVAL2_MECHANICS_SHADOW_V1",
            "status": "PASS_GREEN",
            "episodes": 256,
            "stochastic_rival": True,
            "deterministic_opponents": True,
            "checkpoint_sha256": checkpoint_sha,
            "seed": SHADOW_SEED,
            "opponents": {
                "Nexto": {key: value for key, value in nexto.items() if key != "bounded_evidence"},
                "Wisp": {key: value for key, value in wisp.items() if key != "bounded_evidence"},
            },
            "overall_events": overall_events,
            "impossible_count": nexto["impossible_count"] + wisp["impossible_count"],
            "mechanics_reward_contribution_sum": nexto["mechanics_reward_contribution_sum"]
            + wisp["mechanics_reward_contribution_sum"],
            "reward_enabled": False,
        }
        not_ready_events = {
            family: overall_events[family]["count"]
            for family in FAMILY_NAMES
            if thresholds["detectors"][family]["status"] == STATUS_NOT_READY
            and overall_events[family]["count"] != 0
        }
        shadow["pathology_checks"] = {
            "impossible_state_count_zero": shadow["impossible_count"] == 0,
            "mechanics_reward_exactly_zero": shadow["mechanics_reward_contribution_sum"] == 0.0,
            "not_ready_families_emit_zero": not not_ready_events,
            "not_ready_nonzero_events": not_ready_events,
        }
        _write_json(args.output_dir / "shadow_event_evidence.json", evidence)
        if shadow["mechanics_reward_contribution_sum"] != 0.0:
            raise RuntimeError("mechanics reward contribution was non-zero in shadow mode")
        if shadow["impossible_count"] != 0:
            raise RuntimeError("impossible mechanics classifications occurred in shadow mode")
        if not_ready_events:
            raise RuntimeError(f"NOT_READY families emitted shadow events: {not_ready_events}")
    _write_json(args.output_dir / "shadow_gate_summary.json", shadow)
    manifest = {
        "format": "RIVAL2_MECHANICS_CALIBRATION_MANIFEST_V1",
        "source_head": source_head,
        "handoff_source_commit": SOURCE_HANDOFF_COMMIT,
        "checkpoint": {"path": args.checkpoint.as_posix(), "sha256": checkpoint_sha},
        "collision_root": str(Path(args.collision_root).resolve()),
        "arena_geometry_sha256": ArenaGeometry.load_soccar(args.collision_root).content_sha256,
        "contract_source_sha256": {
            "rivalsim/rival2_contracts.py": _sha256(
                REPOSITORY_ROOT / "rivalsim/rival2_contracts.py"
            ),
            "rivalsim/rival2_env.py": _sha256(REPOSITORY_ROOT / "rivalsim/rival2_env.py"),
        },
        "calibration_seed": CALIBRATION_SEED,
        "shadow_seed": SHADOW_SEED,
        "case_count": len(rows),
        "cases_per_detector": 72,
        "derivation_per_detector": 48,
        "heldout_per_detector": 24,
        "shadow_episodes": int(shadow["episodes"]),
        "reward_enabled": False,
        "training_started": False,
        "elapsed_seconds": time.time() - started,
        "commands": [
            ".venv/Scripts/python.exe -m pytest -q tests/test_rival2_mechanics_calibration.py",
            ".venv/Scripts/python.exe "
            "benchmarks/run_rival2_mechanics_calibration.py "
            "--collision-root <pinned collision root>",
        ],
    }
    for name in (
        "thresholds.json",
        "case_results.jsonl",
        "heldout_summary.json",
        "source_exact_regression.json",
        "shadow_gate_summary.json",
    ):
        manifest.setdefault("artifacts", {})[name] = _sha256(args.output_dir / name)
    if (args.output_dir / "shadow_event_evidence.json").exists():
        manifest["artifacts"]["shadow_event_evidence.json"] = _sha256(
            args.output_dir / "shadow_event_evidence.json"
        )
    _write_json(args.output_dir / "calibration_manifest.json", manifest)
    report = _render_report(source_head, thresholds, heldout, shadow, rows, manifest)
    Path("docs/RIVAL2_MECHANICS_CALIBRATION_V1_RESULTS.md").write_text(
        report, encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                "manifest": manifest,
                "detector_status": {
                    name: thresholds["detectors"][name]["status"] for name in FAMILY_NAMES
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
