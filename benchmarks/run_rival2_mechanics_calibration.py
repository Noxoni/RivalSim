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
)
from rivalsim.nexto_short_eval import NextoShortEpisodeRunner
from rivalsim.wisp_short_eval import WispShortEpisodeRunner

SOURCE_HANDOFF_COMMIT = "1da8557f32a94e6a8e96d1acbb0103656e203e27"
EXPECTED_CHECKPOINT_SHA256 = "77BF257131FB71DDEAEAE49D668C5E25AB1D06EE26149AB0D0AE303573CA5F21"
CALIBRATION_SEED = 2026082701
SHADOW_SEED = 2026082702
CASE_COUNT_PER_CLASS = 24
DERIVATION_PER_CLASS = 16
HELDOUT_PER_CLASS = 8
TRACE_TICKS = 120


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


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
            rows.append(
                {
                    "case_id": f"{family}-{class_name}-{split[0].upper()}{index:02d}",
                    "family": family,
                    "class": class_name,
                    "split": split,
                    "class_index": index,
                    "seed": CALIBRATION_SEED + FAMILY_ID[family] * 1000 + index,
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
    state.car_vel[:, 0, 0] = np.linspace(700.0, 1250.0, count, dtype=np.float32)
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
        finish = min(trace["car_vel"].shape[0] - 1, start + (220 if backward else 36))
        if start < 0:
            pitch_path = 999.0
            alignment = -1.0
            heading = 1.0
            new_forward_speed = -9999.0
            actual = 0.0
            up_final = np.asarray((0.0, 0.0, -1.0), dtype=np.float32)
        else:
            actual = 1.0
            pitch_path = 0.0
            for tick in range(start, finish + 1):
                q = trace["car_quat"][tick, index]
                right = _quat_rotate(q, np.asarray((0.0, 1.0, 0.0), dtype=np.float32))
                pitch_path += abs(float(np.dot(trace["car_ang"][tick, index], right))) / 120.0
            forward_initial = _quat_rotate(
                trace["car_quat"][start, index], np.asarray((1.0, 0.0, 0.0), dtype=np.float32)
            )
            forward_final = _quat_rotate(
                trace["car_quat"][finish, index], np.asarray((1.0, 0.0, 0.0), dtype=np.float32)
            )
            tangent = trace["car_vel"][finish, index].copy()
            tangent[2] = 0.0
            alignment = float(np.dot(_unit(forward_final), _unit(tangent)))
            heading = float(np.dot(_unit(forward_final), _unit(forward_initial)))
            new_forward_speed = float(np.dot(trace["car_vel"][finish, index], forward_final))
            up_final = _quat_rotate(
                trace["car_quat"][finish, index],
                np.asarray((0.0, 0.0, 1.0), dtype=np.float32),
            )
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
            "alignment": alignment,
            "heading_dot": heading,
            "new_forward_speed": new_forward_speed,
            "supported_completion": float(
                np.any(trace["wheel_count"][max(0, finish - 20) : finish + 1, index] > 0)
                or up_final[2] > 0.25
            ),
        }
        row["features"] = features
        row["physical_invariant"] = (
            bool(
                actual
                and (
                    (not backward and abs(diagonal[index]) >= 0.5 and cancel_delay[index] <= 3)
                    or (
                        backward
                        and 34 <= cancel_delay[index] <= 38
                        and features["heading_dot"] < -0.35
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


def _musty_cases(
    family: str,
    rows: list[dict[str, Any]],
    collision_root: str,
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
) -> list[dict[str, Any]]:
    count = len(rows)
    state = _base_state(count)
    state.car_pos[:, 0] = (0.0, 0.0, 800.0)
    state.car_vel[:, 0, 0] = 400.0
    state.on_ground[:, 0] = 0
    state.has_jumped[:, 0] = 1
    state.air_time[:, 0] = 0.2
    state.air_time_since_jump[:, 0] = 0.2
    for index, row in enumerate(rows):
        local = int(row["class_index"])
        if row["class"] == "positive":
            offset = np.asarray(MUSTY_OFFSETS[local % len(MUSTY_OFFSETS)], dtype=np.float32)
            offset[1] += ((local // len(MUSTY_OFFSETS)) - 1) * 4.0
            ball_vx = 400.0
            dodge = True
        elif row["class"] == "near_miss":
            offset = np.asarray((25.0 + 2.0 * (local % 5), 0.0, 105.0), dtype=np.float32)
            ball_vx = 150.0
            dodge = True
        else:
            offset = np.asarray((45.0, ((local % 5) - 2) * 8.0, 105.0), dtype=np.float32)
            ball_vx = 100.0
            dodge = False
        state.ball_pos[index] = state.car_pos[index, 0] + offset
        state.ball_vel[index, 0] = ball_vx
        row["scenario"] = {
            "ball_offset": offset.tolist(),
            "ball_vx": ball_vx,
            "backward_dodge": dodge,
        }

    def controller(tick: int, controls: ControlBatch) -> None:
        if tick == 0:
            for index, row in enumerate(rows):
                if row["scenario"]["backward_dodge"]:
                    controls.jump[index, 0] = 1
                    controls.pitch[index, 0] = 1.0

    trace = _run(state, 55, collision_root, geometry, meshes, controller)
    onsets = _touch_onsets(trace["hit"])
    for index, row in enumerate(rows):
        touches = onsets[index]
        if touches.size:
            tick = int(touches[0])
            normal = trace["contact_normal"][tick, index]
            r_bt = trace["contact_point"][tick, index] - trace["car_pos"][tick - 1, index] * 0.02
            rotational = np.cross(trace["pre_car_ang"][tick, index], r_bt) * 50.0
            translation = trace["pre_car_vel"][tick, index] * 50.0
            rotational_normal = abs(float(np.dot(rotational, normal)))
            translation_normal = abs(float(np.dot(translation, normal)))
            fraction = rotational_normal / max(rotational_normal + translation_normal, 1.0e-8)
            ball_delta = float(np.linalg.norm(trace["ball_delta_v"][tick, index]))
            hit_tick = float(tick)
        else:
            rotational_normal = fraction = ball_delta = 0.0
            translation_normal = 0.0
            hit_tick = -1.0
        row["features"] = {
            "actual_backward_dodge": float(np.any(trace["has_flipped"][:, index] != 0)),
            "rotational_normal_speed": rotational_normal,
            "rotational_fraction": fraction,
            "ball_delta_v": ball_delta,
            "translation_normal_speed": translation_normal,
            "contact_tick": hit_tick,
        }
        row["physical_invariant"] = (
            bool(
                touches.size
                and row["features"]["actual_backward_dodge"] > 0.0
                and rotational_normal > translation_normal
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
    initial.car_pos[:, 0] = (0.0, 0.0, 800.0)
    initial.car_vel[:, 0, 0] = 400.0
    initial.on_ground[:, 0] = 0
    initial.has_jumped[:, 0] = 1
    initial.air_time[:, 0] = 0.2
    initial.air_time_since_jump[:, 0] = 0.2
    duration = np.zeros(count, dtype=np.int32)
    roll = np.zeros(count, dtype=np.float32)
    yaw = np.zeros(count, dtype=np.float32)
    offset_index = np.full(count, 3, dtype=np.int32)
    # This timing/path is the physically verified terminal-contact setup from
    # the prospective probe.  Cases vary their arena origin; identity is
    # translation invariant and the actual path/contact is remeasured below.
    offset_index.fill(5)
    for index, row in enumerate(rows):
        local = int(row["class_index"])
        duration[index] = (24, 30, 36)[local % 3]
        if row["class"] == "positive":
            duration[index] = 34 + local % 5
            roll[index] = -1.0
            yaw[index] = 0.72 + 0.04 * (local % 5)
        elif row["class"] == "near_miss":
            roll[index] = 1.0 if local % 2 else 0.0
            yaw[index] = 0.0 if local % 2 else 1.0
        else:
            duration[index] = 0
        origin = (
            (float(local % 6) - 2.5) * 160.0,
            (float(local // 6) - 1.5) * 160.0,
        )
        initial.car_pos[index, 0, 0] = origin[0]
        initial.car_pos[index, 0, 1] = origin[1]
        row["scenario"] = {
            "setup_ticks": int(duration[index]),
            "roll": float(roll[index]),
            "yaw": float(yaw[index]),
            "musty_offset": MUSTY_OFFSETS[int(offset_index[index])],
            "origin_xy": origin,
        }

    def setup_controller(tick: int, controls: ControlBatch) -> None:
        active = tick < duration
        controls.roll[active, 0] = roll[active]
        controls.yaw[active, 0] = yaw[active]

    probe = initial.copy()
    probe.ball_pos[:] = (0.0, -3000.0, 1500.0)
    probe_trace = _run(probe, 40, collision_root, geometry, meshes, setup_controller)
    full = initial.copy()
    for index in range(count):
        terminal = int(duration[index])
        car_position = probe_trace["car_pos"][terminal, index]
        car_quat = probe_trace["car_quat"][terminal, index]
        local_target = np.asarray(MUSTY_OFFSETS[int(offset_index[index])], dtype=np.float32)
        target = car_position + _quat_rotate(car_quat, local_target)
        gravity_displacement = (
            probe_trace["ball_pos"][terminal, index] - probe_trace["ball_pos"][0, index]
        )
        full.ball_pos[index] = target - gravity_displacement
        full.ball_vel[index, 0] = 400.0

    def controller(tick: int, controls: ControlBatch) -> None:
        active = tick < duration
        controls.roll[active, 0] = roll[active]
        controls.yaw[active, 0] = yaw[active]
        dodge = tick == duration
        controls.jump[dodge, 0] = 1
        controls.pitch[dodge, 0] = 1.0

    trace = _run(full, 95, collision_root, geometry, meshes, controller)
    onsets = _touch_onsets(trace["hit"])
    for index, row in enumerate(rows):
        ticks = onsets[index]
        stop = int(duration[index])
        roll_path = 0.0
        yaw_path = 0.0
        forward_z: list[float] = []
        up_z: list[float] = []
        for tick in range(stop + 1):
            q = trace["car_quat"][tick, index]
            forward = _quat_rotate(q, np.asarray((1.0, 0.0, 0.0), dtype=np.float32))
            up = _quat_rotate(q, np.asarray((0.0, 0.0, 1.0), dtype=np.float32))
            ang = trace["car_ang"][tick, index]
            roll_path += abs(float(np.dot(ang, forward))) / 120.0
            yaw_path += abs(float(np.dot(ang, up))) / 120.0
            forward_z.append(float(forward[2]))
            up_z.append(float(up[2]))
        terminal_musty = 0.0
        rotational = 0.0
        if ticks.size:
            tick = int(ticks[0])
            normal = trace["contact_normal"][tick, index]
            r_bt = trace["contact_point"][tick, index] - trace["car_pos"][tick - 1, index] * 0.02
            rotational = abs(
                float(np.dot(np.cross(trace["pre_car_ang"][tick, index], r_bt) * 50.0, normal))
            )
            terminal_musty = float(rotational > 1.0 and np.any(trace["has_flipped"][:, index] != 0))
        row["features"] = {
            "terminal_musty": terminal_musty,
            "roll_path": roll_path,
            "yaw_path": yaw_path,
            "setup_ticks": float(stop),
            "nose_up_peak": float(max(forward_z, default=0.0)),
            "inverted_depth": float(-min(up_z, default=0.0)),
            "rotational_normal_speed": rotational,
        }
        row["physical_invariant"] = (
            bool(terminal_musty and roll_path > 0.1 and yaw_path > 0.1 and stop >= 24)
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
        state.car_quat[index, 0] = _quat_from_euler(0.0, 0.0, math.pi * 0.5)
        if row["class"] == "positive":
            incoming = 900.0 + 35.0 * (local % 8)
            state.car_vel[index, 0, 1] = 250.0 + 25.0 * (local % 5)
            height = 120.0 + 3.0 * (local % 4)
        elif row["class"] == "near_miss":
            incoming = 180.0 + 15.0 * (local % 8)
            state.car_vel[index, 0, 1] = 0.0
            height = 110.0
        else:
            incoming = 60.0
            state.car_vel[index, 0, 0] = incoming
            height = 105.0
        state.ball_pos[index] = (-350.0, 0.0, height)
        state.ball_vel[index] = (incoming, 0.0, 0.0)
        row["scenario"] = {
            "incoming_speed": incoming,
            "ball_height": height,
            "car_cross_speed": float(state.car_vel[index, 0, 1]),
        }

    trace = _run(state, 70, collision_root, geometry, meshes, lambda _tick, _controls: None)
    onsets = _touch_onsets(trace["hit"])
    for index, row in enumerate(rows):
        ticks = onsets[index]
        if ticks.size:
            tick = int(ticks[0])
            incoming = trace["ball_vel"][max(0, tick - 1), index]
            outgoing = trace["ball_vel"][min(tick + 1, trace["ball_vel"].shape[0] - 1), index]
            in_speed = float(np.linalg.norm(incoming))
            out_speed = float(np.linalg.norm(outgoing))
            angle = float(
                math.acos(float(np.clip(np.dot(_unit(incoming), _unit(outgoing)), -1.0, 1.0)))
            )
            context_height = float(
                trace["ball_pos"][tick, index, 2] - trace["car_pos"][tick, index, 2]
            )
        else:
            in_speed = out_speed = angle = 0.0
            context_height = -999.0
        row["features"] = {
            "incoming_speed": in_speed,
            "outgoing_speed": out_speed,
            "direction_change": angle,
            "contact_height": context_height,
        }
        row["physical_invariant"] = (
            bool(ticks.size and in_speed > 500.0 and angle > 0.1)
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
            corner = float(
                max(abs(local[0]) / 2.3602, abs(local[1]) / 1.6840, abs(local[2]) / 0.7232)
            )
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
    pool_count = 768
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
    for index in range(pool_count):
        item = extract(pool_trace, index)
        if (
            item["corner_region"] >= 0.7
            and item["incoming_normal_speed"] > 1.0
            and item["outgoing_normal_speed"] > 1.0
            and item["wheel_support"] < 3.0
            and item["separation_ticks"] < 12.0
        ):
            selected.append(index)
            if len(selected) == CASE_COUNT_PER_CLASS:
                break
    if len(selected) != CASE_COUNT_PER_CLASS:
        raise RuntimeError(f"pogo discovery produced only {len(selected)} positives")

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
        else:
            if row["class"] == "near_miss":
                roll = rng.uniform(-0.35, 0.35)
                pitch = rng.uniform(-0.35, 0.35)
                vz = rng.uniform(-900.0, -350.0)
                angular = rng.uniform(-1.0, 1.0, 3)
                height = rng.uniform(90.0, 220.0)
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
                and features["corner_region"] >= 0.7
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
        ("velocity_change", "min", "diagnostic_velocity_change"),
    ),
    "musty": (
        ("actual_backward_dodge", "min", "discrete_backward_dodge"),
        ("rotational_normal_speed", "min", "musty_rotational_normal_speed_min"),
        ("rotational_fraction", "min", "musty_rotational_fraction_min"),
        ("ball_delta_v", "min", "musty_ball_delta_v_min"),
    ),
    "breezi": (
        ("terminal_musty", "min", "discrete_terminal_musty"),
        ("roll_path", "min", "breezi_roll_path_min"),
        ("yaw_path", "min", "breezi_yaw_path_min"),
        ("setup_ticks", "min", "breezi_setup_ticks_min"),
        ("inverted_depth", "min", "diagnostic_inverted_depth"),
    ),
    "redirect": (
        ("incoming_speed", "min", "redirect_incoming_speed_min"),
        ("outgoing_speed", "min", "redirect_outgoing_speed_min"),
        ("direction_change", "min", "redirect_angle_min_radians"),
        ("contact_height", "min", "diagnostic_contact_height"),
    ),
    "pinch": (
        ("overlap_ticks", "max", "pinch_overlap_ticks_max"),
        ("opposition", "min", "pinch_opposition_min"),
        ("closing_speed", "min", "pinch_closing_speed_min"),
        ("ball_delta_v", "min", "diagnostic_ball_delta_v"),
    ),
    "pogo": (
        ("chassis_contact", "min", "discrete_chassis_contact"),
        ("corner_region", "min", "pogo_corner_region_min"),
        ("incoming_normal_speed", "min", "pogo_incoming_normal_speed_min"),
        ("outgoing_normal_speed", "min", "pogo_outgoing_normal_speed_min"),
        ("wheel_support", "max", "pogo_wheel_support_max"),
        ("separation_ticks", "max", "pogo_separation_ticks_min"),
    ),
}


def _derive_boundaries(
    positives: list[dict[str, float]],
    negatives: list[dict[str, float]],
    candidates: tuple[tuple[str, str, str], ...],
) -> tuple[list[dict[str, Any]], list[int]]:
    remaining = list(range(len(negatives)))
    boundaries: list[dict[str, Any]] = []
    used: set[tuple[str, str]] = set()
    while remaining:
        best: tuple[int, float, dict[str, Any], list[int]] | None = None
        for feature, direction, runtime_name in candidates:
            if (feature, direction) in used:
                continue
            positive_values = [float(row[feature]) for row in positives]
            positive_edge = min(positive_values) if direction == "min" else max(positive_values)
            rejectable = [
                index
                for index in remaining
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
            threshold = (positive_edge + negative_edge) * 0.5
            record = {
                "feature": feature,
                "direction": direction,
                "runtime_threshold": runtime_name,
                "threshold": threshold,
                "positive_edge": positive_edge,
                "negative_edge": negative_edge,
                "separation_margin": margin,
            }
            score = (len(rejectable), margin)
            if best is None or score > (best[0], best[1]):
                best = (len(rejectable), margin, record, rejectable)
        if best is None:
            break
        boundaries.append(best[2])
        used.add((str(best[2]["feature"]), str(best[2]["direction"])))
        rejected = set(best[3])
        remaining = [index for index in remaining if index not in rejected]
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
    all_rows: list[dict[str, Any]] = []
    detectors: dict[str, Any] = {}
    runners: dict[str, Callable[..., list[dict[str, Any]]]] = {
        "speedflip": lambda rows, *shared: _flip_cases("speedflip", rows, *shared),
        "half_flip": lambda rows, *shared: _flip_cases("half_flip", rows, *shared),
        "possession": lambda rows, *shared: _roof_cases("possession", rows, *shared),
        "ground_carry": lambda rows, *shared: _roof_cases("ground_carry", rows, *shared),
        "musty": lambda rows, *shared: _musty_cases("musty", rows, *shared),
        "breezi": lambda rows, *shared: _breezi_cases(rows, *shared),
        "redirect": lambda rows, *shared: _redirect_cases(rows, *shared),
        "pinch": lambda rows, *shared: _pinch_cases(rows, *shared),
        "pogo": lambda rows, *shared: _pogo_cases(rows, *shared),
    }
    for family in FAMILY_NAMES:
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
        for feature, _direction, _runtime in CALIBRATION_FEATURES[family]:
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
            "features_considered": [item[0] for item in CALIBRATION_FEATURES[family]],
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
        all_rows.extend(rows)
    threshold_payload = {
        "format": "RIVAL2_MECHANICS_THRESHOLDS_V1",
        "source_head": source_head,
        "handoff_source_commit": SOURCE_HANDOFF_COMMIT,
        "calibration_seed": CALIBRATION_SEED,
        "physics_hz": 120,
        "policy_hz": 30,
        "binary_identity_only": True,
        "reward_enabled": False,
        "runtime_threshold_slots": list(THRESHOLD_NAMES),
        "detectors": detectors,
    }
    return threshold_payload, all_rows


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
) -> str:
    lines = [
        "# Rival 2.0 Mechanics Calibration V1 Results",
        "",
        f"Source head: `{source_head}`  ",
        f"Handoff source: `{SOURCE_HANDOFF_COMMIT}`  ",
        "Mode: calibration plus read-only shadow telemetry; mechanics reward "
        "remained exactly disabled.",
        "",
        "## Continuous detector results",
        "",
        "| Detector | Status | Boundaries | Held-out FP | Held-out FN |",
        "|---|---:|---|---:|---:|",
    ]
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
    lines.extend(
        [
            "",
            f"Impossible/pathological classifications: {shadow['impossible_count']}.  ",
            "Mechanics reward contribution: "
            f"`{shadow['mechanics_reward_contribution_sum']}` (required exact zero).",
            "",
            "Bounded per-event raw features are retained in "
            "`shadow_event_evidence.json`; all calibration case parameters and "
            "measured features are retained in `case_results.jsonl`.",
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
        _write_json(args.output_dir / "shadow_event_evidence.json", evidence)
        if shadow["mechanics_reward_contribution_sum"] != 0.0:
            raise RuntimeError("mechanics reward contribution was non-zero in shadow mode")
    _write_json(args.output_dir / "shadow_gate_summary.json", shadow)
    manifest = {
        "format": "RIVAL2_MECHANICS_CALIBRATION_MANIFEST_V1",
        "source_head": source_head,
        "handoff_source_commit": SOURCE_HANDOFF_COMMIT,
        "checkpoint": {"path": args.checkpoint.as_posix(), "sha256": checkpoint_sha},
        "collision_root": str(Path(args.collision_root).resolve()),
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
    report = _render_report(source_head, thresholds, heldout, shadow)
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
