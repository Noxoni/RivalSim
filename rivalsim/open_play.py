"""Kickoff-free open-play state capture, mirroring, and duel runtime.

This module is evaluation-only.  It does not alter Rival's training episode,
policy, reward, action, observation, or simulator contracts.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import warp as wp

from rivalsim.arena import ArenaGeometry, build_face_mesh_index
from rivalsim.ball_world_state import MAX_BALL_CONTACTS
from rivalsim.behavioral_telemetry import (
    GOAL_SCORING_PLANE_Y_UU,
    SURFACE_BACKBOARD,
    SURFACE_SIDE_WALL,
    _surface_category,
)
from rivalsim.kernels.integrated import update_integrated_broadphase_order
from rivalsim.lifecycle_state import HELD_FLOAT_FIELDS, HELD_INT_FIELDS
from rivalsim.rival2_contracts import ORANGE_PAD_REMAP, RIVAL2_REWARD_VERSION
from rivalsim.rival2_env import Rival2TensorBridge, Rival2WorldSim
from rivalsim.rival2_policy import (
    Rival2ActorCritic,
    Rival2PolicyConfig,
    deterministic_hybrid_action,
)
from third_party.nexto.adapter import NextoPolicyAdapter, NextoStateTensors

PHYSICS_HZ = 120
DUEL_LIMIT_TICKS = 60 * PHYSICS_HZ
RIVAL_CADENCE_TICKS = 4
NEXTO_CADENCE_TICKS = 8
DIRECTION_THRESHOLD_UU_PER_SECOND = 100.0
TOUCH_BACKWARD = 0
TOUCH_NEUTRAL = 1
TOUCH_FORWARD = 2
BALL_PROXY_RADIUS_BT = np.float32(1.9249999523162842)
COMMON_RESTORED_TICK = 1_000_000

_WORLD_MUTABLE_ARRAYS = (
    "boost_pad_cooldown",
    "boost_pad_previous_locked_car",
    "_dynamic_proxy_cell",
    "_dynamic_proxy_move_rank",
    "_dynamic_proxy_move_counter",
    "_pair_a_before_b",
)
_COMPONENTS = (
    "state",
    "vehicle",
    "ball_world",
    "car_ball",
    "car_ball_b",
    "car_car",
    "lifecycle",
    "rival2",
)


def world_array_paths(world: Rival2WorldSim) -> dict[str, wp.array]:
    """Return every per-world mutable array required for continuation."""

    arrays: dict[str, wp.array] = {}
    for name in _WORLD_MUTABLE_ARRAYS:
        arrays[f"world.{name}"] = getattr(world, name)
    for component_name in _COMPONENTS:
        component = getattr(world, component_name)
        for field_name, value in vars(component).items():
            if isinstance(value, wp.array):
                arrays[f"{component_name}.{field_name}"] = value
    for path, value in arrays.items():
        if int(np.prod(value.shape)) % world.num_envs != 0:
            raise RuntimeError(f"non-world-shaped continuation array: {path} {value.shape}")
    return dict(sorted(arrays.items()))


def _flat_torch(array: wp.array, worlds: int) -> torch.Tensor:
    return wp.to_torch(array).reshape(worlds, -1)


class DeviceContinuationBank:
    """One captured continuation state per source world, retained on CUDA."""

    def __init__(self, world: Rival2WorldSim):
        self.worlds = world.num_envs
        self.device = torch.device(world.device)
        self.paths = world_array_paths(world)
        self.values = {
            path: torch.empty_like(_flat_torch(array, self.worlds))
            for path, array in self.paths.items()
        }
        self.captured = torch.zeros(self.worlds, dtype=torch.bool, device=self.device)
        self.capture_tick = torch.full(
            (self.worlds,), -1, dtype=torch.int32, device=self.device
        )

    def capture(self, world: Rival2WorldSim, eligible: torch.Tensor, tick: int) -> None:
        if eligible.shape != (self.worlds,) or eligible.dtype != torch.bool:
            raise ValueError("eligible mask shape/dtype mismatch")
        selected = eligible & ~self.captured
        sources = world_array_paths(world)
        for path, destination in self.values.items():
            source = _flat_torch(sources[path], self.worlds)
            destination.copy_(torch.where(selected[:, None], source, destination))
        self.capture_tick.copy_(
            torch.where(
                selected,
                torch.full_like(self.capture_tick, int(tick)),
                self.capture_tick,
            )
        )
        self.captured.logical_or_(selected)

    def complete_count(self) -> int:
        return int(self.captured.sum().item())

    def export(self) -> tuple[dict[str, np.ndarray], np.ndarray]:
        if self.complete_count() != self.worlds:
            raise RuntimeError("cannot export an incomplete continuation bank")
        values = {
            path: tensor.detach().cpu().numpy().copy()
            for path, tensor in self.values.items()
        }
        return values, self.capture_tick.detach().cpu().numpy().copy()


def build_face_mirror_maps(
    geometry: ArenaGeometry,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build exact 180-degree triangle/mesh maps from immutable CMF vertices."""

    triangles = geometry.vertices_uu[geometry.triangles]

    def key(value: np.ndarray) -> tuple[tuple[float, float, float], ...]:
        return tuple(sorted(tuple(float(item) for item in point) for point in value))

    lookup = {key(triangle): index for index, triangle in enumerate(triangles)}
    sign = np.asarray((-1.0, -1.0, 1.0), dtype=np.float32)
    face_map = np.asarray(
        [lookup.get(key(triangle * sign), -1) for triangle in triangles],
        dtype=np.int32,
    )
    if np.any(face_map < 0) or np.unique(face_map).size != geometry.triangle_count:
        raise RuntimeError("Soccar CMF faces do not form an exact 180-degree bijection")
    if not np.array_equal(face_map[face_map], np.arange(geometry.triangle_count)):
        raise RuntimeError("Soccar face mirror map is not involutive")
    face_mesh = build_face_mesh_index(geometry)
    mesh_map = np.full(20, -1, dtype=np.int32)
    for source_face, destination_face in enumerate(face_map):
        source_mesh = int(face_mesh[source_face])
        destination_mesh = int(face_mesh[destination_face])
        if mesh_map[source_mesh] not in (-1, destination_mesh):
            raise RuntimeError("one source CMF maps to multiple destination CMFs")
        mesh_map[source_mesh] = destination_mesh
    mesh_map[16:] = np.asarray((16, 17, 19, 18), dtype=np.int32)
    if np.any(mesh_map < 0) or not np.array_equal(mesh_map[mesh_map], np.arange(20)):
        raise RuntimeError("Soccar body mirror map is not involutive")
    evidence = {
        "triangle_count": geometry.triangle_count,
        "mapped_triangle_count": int((face_map >= 0).sum()),
        "bijection": True,
        "involutive": True,
        "mesh_map": mesh_map.tolist(),
        "face_map_sha256": hashlib.sha256(face_map.astype("<i4").tobytes()).hexdigest().upper(),
    }
    return face_map, mesh_map, evidence


def _rotate_vectors(value: np.ndarray) -> np.ndarray:
    result = value.reshape(value.shape[0], -1, 3).copy()
    result[..., 0] *= np.float32(-1.0)
    result[..., 1] *= np.float32(-1.0)
    return result.reshape(value.shape)


def _rotate_quaternions(value: np.ndarray) -> np.ndarray:
    source = value.reshape(value.shape[0], -1, 4)
    result = np.empty_like(source)
    # q_z(pi) * q in Warp's xyzw storage. Applying this twice yields -q,
    # the exactly equivalent quaternion representation of the same rotation.
    result[..., 0] = -source[..., 1]
    result[..., 1] = source[..., 0]
    result[..., 2] = source[..., 3]
    result[..., 3] = -source[..., 2]
    return result.reshape(value.shape)


def _swap_cars(value: np.ndarray, width_per_car: int) -> np.ndarray:
    reshaped = value.reshape(value.shape[0], 2, width_per_car)
    return reshaped[:, ::-1].copy().reshape(value.shape)


def _swap_rotate_car_vectors(value: np.ndarray, vectors_per_car: int = 1) -> np.ndarray:
    swapped = _swap_cars(value, vectors_per_car * 3)
    return _rotate_vectors(swapped)


def _swap_rotate_car_quaternions(value: np.ndarray) -> np.ndarray:
    return _rotate_quaternions(_swap_cars(value, 4))


def _map_car_ids(value: np.ndarray) -> np.ndarray:
    result = value.copy()
    result[value == 1] = 2
    result[value == 2] = 1
    return result


def _map_local_car_indices(value: np.ndarray) -> np.ndarray:
    result = value.copy()
    result[value == 0] = 1
    result[value == 1] = 0
    return result


def _map_faces(value: np.ndarray, face_map: np.ndarray) -> np.ndarray:
    result = value.copy()
    valid = (value >= 0) & (value < face_map.size)
    result[valid] = face_map[value[valid]]
    result[value == -12] = -13
    result[value == -13] = -12
    return result


def _map_meshes(value: np.ndarray, mesh_map: np.ndarray) -> np.ndarray:
    result = value.copy()
    valid = (value >= 0) & (value < mesh_map.size)
    result[valid] = mesh_map[value[valid]]
    return result


def _mirror_vehicle_field(
    field: str, value: np.ndarray, face_map: np.ndarray, mesh_map: np.ndarray
) -> np.ndarray:
    car_vectors = {
        "solver_position",
        "rigid_position_bt",
        "solver_velocity",
        "rigid_velocity_bt",
        "solver_angular_velocity",
        "auto_roll_acceleration",
        "auto_roll_angular_acceleration",
        "total_force_bt",
        "total_torque_bt",
        "world_contact_normal",
    }
    wheel_vectors = {
        "wheel_ray_start",
        "wheel_direction",
        "wheel_hit_point",
        "wheel_hit_point_bt",
        "wheel_hit_normal",
        "wheel_axle",
        "wheel_forward",
        "wheel_friction_impulse",
        "wheel_friction_impulse_bt",
        "wheel_friction_relative_bt",
        "debug_wheel_ray_from_bt",
        "debug_wheel_ray_to_bt",
        "debug_wheel_linear_bt",
        "debug_wheel_angular",
    }
    contact_vectors = {
        "contact_point",
        "contact_point_b",
        "contact_normal",
        "contact_tangent",
    }
    if field in car_vectors:
        return _swap_rotate_car_vectors(value)
    if field == "solver_orientation":
        return _swap_rotate_car_quaternions(value)
    if field == "inverse_inertia_world":
        matrix = _swap_cars(value, 9).reshape(value.shape[0], 2, 3, 3)
        sign = np.asarray((-1.0, -1.0, 1.0), dtype=np.float32)
        return (matrix * sign[None, None, :, None] * sign[None, None, None, :]).reshape(value.shape)
    if field in wheel_vectors:
        return _swap_rotate_car_vectors(value, 4)
    if field in contact_vectors:
        return _swap_rotate_car_vectors(value, 12)
    if field == "contact_local_a":
        return _swap_cars(value, 12 * 3)
    if field in {"contact_face", "mesh_candidate_face", "wheel_hit_face"}:
        width = {"contact_face": 12, "mesh_candidate_face": 128, "wheel_hit_face": 4}[field]
        return _map_faces(_swap_cars(value, width), face_map)
    if field == "contact_mesh":
        return _map_meshes(_swap_cars(value, 12), mesh_map)
    if field == "plane_support_direction":
        source = _swap_cars(value, 12).reshape(value.shape[0], 2, 4, 3)
        return source[:, :, [0, 1, 3, 2]].copy().reshape(value.shape)
    if value.shape[1] % 2 == 0:
        return _swap_cars(value, value.shape[1] // 2)
    return value.copy()


def _mirror_ball_world_field(
    field: str, value: np.ndarray, face_map: np.ndarray, mesh_map: np.ndarray
) -> np.ndarray:
    if field in {"position_bt", "velocity_bt", "contact_point_b_bt", "contact_normal", "contact_tangent"}:
        return _rotate_vectors(value)
    if field == "contact_local_a_bt":
        return value.copy()
    if field in {"contact_face", "candidate_face"}:
        return _map_faces(value, face_map)
    if field == "contact_mesh":
        return _map_meshes(value, mesh_map)
    return value.copy()


def _mirror_car_ball_field(field: str, value: np.ndarray) -> np.ndarray:
    global_vectors = {
        "contact_point_a_bt",
        "contact_point_b_bt",
        "contact_normal",
        "contact_tangent",
        "extra_hit_velocity_uu",
        "relative_pos_on_ball_uu",
        "pre_car_position_bt",
        "pre_car_velocity_bt",
        "pre_car_angular_velocity",
        "pre_ball_position_bt",
        "pre_ball_velocity_bt",
        "pre_ball_angular_velocity",
        "manifold_normal",
        "manifold_tangent",
    }
    if field in global_vectors:
        return _rotate_vectors(value)
    if field in {"pre_car_quaternion", "pre_ball_quaternion"}:
        return _rotate_quaternions(value)
    return value.copy()


def _mirror_car_car_field(
    field: str, value: np.ndarray, all_values: dict[str, np.ndarray]
) -> np.ndarray:
    car_vectors = {
        "pre_position_bt",
        "pre_velocity_bt",
        "pre_angular_velocity",
        "queued_velocity_bt",
    }
    if field in car_vectors:
        return _swap_rotate_car_vectors(value)
    if field == "pre_quaternion":
        return _swap_rotate_car_quaternions(value)
    if field in {"pre_on_ground", "pre_is_supersonic", "pre_supersonic_time", "car_contact_cooldown", "car_is_demoed"}:
        return _swap_cars(value, 1)
    if field == "car_contact_id":
        return _map_local_car_indices(_swap_cars(value, 1))
    if field == "pre_tick_first_car":
        return 1 - value
    if field in {"event_bumper", "event_victim"}:
        return _map_local_car_indices(value)
    if field == "manifold_local_a_bt":
        return all_values["car_car.manifold_local_b_bt"].copy()
    if field == "manifold_local_b_bt":
        return all_values["car_car.manifold_local_a_bt"].copy()
    if field in {"manifold_normal", "manifold_tangent", "contact_normal"}:
        return -_rotate_vectors(value)
    if field == "contact_point_b_bt":
        return _rotate_vectors(value)
    return value.copy()


def _mirror_lifecycle_field(field: str, value: np.ndarray) -> np.ndarray:
    pad_fields = {"pad_cooldown_before", "pad_pickup_car", "pad_reactivated"}
    car_fields = {
        "demo_respawn_timer",
        "demo_held_valid",
        "demo_request",
        "respawn_pending",
        "respawn_event",
        "respawn_location",
        "respawn_selector",
    }
    pad_map = np.asarray(ORANGE_PAD_REMAP, dtype=np.int64)
    if field in pad_fields:
        result = value[:, pad_map].copy()
        if field == "pad_pickup_car":
            result = _map_car_ids(result)
        return result
    if field in car_fields:
        result = _swap_cars(value, 1)
        if field in {"respawn_location", "respawn_selector"}:
            location_map = np.asarray((2, 3, 0, 1), dtype=np.int32)
            valid = result >= 0
            result[valid] = location_map[result[valid]]
        return result
    if field == "held_float":
        result = _swap_cars(value, HELD_FLOAT_FIELDS).reshape(
            value.shape[0], 2, HELD_FLOAT_FIELDS
        )
        for start in (0, 3, 10):
            result[..., start] *= np.float32(-1.0)
            result[..., start + 1] *= np.float32(-1.0)
        result[..., 6:10] = _rotate_quaternions(result[..., 6:10].reshape(value.shape[0], -1)).reshape(value.shape[0], 2, 4)
        return result.reshape(value.shape)
    if field == "held_int":
        return _swap_cars(value, HELD_INT_FIELDS)
    if field in {"blue_score", "orange_score"}:
        other = "orange_score" if field == "blue_score" else "blue_score"
        # The caller replaces this from the complete bank after dispatch.
        return value.copy() if other else value.copy()
    if field == "scoring_team":
        return _map_local_car_indices(value)
    return value.copy()


def mirror_continuation_bank(
    values: dict[str, np.ndarray],
    face_map: np.ndarray,
    mesh_map: np.ndarray,
) -> dict[str, np.ndarray]:
    """Apply the exact physical/team mirror to a captured host bank."""

    result: dict[str, np.ndarray] = {}
    pad_map = np.asarray(ORANGE_PAD_REMAP, dtype=np.int64)
    state_car_vectors = {"car_pos", "car_vel", "car_ang_vel"}
    state_car_quat = {"car_quat"}
    state_ball_vectors = {"ball_pos", "ball_vel", "ball_ang_vel"}
    for path, value in values.items():
        component, field = path.split(".", 1)
        if component == "world":
            if field in {"boost_pad_cooldown", "boost_pad_previous_locked_car"}:
                transformed = value[:, pad_map].copy()
                if field == "boost_pad_previous_locked_car":
                    transformed = _map_car_ids(transformed)
            elif field == "_dynamic_proxy_move_rank":
                transformed = value[:, [0, 2, 1]].copy()
            elif field == "_pair_a_before_b":
                transformed = 1 - value
            elif field == "_dynamic_proxy_cell":
                transformed = value[:, [0, 2, 1]].copy()
            else:
                transformed = value.copy()
        elif component == "state":
            if field in state_car_vectors:
                transformed = _swap_rotate_car_vectors(value)
            elif field in state_car_quat:
                transformed = _swap_rotate_car_quaternions(value)
            elif field == "flip_rel_torque":
                transformed = _swap_cars(value, 3)
            elif field in state_ball_vectors:
                transformed = _rotate_vectors(value)
            elif field == "ball_quat":
                transformed = _rotate_quaternions(value)
            elif value.shape[1] % 2 == 0 and field not in {"ball_pos", "ball_vel", "ball_ang_vel"}:
                transformed = _swap_cars(value, value.shape[1] // 2)
            else:
                transformed = value.copy()
        elif component == "vehicle":
            transformed = _mirror_vehicle_field(field, value, face_map, mesh_map)
        elif component == "ball_world":
            transformed = _mirror_ball_world_field(field, value, face_map, mesh_map)
        elif component in {"car_ball", "car_ball_b"}:
            other = "car_ball_b" if component == "car_ball" else "car_ball"
            transformed = _mirror_car_ball_field(field, values[f"{other}.{field}"])
        elif component == "car_car":
            transformed = _mirror_car_car_field(field, value, values)
        elif component == "lifecycle":
            transformed = _mirror_lifecycle_field(field, value)
        elif component == "rival2":
            if field in {"touch_count", "touch_contact_latched", "demo_by_count", "demoed_event", "reward"}:
                transformed = _swap_cars(value, value.shape[1] // 2)
            elif field == "previous_action":
                transformed = _swap_cars(value, 8)
            elif field in {"ball_y_before", "ball_y_after"}:
                transformed = -value
            elif field == "scoring_team_latched":
                transformed = _map_local_car_indices(value)
            else:
                transformed = value.copy()
        else:
            raise RuntimeError(f"unhandled continuation component: {component}")
        result[path] = transformed

    result["lifecycle.blue_score"] = values["lifecycle.orange_score"].copy()
    result["lifecycle.orange_score"] = values["lifecycle.blue_score"].copy()
    pre_ball = _rotate_vectors(values["car_ball.pre_ball_position_bt"])
    proxy = pre_ball.reshape(pre_ball.shape[0], 3).copy()
    proxy -= BALL_PROXY_RADIUS_BT
    result["ball_world.broadphase_proxy_min_bt"] = proxy.reshape(
        result["ball_world.broadphase_proxy_min_bt"].shape
    )
    return result


def mirror_involution_report(
    values: dict[str, np.ndarray], face_map: np.ndarray, mesh_map: np.ndarray
) -> dict[str, Any]:
    once = mirror_continuation_bank(values, face_map, mesh_map)
    twice = mirror_continuation_bank(once, face_map, mesh_map)
    failed: list[str] = []
    maximum = 0.0
    for path, source in values.items():
        actual = twice[path]
        if path in {
            "state.car_quat",
            "state.ball_quat",
            "vehicle.solver_orientation",
            "car_ball.pre_car_quaternion",
            "car_ball.pre_ball_quaternion",
            "car_ball_b.pre_car_quaternion",
            "car_ball_b.pre_ball_quaternion",
            "car_car.pre_quaternion",
        }:
            shaped_source = source.reshape(source.shape[0], -1, 4)
            shaped_actual = actual.reshape(actual.shape[0], -1, 4)
            direct = np.max(np.abs(shaped_actual - shaped_source), axis=-1)
            negated = np.max(np.abs(shaped_actual + shaped_source), axis=-1)
            error = float(np.max(np.minimum(direct, negated)))
        elif path == "lifecycle.held_float":
            # The packed demolition snapshot embeds the car quaternion at
            # columns 6:10.  q and -q are the identical stored rotation, just
            # as in the unpacked quaternion fields above.  All other packed
            # scalars must remain numerically exact.
            shaped_source = source.reshape(source.shape[0], 2, HELD_FLOAT_FIELDS)
            shaped_actual = actual.reshape(actual.shape[0], 2, HELD_FLOAT_FIELDS)
            scalar_columns = np.r_[0:6, 10:HELD_FLOAT_FIELDS]
            scalar_error = float(
                np.max(
                    np.abs(
                        shaped_actual[..., scalar_columns]
                        - shaped_source[..., scalar_columns]
                    ),
                    initial=0.0,
                )
            )
            source_quaternion = shaped_source[..., 6:10]
            actual_quaternion = shaped_actual[..., 6:10]
            direct = np.max(np.abs(actual_quaternion - source_quaternion), axis=-1)
            negated = np.max(np.abs(actual_quaternion + source_quaternion), axis=-1)
            error = max(scalar_error, float(np.max(np.minimum(direct, negated))))
        elif np.issubdtype(source.dtype, np.floating):
            error = float(np.max(np.abs(actual - source), initial=0.0))
        else:
            error = 0.0 if np.array_equal(actual, source) else float("inf")
        maximum = max(maximum, error)
        if not np.isfinite(error) or error > 1e-6:
            failed.append(path)
    return {
        "fields_checked": len(values),
        "maximum_numeric_error_quaternion_sign_equivalent": maximum,
        "failed_fields": failed,
        "pass": not failed,
    }


def _copy_host_rows(array: wp.array, rows: np.ndarray, worlds: int) -> None:
    target = _flat_torch(array, worlds)
    source = torch.from_numpy(np.ascontiguousarray(rows)).to(
        device=target.device, dtype=target.dtype
    )
    target.copy_(source)


def _neutralize_duel_boundary(world: Rival2WorldSim) -> None:
    """Open a non-kickoff decision boundary without changing physical timers."""

    views = {
        path: _flat_torch(array, world.num_envs)
        for path, array in world_array_paths(world).items()
    }
    for path in (
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
        "lifecycle.goal_scored",
        "lifecycle.kickoff_reset",
        "lifecycle.full_reset",
        "lifecycle.reset_required",
        "lifecycle.terminated",
        "lifecycle.truncated",
        "lifecycle.ball_scored_last",
    ):
        views[path].zero_()
    views["rival2.scoring_team_latched"].fill_(-1)
    views["lifecycle.scoring_team"].fill_(-1)
    views["lifecycle.auto_kickoff"].zero_()
    ball_y = views["state.ball_pos"][:, 1:2]
    views["rival2.ball_y_before"].copy_(ball_y)
    views["rival2.ball_y_after"].copy_(ball_y)


def restore_four_way_duels(
    world: Rival2WorldSim,
    values: dict[str, np.ndarray],
    capture_tick: np.ndarray,
    face_map: np.ndarray,
    mesh_map: np.ndarray,
    *,
    neutral_policy_memory: bool = True,
    common_tick: int = COMMON_RESTORED_TICK,
) -> dict[str, Any]:
    """Restore original/original/mirror/mirror rows into one 4x world batch."""

    base_worlds = int(capture_tick.size)
    if world.num_envs != base_worlds * 4:
        raise ValueError("duel world count must be four times the state-bank size")
    mirrored = mirror_continuation_bank(values, face_map, mesh_map)
    destinations = world_array_paths(world)
    for path, destination in destinations.items():
        original = values[path]
        mirror = mirrored[path]
        rows = np.empty((base_worlds, 4, original.shape[1]), dtype=original.dtype)
        rows[:, 0] = original
        rows[:, 1] = original
        rows[:, 2] = mirror
        rows[:, 3] = mirror
        flat = rows.reshape(world.num_envs, -1)
        if path in {
            "car_ball.last_extra_impulse_tick",
            "car_ball_b.last_extra_impulse_tick",
        }:
            source_ticks = np.repeat(capture_tick[:, None], 4, axis=1).reshape(-1, 1)
            valid = flat >= 0
            flat = flat.copy()
            flat[valid] += np.broadcast_to(
                np.int32(common_tick) - source_ticks, flat.shape
            )[valid]
        _copy_host_rows(destination, flat, world.num_envs)

    tick_view = wp.to_torch(world.tick_counter)
    tick_view.fill_(int(common_tick))
    world.tick_count = int(common_tick)

    # Rebuild only the cached dynamic-cell IDs from the mirrored pre-tick
    # transforms using the exact source kernel. Preserve mapped insertion ranks
    # and counters, which encode the container/broadphase lifecycle history.
    rank = wp.to_torch(world._dynamic_proxy_move_rank).clone()
    counter = wp.to_torch(world._dynamic_proxy_move_counter).clone()
    wp.to_torch(world._dynamic_proxy_cell).fill_(-1)
    pair = world.car_car
    pair_b = world.car_ball_b
    wp.launch(
        update_integrated_broadphase_order,
        dim=world.num_envs,
        inputs=[
            world.tick_counter,
            pair.pre_position_bt,
            pair.pre_velocity_bt,
            pair.pre_quaternion,
            pair.pre_angular_velocity,
            pair_b.pre_ball_position_bt,
            pair_b.pre_ball_velocity_bt,
            pair_b.pre_ball_quaternion,
            pair_b.pre_ball_angular_velocity,
            world._dynamic_proxy_cell,
            world._dynamic_proxy_move_rank,
            world._dynamic_proxy_move_counter,
            world._pair_a_before_b,
        ],
        device=world.device,
    )
    wp.to_torch(world._dynamic_proxy_move_rank).copy_(rank)
    wp.to_torch(world._dynamic_proxy_move_counter).copy_(counter)
    rank_rows = rank.reshape(world.num_envs, 3)
    wp.to_torch(world._pair_a_before_b).copy_(
        (rank_rows[:, 1] < rank_rows[:, 2]).to(torch.int32)
    )
    if neutral_policy_memory:
        _neutralize_duel_boundary(world)
    wp.synchronize_device(world.device)
    return {
        "base_states": base_worlds,
        "duel_worlds": world.num_envs,
        "common_rebased_tick": common_tick,
        "neutral_previous_action": neutral_policy_memory,
        "dynamic_proxy_cells_recomputed_by_source_kernel": True,
    }


@wp.func
def _open_direction_category(value: float) -> int:
    result = TOUCH_NEUTRAL
    if value >= DIRECTION_THRESHOLD_UU_PER_SECOND:
        result = TOUCH_FORWARD
    elif value <= -DIRECTION_THRESHOLD_UU_PER_SECOND:
        result = TOUCH_BACKWARD
    return result


@wp.func
def _open_finalize_possession(
    env: int,
    position: wp.vec3,
    last_toucher: wp.array(dtype=wp.int32),
    touch_start_position: wp.array(dtype=wp.vec3),
    active_surface_bits: wp.array(dtype=wp.int32),
    displacement_count: wp.array(dtype=wp.int32),
    wall_continuation_count: wp.array(dtype=wp.int32),
    backboard_continuation_count: wp.array(dtype=wp.int32),
):
    toucher = last_toucher[env]
    if toucher >= 0:
        sign = 1.0
        if toucher == 1:
            sign = -1.0
        net_y = sign * (position[1] - touch_start_position[env][1])
        category = _open_direction_category(net_y)
        car = env * 2 + toucher
        displacement_count[car * 3 + category] = displacement_count[car * 3 + category] + 1
        bits = active_surface_bits[env]
        side_wall_bit = wp.int32(1 << (SURFACE_SIDE_WALL - 1))
        backboard_bit = wp.int32(1 << (SURFACE_BACKBOARD - 1))
        if (bits & side_wall_bit) != 0:
            wall_continuation_count[car] = wall_continuation_count[car] + 1
        if (bits & backboard_bit) != 0:
            backboard_continuation_count[car] = backboard_continuation_count[car] + 1


@wp.kernel(enable_backward=False)
def collect_open_play_tick(
    ball_position: wp.array(dtype=wp.vec3),
    ball_velocity: wp.array(dtype=wp.vec3),
    car_a_hit_this_tick: wp.array(dtype=wp.int32),
    car_b_hit_this_tick: wp.array(dtype=wp.int32),
    car_a_pre_ball_position_bt: wp.array(dtype=wp.vec3),
    car_b_pre_ball_position_bt: wp.array(dtype=wp.vec3),
    car_a_pre_ball_velocity_bt: wp.array(dtype=wp.vec3),
    car_b_pre_ball_velocity_bt: wp.array(dtype=wp.vec3),
    pre_tick_first_car: wp.array(dtype=wp.int32),
    ball_contact_count: wp.array(dtype=wp.int32),
    ball_contact_normal: wp.array(dtype=wp.vec3),
    goal_scored: wp.array(dtype=wp.int32),
    scoring_team: wp.array(dtype=wp.int32),
    kickoff_reset: wp.array(dtype=wp.int32),
    full_reset: wp.array(dtype=wp.int32),
    reset_required: wp.array(dtype=wp.int32),
    bump_event_count: wp.array(dtype=wp.int32),
    bump_event_bumper: wp.array(dtype=wp.int32),
    bump_event_is_demo: wp.array(dtype=wp.int32),
    done: wp.array(dtype=wp.int32),
    winner: wp.array(dtype=wp.int32),
    elapsed_ticks: wp.array(dtype=wp.int32),
    goal_tick: wp.array(dtype=wp.int32),
    first_toucher: wp.array(dtype=wp.int32),
    last_toucher: wp.array(dtype=wp.int32),
    last_touch_tick: wp.array(dtype=wp.int32),
    final_touch_to_goal_ticks: wp.array(dtype=wp.int32),
    scorer_matches_last_toucher: wp.array(dtype=wp.int32),
    touch_contact_latched: wp.array(dtype=wp.int32),
    previous_ball_position: wp.array(dtype=wp.vec3),
    touch_start_position: wp.array(dtype=wp.vec3),
    active_surface_bits: wp.array(dtype=wp.int32),
    touch_count: wp.array(dtype=wp.int32),
    possession_total: wp.array(dtype=wp.int32),
    possession_same: wp.array(dtype=wp.int32),
    possession_opponent: wp.array(dtype=wp.int32),
    direction_count: wp.array(dtype=wp.int32),
    displacement_count: wp.array(dtype=wp.int32),
    wall_continuation_count: wp.array(dtype=wp.int32),
    backboard_continuation_count: wp.array(dtype=wp.int32),
    demo_count: wp.array(dtype=wp.int32),
    goal_entry_valid: wp.array(dtype=wp.int32),
    goal_entry_x: wp.array(dtype=wp.float32),
    goal_entry_z: wp.array(dtype=wp.float32),
    kickoff_event_count: wp.array(dtype=wp.int32),
    reset_event_count: wp.array(dtype=wp.int32),
):
    env = wp.tid()
    if done[env] != 0:
        return
    if kickoff_reset[env] != 0:
        kickoff_event_count[env] = kickoff_event_count[env] + 1
    # ``reset_required`` is only the post-goal request bit.  The open-play
    # runtime intentionally never services it.  Count actual reset execution,
    # represented by ``full_reset``/``kickoff_reset``, not the inert request.
    if full_reset[env] != 0:
        reset_event_count[env] = reset_event_count[env] + 1

    position_after = ball_position[env]
    velocity_after = ball_velocity[env]
    car_base = env * 2
    reports_a = wp.int32(car_a_hit_this_tick[env] != 0)
    reports_b = wp.int32(car_b_hit_this_tick[env] != 0)
    touched_a = wp.int32(reports_a != 0 and touch_contact_latched[car_base] == 0)
    touched_b = wp.int32(reports_b != 0 and touch_contact_latched[car_base + 1] == 0)
    touch_contact_latched[car_base] = reports_a
    touch_contact_latched[car_base + 1] = reports_b

    first = pre_tick_first_car[env]
    for ordinal in range(2):
        local_toucher = first
        if ordinal == 1:
            local_toucher = 1 - first
        accepted = touched_a
        position_before = car_a_pre_ball_position_bt[env] * 50.0
        velocity_before = car_a_pre_ball_velocity_bt[env] * 50.0
        if local_toucher == 1:
            accepted = touched_b
            position_before = car_b_pre_ball_position_bt[env] * 50.0
            velocity_before = car_b_pre_ball_velocity_bt[env] * 50.0
        if accepted != 0:
            previous_toucher = last_toucher[env]
            if previous_toucher >= 0:
                _open_finalize_possession(
                    env,
                    position_before,
                    last_toucher,
                    touch_start_position,
                    active_surface_bits,
                    displacement_count,
                    wall_continuation_count,
                    backboard_continuation_count,
                )
                previous_car = car_base + previous_toucher
                possession_total[previous_car] = possession_total[previous_car] + 1
                if previous_toucher == local_toucher:
                    possession_same[previous_car] = possession_same[previous_car] + 1
                else:
                    possession_opponent[previous_car] = possession_opponent[previous_car] + 1
            car = car_base + local_toucher
            touch_count[car] = touch_count[car] + 1
            if first_toucher[env] < 0:
                first_toucher[env] = local_toucher
            sign = 1.0
            if local_toucher == 1:
                sign = -1.0
            category = _open_direction_category(
                sign * (velocity_after[1] - velocity_before[1])
            )
            direction_count[car * 3 + category] = direction_count[car * 3 + category] + 1
            last_toucher[env] = local_toucher
            last_touch_tick[env] = elapsed_ticks[env]
            touch_start_position[env] = position_after
            active_surface_bits[env] = 0

    current_toucher = last_toucher[env]
    if current_toucher >= 0:
        contacts = ball_contact_count[env]
        contact_base = env * MAX_BALL_CONTACTS
        for relative in range(MAX_BALL_CONTACTS):
            if relative < contacts:
                surface = _surface_category(ball_contact_normal[contact_base + relative])
                active_surface_bits[env] = active_surface_bits[env] | wp.int32(1 << (surface - 1))

    event_base = env * 4
    events = bump_event_count[env]
    for relative in range(4):
        if relative < events:
            event = event_base + relative
            if bump_event_is_demo[event] != 0:
                bumper = bump_event_bumper[event]
                if bumper >= 0 and bumper < 2:
                    demo_count[car_base + bumper] = demo_count[car_base + bumper] + 1

    if goal_scored[env] != 0:
        _open_finalize_possession(
            env,
            position_after,
            last_toucher,
            touch_start_position,
            active_surface_bits,
            displacement_count,
            wall_continuation_count,
            backboard_continuation_count,
        )
        scorer = scoring_team[env]
        done[env] = 1
        winner[env] = scorer
        goal_tick[env] = elapsed_ticks[env] + 1
        final_toucher = last_toucher[env]
        if final_toucher >= 0:
            final_touch_to_goal_ticks[env] = elapsed_ticks[env] - last_touch_tick[env]
            scorer_matches_last_toucher[env] = wp.int32(final_toucher == scorer)
        scoring_sign = 1.0
        if scorer == 1:
            scoring_sign = -1.0
        scoring_plane = scoring_sign * GOAL_SCORING_PLANE_Y_UU
        before = previous_ball_position[env]
        delta_y = position_after[1] - before[1]
        if wp.abs(delta_y) > 0.000001:
            fraction = (scoring_plane - before[1]) / delta_y
            if fraction >= 0.0 and fraction <= 1.0:
                crossing = before + (position_after - before) * fraction
                goal_entry_valid[env] = 1
                goal_entry_x[env] = scoring_sign * crossing[0]
                goal_entry_z[env] = crossing[2]
    elapsed_ticks[env] = elapsed_ticks[env] + 1
    previous_ball_position[env] = position_after


class OpenPlayTelemetry:
    _WORLD_INT_FIELDS = (
        "done",
        "winner",
        "elapsed_ticks",
        "goal_tick",
        "first_toucher",
        "last_toucher",
        "last_touch_tick",
        "final_touch_to_goal_ticks",
        "scorer_matches_last_toucher",
        "goal_entry_valid",
        "kickoff_event_count",
        "reset_event_count",
    )
    _CAR_FIELDS = (
        "touch_count",
        "possession_total",
        "possession_same",
        "possession_opponent",
        "wall_continuation_count",
        "backboard_continuation_count",
        "demo_count",
    )

    def __init__(self, world: Rival2WorldSim):
        self.num_worlds = world.num_envs
        self.device = world.device
        for name in self._WORLD_INT_FIELDS:
            initial = -1 if name in {
                "winner",
                "goal_tick",
                "first_toucher",
                "last_toucher",
                "last_touch_tick",
                "final_touch_to_goal_ticks",
            } else 0
            setattr(self, name, wp.full(self.num_worlds, initial, dtype=wp.int32, device=self.device))
        car_count = self.num_worlds * 2
        self.touch_contact_latched = wp.zeros(car_count, dtype=wp.int32, device=self.device)
        self.previous_ball_position = wp.empty(self.num_worlds, dtype=wp.vec3, device=self.device)
        wp.copy(self.previous_ball_position, world.state.ball_pos)
        self.touch_start_position = wp.zeros(self.num_worlds, dtype=wp.vec3, device=self.device)
        self.active_surface_bits = wp.zeros(self.num_worlds, dtype=wp.int32, device=self.device)
        for name in self._CAR_FIELDS:
            setattr(self, name, wp.zeros(car_count, dtype=wp.int32, device=self.device))
        self.direction_count = wp.zeros(car_count * 3, dtype=wp.int32, device=self.device)
        self.displacement_count = wp.zeros(car_count * 3, dtype=wp.int32, device=self.device)
        self.goal_entry_x = wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)
        self.goal_entry_z = wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)
        self._original_launch: Any | None = None

    def attach(self, world: Rival2WorldSim) -> None:
        original = world._launch_tick
        self._original_launch = original

        def instrumented() -> None:
            original()
            self._launch(world)

        world._launch_tick = instrumented

    def _launch(self, world: Rival2WorldSim) -> None:
        wp.launch(
            collect_open_play_tick,
            dim=self.num_worlds,
            inputs=[
                world.state.ball_pos,
                world.state.ball_vel,
                world.car_ball.hit_this_tick,
                world.car_ball_b.hit_this_tick,
                world.car_ball.pre_ball_position_bt,
                world.car_ball_b.pre_ball_position_bt,
                world.car_ball.pre_ball_velocity_bt,
                world.car_ball_b.pre_ball_velocity_bt,
                world.car_car.pre_tick_first_car,
                world.ball_world.contact_count,
                world.ball_world.contact_normal,
                world.lifecycle.goal_scored,
                world.lifecycle.scoring_team,
                world.lifecycle.kickoff_reset,
                world.lifecycle.full_reset,
                world.lifecycle.reset_required,
                world.car_car.event_count,
                world.car_car.event_bumper,
                world.car_car.event_is_demo,
                *[getattr(self, name) for name in self._WORLD_INT_FIELDS[:9]],
                self.touch_contact_latched,
                self.previous_ball_position,
                self.touch_start_position,
                self.active_surface_bits,
                *[getattr(self, name) for name in self._CAR_FIELDS[:4]],
                self.direction_count,
                self.displacement_count,
                *[getattr(self, name) for name in self._CAR_FIELDS[4:]],
                self.goal_entry_valid,
                self.goal_entry_x,
                self.goal_entry_z,
                self.kickoff_event_count,
                self.reset_event_count,
            ],
            device=self.device,
        )

    def numpy(self) -> dict[str, np.ndarray]:
        wp.synchronize_device(self.device)
        result: dict[str, np.ndarray] = {}
        for name in self._WORLD_INT_FIELDS:
            result[name] = np.asarray(getattr(self, name).numpy(), dtype=np.int32)
        for name in self._CAR_FIELDS:
            result[name] = np.asarray(getattr(self, name).numpy(), dtype=np.int32).reshape(self.num_worlds, 2)
        result["direction_count"] = np.asarray(self.direction_count.numpy(), dtype=np.int32).reshape(self.num_worlds, 2, 3)
        result["displacement_count"] = np.asarray(self.displacement_count.numpy(), dtype=np.int32).reshape(self.num_worlds, 2, 3)
        result["goal_entry_x"] = np.asarray(self.goal_entry_x.numpy(), dtype=np.float32)
        result["goal_entry_z"] = np.asarray(self.goal_entry_z.numpy(), dtype=np.float32)
        return result


@dataclass(frozen=True, slots=True)
class OpenPlayTiming:
    physics_ticks: int
    seconds: float
    world_ticks_per_second: float


class OpenPlayDuelRunner:
    """Deterministic mixed-cadence first-goal evaluator with no resets."""

    def __init__(
        self,
        collision_root: str,
        checkpoint_path: str | Path,
        values: dict[str, np.ndarray],
        capture_tick: np.ndarray,
        face_map: np.ndarray,
        mesh_map: np.ndarray,
        *,
        evaluation_seed: int,
        device: str = "cuda:0",
    ):
        self.base_worlds = int(capture_tick.size)
        self.num_worlds = self.base_worlds * 4
        self.device = torch.device(device)
        kickoff_selector = np.zeros(self.num_worlds, dtype=np.int32)
        self.world = Rival2WorldSim(
            self.num_worlds,
            collision_root,
            device=device,
            seed=evaluation_seed,
            kickoff_selector=kickoff_selector,
            car_lifecycle_seed=evaluation_seed,
        )
        # Own one stream for the complete mixed Warp/PyTorch hot path.  This
        # also keeps graph capture independent of any short-lived source-
        # harvest stream wrapper.
        self.warp_stream = wp.Stream(self.world.device)
        self.torch_stream = wp.stream_to_torch(self.warp_stream)
        self._activate_stream()
        self.restore_report = restore_four_way_duels(
            self.world,
            values,
            capture_tick,
            face_map,
            mesh_map,
            neutral_policy_memory=True,
        )
        self.bridge = Rival2TensorBridge(self.world)
        self.telemetry = OpenPlayTelemetry(self.world)
        self.telemetry.attach(self.world)

        checkpoint_path = Path(checkpoint_path)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest().upper()
        if payload.get("format") != "RIVAL2_CHECKPOINT_V1":
            raise RuntimeError("unsupported Rival checkpoint format")
        if payload.get("reward_version") != RIVAL2_REWARD_VERSION:
            raise RuntimeError("open-play runner requires final Reward V1 checkpoint")
        policy_config = Rival2PolicyConfig(**payload["policy_config"])
        if policy_config.content_hash != payload["policy_config_hash"]:
            raise RuntimeError("Rival checkpoint policy contract mismatch")
        self.rival_policy = Rival2ActorCritic(policy_config).to(self.device)
        self.rival_policy.load_state_dict(payload["model"])
        self.rival_policy.eval()
        self.checkpoint_identity = {
            "path": checkpoint_path.as_posix(),
            "sha256": checkpoint_sha,
            "size_bytes": checkpoint_path.stat().st_size,
            "iteration": int(payload["iteration"]),
            "policy_version": int(payload["policy_version"]),
            "total_agent_samples": int(payload["total_agent_samples"]),
            "policy_config": asdict(policy_config),
            "policy_config_hash": policy_config.content_hash,
            "reward_version": payload["reward_version"],
        }
        del payload

        variants = torch.arange(self.num_worlds, device=self.device) % 4
        self.rival_side = torch.where(
            (variants == 0) | (variants == 2),
            torch.zeros_like(variants),
            torch.ones_like(variants),
        ).to(torch.long)
        self.nexto_side = 1 - self.rival_side
        self.batch_index = torch.arange(self.num_worlds, device=self.device)
        self.nexto = NextoPolicyAdapter(self.num_worlds, device=self.device)
        self.nexto.set_player_index(self.nexto_side)
        self.nexto_state = NextoStateTensors.from_bridge(self.bridge)
        self.no_kickoff = torch.zeros(self.num_worlds, dtype=torch.bool, device=self.device)
        self.rival_observation = self.bridge.observation()
        self.rival_action = torch.zeros((self.num_worlds, 8), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((self.num_worlds, 2, 8), dtype=torch.float32, device=self.device)
        self.host_tick = 0
        self.world.capture_graph(block_ticks=1)
        self.world.reset_transfer_counters()
        torch.cuda.reset_peak_memory_stats(self.device)

    def _activate_stream(self) -> None:
        torch.cuda.set_stream(self.torch_stream)
        wp.set_stream(self.warp_stream, device=self.world.device, sync=False)

    def tick(self) -> None:
        self._activate_stream()
        if self.host_tick % RIVAL_CADENCE_TICKS == 0:
            observation = self.rival_observation[self.batch_index, self.rival_side]
            with torch.inference_mode():
                actor, _value = self.rival_policy(observation)
                self.rival_action.copy_(deterministic_hybrid_action(actor))
            self.world.begin_decision()
        nexto_action, _indices = self.nexto.tick_action(self.nexto_state, self.no_kickoff)
        self.actions[self.batch_index, self.rival_side] = self.rival_action
        self.actions[self.batch_index, self.nexto_side] = nexto_action
        self.bridge.set_actions(self.actions)
        self.world.step_graph(1)
        self.host_tick += 1
        if self.host_tick % RIVAL_CADENCE_TICKS == 0:
            self.rival_observation = self.bridge.observation()

    def run_ticks(self, ticks: int) -> OpenPlayTiming:
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        for _ in range(ticks):
            self.tick()
        torch.cuda.synchronize(self.device)
        seconds = time.perf_counter() - started
        return OpenPlayTiming(ticks, seconds, self.num_worlds * ticks / seconds)

    def profile_ticks(self, ticks: int = 8) -> tuple[OpenPlayTiming, list[str]]:
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as profile:
            for _ in range(ticks):
                self.tick()
            torch.cuda.synchronize(self.device)
        seconds = time.perf_counter() - started
        transfers = [
            event.name
            for event in profile.events()
            if "memcpy htod" in event.name.lower() or "memcpy dtoh" in event.name.lower()
        ]
        return OpenPlayTiming(ticks, seconds, self.num_worlds * ticks / seconds), transfers

    def export(self) -> dict[str, Any]:
        return {
            "raw": self.telemetry.numpy(),
            "checkpoint": self.checkpoint_identity,
            "restore": self.restore_report,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(self.device)),
            "world_host_to_device_bytes_after_initialization": int(self.world.host_to_device_bytes),
            "world_device_to_host_bytes_before_export": int(self.world.device_to_host_bytes),
            "nexto_timed_h2d_bytes": int(self.nexto.timed_h2d_bytes),
            "nexto_timed_d2h_bytes": int(self.nexto.timed_d2h_bytes),
            "nexto_inference_calls": int(self.nexto.inference_calls),
        }


__all__ = [
    "COMMON_RESTORED_TICK",
    "DUEL_LIMIT_TICKS",
    "DeviceContinuationBank",
    "OpenPlayDuelRunner",
    "OpenPlayTelemetry",
    "OpenPlayTiming",
    "build_face_mirror_maps",
    "mirror_continuation_bank",
    "mirror_involution_report",
    "restore_four_way_duels",
    "world_array_paths",
]
