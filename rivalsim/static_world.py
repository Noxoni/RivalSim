"""v0.2 shared-arena simulator with selectable B0/B1/B2/B3 layers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from rivalsim.arena import ArenaGeometry, WarpArenaMeshes
from rivalsim.kernels.boost_pad import PAD_COUNT, SOCCAR_PAD_POSITIONS, boost_pad_tick
from rivalsim.kernels.vehicle import (
    chassis_contacts_v021,
    increment_tick_counter,
    load_action_tape,
    wheel_pre_tick,
)
from rivalsim.simulator import RivalSim
from rivalsim.state import StateSnapshot
from rivalsim.vehicle_state import VehicleSnapshot, VehicleState

VARIANT_LEVEL = {"B0": 0, "B1": 1, "B2": 2, "B3": 3}


@dataclass(frozen=True, slots=True)
class ActionTape:
    throttle: np.ndarray
    steer: np.ndarray
    pitch: np.ndarray
    yaw: np.ndarray
    roll: np.ndarray
    jump: np.ndarray
    boost: np.ndarray
    handbrake: np.ndarray
    hold_ticks: int = 4

    @classmethod
    def deterministic(cls, length: int = 64, hold_ticks: int = 4) -> ActionTape:
        if length <= 0 or hold_ticks <= 0:
            raise ValueError("action tape length and hold_ticks must be positive")
        index = np.arange(length, dtype=np.float32)
        throttle = np.sin(index * np.float32(0.37)).astype(np.float32)
        steer = np.sin(index * np.float32(0.73) + np.float32(0.2)).astype(np.float32)
        pitch = (np.sin(index * np.float32(0.19)) * np.float32(0.35)).astype(np.float32)
        yaw = (np.cos(index * np.float32(0.23)) * np.float32(0.25)).astype(np.float32)
        roll = (np.sin(index * np.float32(0.29)) * np.float32(0.2)).astype(np.float32)
        jump = ((np.arange(length) % 29) == 0).astype(np.int32)
        boost = ((np.arange(length) % 11) < 3).astype(np.int32)
        handbrake = ((np.arange(length) % 17) < 4).astype(np.int32)
        return cls(throttle, steer, pitch, yaw, roll, jump, boost, handbrake, hold_ticks)

    @property
    def length(self) -> int:
        return len(self.throttle)

    @property
    def nbytes(self) -> int:
        return sum(
            value.nbytes
            for value in (
                self.throttle,
                self.steer,
                self.pitch,
                self.yaw,
                self.roll,
                self.jump,
                self.boost,
                self.handbrake,
            )
        )


class DeviceActionTape:
    def __init__(self, tape: ActionTape, device: str):
        self.length = tape.length
        self.hold_ticks = tape.hold_ticks
        for name in ("throttle", "steer", "pitch", "yaw", "roll"):
            setattr(self, name, wp.array(getattr(tape, name), dtype=wp.float32, device=device))
        for name in ("jump", "boost", "handbrake"):
            setattr(self, name, wp.array(getattr(tape, name), dtype=wp.int32, device=device))


class StaticWorldSim(RivalSim):
    """RivalSim v0.2 with one shared immutable arena mesh per device."""

    def __init__(
        self,
        num_envs: int,
        collision_root: str,
        *,
        variant: str = "B3",
        device: str = "cuda:0",
        seed: int = 0,
        initial: StateSnapshot | None = None,
        geometry: ArenaGeometry | None = None,
        meshes: WarpArenaMeshes | None = None,
        ray_backend: str = "cubql",
        action_tape: ActionTape | None = None,
    ):
        if variant not in VARIANT_LEVEL:
            raise ValueError(f"unknown benchmark variant: {variant}")
        super().__init__(num_envs, device=device, seed=seed, randomize=initial is None)
        if initial is not None:
            self.reset(initial)
        self.variant = variant
        self.variant_level = VARIANT_LEVEL[variant]
        self.defer_car_angular_cap = self.variant_level >= 3
        self.defer_car_linear_cap = self.variant_level >= 3
        self.geometry = geometry or ArenaGeometry.load_soccar(collision_root)
        self.meshes = meshes or WarpArenaMeshes(self.geometry, self.device)
        if ray_backend not in {"default", "cubql"}:
            raise ValueError("ray_backend must be 'default' or 'cubql'")
        self.ray_backend = ray_backend
        self.ray_mesh = getattr(self.meshes, f"{ray_backend}_bt")
        self.vehicle = VehicleState(self.state.car_count, self.device)
        self.boost_pad_positions = wp.array(
            SOCCAR_PAD_POSITIONS, dtype=wp.vec3, device=self.device
        )
        self.boost_pad_cooldown = wp.zeros(
            self.num_envs * PAD_COUNT, dtype=wp.float32, device=self.device
        )
        self.boost_pad_previous_locked_car = wp.zeros(
            self.num_envs * PAD_COUNT, dtype=wp.int32, device=self.device
        )
        self.inertia_transpose_mix = 1.0
        self.plane_bt_mode = 4
        self.support_hysteresis = 0.0000075
        self.tick_counter = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.action_tape: DeviceActionTape | None = None
        if action_tape is not None:
            self.set_action_tape(action_tape)

    @property
    def logical_state_bytes(self) -> int:
        base = StateSnapshot.empty(self.num_envs).nbytes
        vehicle = getattr(self, "vehicle", None)
        boost_pads = 0
        if hasattr(self, "boost_pad_cooldown"):
            boost_pads = self.num_envs * PAD_COUNT * 2 * 4
        return base + (0 if vehicle is None else vehicle.logical_bytes) + boost_pads

    def reset(self, state: StateSnapshot | None = None, *, seed: int | None = None) -> None:
        super().reset(state, seed=seed)
        if hasattr(self, "vehicle"):
            self.vehicle = VehicleState(self.state.car_count, self.device)
        if hasattr(self, "boost_pad_cooldown"):
            self.boost_pad_cooldown = wp.zeros(
                self.num_envs * PAD_COUNT, dtype=wp.float32, device=self.device
            )
            self.boost_pad_previous_locked_car = wp.zeros(
                self.num_envs * PAD_COUNT, dtype=wp.int32, device=self.device
            )
        self.tick_counter = wp.zeros(1, dtype=wp.int32, device=self.device)

    def set_action_tape(self, tape: ActionTape) -> None:
        self.action_tape = DeviceActionTape(tape, self.device)
        self.host_to_device_bytes += tape.nbytes
        self._captured_graph = None
        self._captured_graph_ticks = 0

    def disable_action_tape(self) -> None:
        self.action_tape = None
        self._captured_graph = None
        self._captured_graph_ticks = 0

    def vehicle_snapshot(self) -> VehicleSnapshot:
        self.synchronize()
        result = self.vehicle.snapshot()
        self.device_to_host_bytes += self.vehicle.logical_bytes
        return result

    def _launch_tick(self) -> None:
        if self.action_tape is not None:
            tape = self.action_tape
            controls = self.controls
            wp.launch(
                load_action_tape,
                dim=self.state.car_count,
                inputs=[
                    self.tick_counter,
                    tape.hold_ticks,
                    tape.length,
                    tape.throttle,
                    tape.steer,
                    tape.pitch,
                    tape.yaw,
                    tape.roll,
                    tape.jump,
                    tape.boost,
                    tape.handbrake,
                    controls.throttle,
                    controls.steer,
                    controls.pitch,
                    controls.yaw,
                    controls.roll,
                    controls.jump,
                    controls.boost,
                    controls.handbrake,
                ],
                device=self.device,
            )
        if self.variant_level >= 1:
            state = self.state
            controls = self.controls
            vehicle = self.vehicle
            wp.launch(
                wheel_pre_tick,
                dim=state.car_count,
                inputs=[
                    self.tick_counter,
                    self.ray_mesh.id,
                    self.meshes.points_bt,
                    self.meshes.indices,
                    self.meshes.bullet_face_normals,
                    self.meshes.bullet_bvh_rank,
                    self.meshes.face_mesh_index,
                    int(self.variant_level >= 2),
                    state.car_pos,
                    state.car_vel,
                    state.car_quat,
                    state.car_ang_vel,
                    vehicle.solver_position,
                    vehicle.rigid_position_bt,
                    vehicle.solver_orientation,
                    vehicle.solver_velocity,
                    vehicle.rigid_velocity_bt,
                    vehicle.solver_angular_velocity,
                    vehicle.auto_roll_acceleration,
                    vehicle.auto_roll_angular_acceleration,
                    vehicle.total_force_bt,
                    vehicle.total_torque_bt,
                    vehicle.inverse_inertia_world,
                    vehicle.contact_count,
                    vehicle.world_contact_normal,
                    state.on_ground,
                    state.air_control_disabled,
                    state.boost,
                    controls.throttle,
                    controls.steer,
                    controls.boost,
                    controls.handbrake,
                    vehicle.wheel_ray_start,
                    vehicle.wheel_direction,
                    vehicle.wheel_hit_point,
                    vehicle.wheel_hit_point_bt,
                    vehicle.wheel_hit_normal,
                    vehicle.wheel_hit_distance,
                    vehicle.wheel_hit_face,
                    vehicle.suspension_length,
                    vehicle.suspension_velocity,
                    vehicle.suspension_clipped_factor,
                    vehicle.suspension_force,
                    vehicle.suspension_pushback,
                    vehicle.suspension_force_bt,
                    vehicle.suspension_pushback_bt,
                    vehicle.debug_wheel_ray_from_bt,
                    vehicle.debug_wheel_ray_to_bt,
                    vehicle.debug_wheel_ray_fraction,
                    vehicle.debug_wheel_linear_bt,
                    vehicle.debug_wheel_angular,
                    vehicle.wheel_axle,
                    vehicle.wheel_forward,
                    vehicle.wheel_friction_impulse,
                    vehicle.wheel_friction_impulse_bt,
                    vehicle.wheel_friction_relative_bt,
                    vehicle.side_impulse,
                    vehicle.rolling_impulse,
                    vehicle.engine_acceleration,
                    vehicle.brake_acceleration,
                    vehicle.steer_angle,
                    vehicle.lateral_friction,
                    vehicle.longitudinal_friction,
                    vehicle.wheel_contact,
                    vehicle.wheel_world_contact,
                    vehicle.handbrake_value,
                    vehicle.wheels_with_contact,
                ],
                device=self.device,
            )
        # B1 is intentionally the isolated ray-query component benchmark; the
        # contact-rich origins remain fixed while the device action tape turns.
        if self.variant_level != 1:
            super()._launch_tick()
        if self.variant_level >= 3:
            state = self.state
            vehicle = self.vehicle
            wp.launch(
                chassis_contacts_v021,
                dim=state.car_count,
                inputs=[
                    self.tick_counter,
                    self.meshes.default.id,
                    self.meshes.points_bt,
                    self.meshes.indices,
                    self.meshes.internal_edge_face_normals,
                    self.meshes.internal_edge_crosses,
                    self.meshes.internal_edge_normal_bs,
                    self.meshes.internal_edge_angles,
                    self.meshes.internal_edge_flags,
                    self.meshes.bullet_bvh_rank,
                    self.meshes.face_mesh_index,
                    self.inertia_transpose_mix,
                    self.plane_bt_mode,
                    self.support_hysteresis,
                    state.car_pos,
                    state.car_vel,
                    state.car_quat,
                    state.car_ang_vel,
                    vehicle.solver_position,
                    vehicle.rigid_position_bt,
                    vehicle.solver_orientation,
                    vehicle.solver_velocity,
                    vehicle.rigid_velocity_bt,
                    vehicle.solver_angular_velocity,
                    vehicle.auto_roll_acceleration,
                    vehicle.auto_roll_angular_acceleration,
                    vehicle.total_force_bt,
                    vehicle.total_torque_bt,
                    vehicle.candidate_count,
                    vehicle.mesh_candidate_count,
                    vehicle.mesh_candidate_overflow,
                    vehicle.contact_overflow,
                    vehicle.contact_count,
                    vehicle.world_contact_normal,
                    vehicle.candidate_total,
                    vehicle.contact_total,
                    vehicle.candidate_max,
                    vehicle.contact_max,
                    vehicle.penetration_max,
                    vehicle.contact_point,
                    vehicle.contact_local_a,
                    vehicle.contact_point_b,
                    vehicle.contact_normal,
                    vehicle.contact_tangent,
                    vehicle.contact_face,
                    vehicle.contact_mesh,
                    vehicle.contact_distance,
                    vehicle.contact_distance_bt,
                    vehicle.contact_penetration,
                    vehicle.contact_normal_jacobian,
                    vehicle.contact_tangent_jacobian,
                    vehicle.contact_normal_rhs,
                    vehicle.contact_tangent_rhs,
                    vehicle.contact_push_rhs,
                    vehicle.contact_normal_impulse,
                    vehicle.contact_tangent_impulse,
                    vehicle.contact_push_impulse,
                    vehicle.contact_lifetime,
                    vehicle.mesh_candidate_face,
                    vehicle.plane_support_direction,
                ],
                device=self.device,
            )
            wp.launch(
                boost_pad_tick,
                dim=self.num_envs,
                inputs=[
                    self.boost_pad_positions,
                    state.car_pos,
                    state.car_quat,
                    state.boost,
                    self.boost_pad_cooldown,
                    self.boost_pad_previous_locked_car,
                ],
                device=self.device,
            )
        wp.launch(
            increment_tick_counter,
            dim=1,
            inputs=[self.tick_counter],
            device=self.device,
        )


def make_contact_rich_state(num_envs: int, seed: int = 20260823) -> StateSnapshot:
    """Deterministic floor/ramp/wall/ceiling/landing/body-contact mixture."""

    state = StateSnapshot.empty(num_envs)
    rng = np.random.default_rng(seed)
    car_count = num_envs * 2
    patterns = np.arange(car_count) % 16
    pos = state.car_pos.reshape(car_count, 3)
    velocity = state.car_vel.reshape(car_count, 3)
    angular = state.car_ang_vel.reshape(car_count, 3)
    quat = state.car_quat.reshape(car_count, 4)
    boost = state.boost.reshape(car_count)
    pos[:, 0] = rng.uniform(-2200, 2200, car_count)
    pos[:, 1] = rng.uniform(-3200, 3200, car_count)
    pos[:, 2] = 17.0
    velocity.fill(0.0)
    angular[:] = rng.uniform(-0.8, 0.8, (car_count, 3))
    yaw = rng.uniform(-np.pi, np.pi, car_count)
    pitch = np.zeros(car_count)
    roll = np.zeros(car_count)

    velocity[patterns == 1, 0] = 400.0
    velocity[patterns == 2, 0] = 1200.0
    velocity[patterns == 3] = (650.0, 180.0, 0.0)
    angular[patterns == 3] = (0.0, 0.0, 0.8)
    velocity[patterns == 4, 0] = 1000.0

    mask = patterns == 5
    count = int(mask.sum())
    pos[mask, 0] = rng.choice((-3850.0, 3850.0), count)
    pos[mask, 1] = rng.choice((-4300.0, 4300.0), count)
    pos[mask, 2] = rng.uniform(90, 240, count)
    pitch[mask] = rng.choice((-0.35, 0.35), count)
    velocity[mask] = rng.uniform(-600, 600, (count, 3))

    mask = patterns == 6
    count = int(mask.sum())
    pos[mask, 0] = -4070.0
    pos[mask, 1] = rng.uniform(-3500, 3500, count)
    pos[mask, 2] = rng.uniform(300, 1500, count)
    pitch[mask] = np.pi / 2.0
    velocity[mask, 0] = 40.0
    velocity[mask, 1] = rng.uniform(-500, 500, count)
    velocity[mask, 2] = rng.uniform(-100, 100, count)

    mask = patterns == 7
    count = int(mask.sum())
    pos[mask, 0] = rng.uniform(-700, 700, count)
    pos[mask, 1] = rng.choice((-5100.0, 5100.0), count)
    pos[mask, 2] = rng.uniform(120, 700, count)
    roll[mask] = rng.choice((-np.pi / 2.0, np.pi / 2.0), count)
    velocity[mask] = rng.uniform(-350, 350, (count, 3))

    mask = patterns == 8
    count = int(mask.sum())
    pos[mask, 0] = rng.choice((-3850.0, 3850.0), count)
    pos[mask, 1] = rng.choice((-4600.0, 4600.0), count)
    pos[mask, 2] = rng.uniform(90, 300, count)
    pitch[mask] = rng.choice((-0.35, 0.35), count)
    roll[mask] = rng.choice((-0.2, 0.2), count)
    velocity[mask] = rng.uniform(-600, 600, (count, 3))

    mask = patterns == 9
    count = int(mask.sum())
    pos[mask, 0] = rng.uniform(-2500, 2500, count)
    pos[mask, 1] = rng.uniform(-3500, 3500, count)
    pos[mask, 2] = 2028.0
    roll[mask] = np.pi
    velocity[mask, 0:2] = rng.uniform(-300, 300, (count, 2))
    velocity[mask, 2] = 15.0

    mask = patterns == 10
    count = int(mask.sum())
    pos[mask, 2] = 180.0
    velocity[mask, 0:2] = rng.uniform(-150, 150, (count, 2))
    velocity[mask, 2] = -600.0
    mask = patterns == 11
    pos[mask, 2] = 35.0
    pitch[mask] = 0.30
    velocity[mask, 2] = -250.0
    mask = patterns == 12
    pos[mask, 2] = 27.0
    roll[mask] = -0.18
    pitch[mask] = 0.12
    velocity[mask, 2] = -120.0

    mask = patterns == 13
    count = int(mask.sum())
    pos[mask, 2] = 4.0
    pitch[mask] = 0.55
    velocity[mask, 0:2] = rng.uniform(-500, 500, (count, 2))
    velocity[mask, 2] = -180.0
    mask = patterns == 14
    count = int(mask.sum())
    pos[mask, 2] = 5.0
    roll[mask] = 0.8
    velocity[mask, 0] = rng.uniform(-400, 400, count)
    velocity[mask, 1] = rng.uniform(-600, 600, count)
    velocity[mask, 2] = -220.0
    mask = patterns == 15
    pos[mask, 2] = 31.0
    roll[mask] = 0.42
    pitch[mask] = -0.12
    velocity[mask] = (350.0, 80.0, -160.0)

    quat[:] = _quats_from_euler(roll, pitch, yaw)
    boost[:] = rng.uniform(20.0, 100.0, car_count)
    state.validate()
    return state


def _quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    return np.asarray(
        (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ),
        dtype=np.float32,
    )


def _quats_from_euler(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    result = np.column_stack(
        (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        )
    )
    return np.asarray(result, dtype=np.float32)
