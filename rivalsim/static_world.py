"""v0.2 shared-arena simulator with selectable B0/B1/B2/B3 layers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from rivalsim.arena import ArenaGeometry, WarpArenaMeshes
from rivalsim.ball_world_state import BallWorldState
from rivalsim.car_ball_state import CarBallState
from rivalsim.car_car_state import CarCarState
from rivalsim.kernels.ball_world import ball_world_tick, initialize_ball_world_internal
from rivalsim.kernels.boost_pad import (
    PAD_COUNT,
    SOCCAR_PAD_POSITIONS,
    boost_pad_tick,
    boost_pad_tick_lifecycle,
)
from rivalsim.kernels.car_ball import capture_car_ball_inputs, car_ball_tick
from rivalsim.kernels.car_car import (
    capture_car_car_inputs,
    car_car_tick,
)
from rivalsim.kernels.integrated import (
    integrated_two_car_ball_tick,
    update_integrated_broadphase_order,
)
from rivalsim.kernels.lifecycle import lifecycle_post_tick, lifecycle_pre_tick
from rivalsim.kernels.rsqrtss_amd import amd_rsqrtss_table
from rivalsim.kernels.vehicle import (
    apply_resident_mechanics_pre_solve,
    chassis_contacts_v021,
    increment_tick_counter,
    load_action_tape,
    wheel_pre_tick,
)
from rivalsim.lifecycle_state import LifecycleSnapshot, LifecycleState
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
        self.enable_ball_wheel_rays = False
        self.amd_rsqrtss_mantissa = wp.array(
            amd_rsqrtss_table(), dtype=wp.uint16, device=self.device
        )
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
            ball_world = getattr(self, "ball_world", None)
            # RocketSim's vehicle raycaster and bilateral wheel-friction path
            # read the ball's btRigidBody transform and velocity directly.
            # Keep those internal Bullet-unit values resident instead of
            # round-tripping them through the public UU state, which can move
            # a dynamic wheel witness by one float32 ULP after integration.
            ball_position_bt = (
                state.ball_pos
                if ball_world is None
                else ball_world.position_bt
            )
            ball_velocity_bt = (
                state.ball_vel
                if ball_world is None
                else ball_world.velocity_bt
            )
            ball_proxy_min_bt = (
                state.ball_pos
                if ball_world is None
                else ball_world.broadphase_proxy_min_bt
            )
            dynamic_ray_mode = int(self.enable_ball_wheel_rays)
            pair = getattr(self, "car_car", None)
            # The pinned source visits ``Arena::_cars`` serially. Phase C keeps
            # the resulting logical A/B order as per-world lifecycle state and
            # uses two launch boundaries so the second prepass observes the
            # first prepass's resident rigid-body values. Non-car-car modes do
            # not read the placeholder array because ``visit_ordinal`` is -1.
            pre_tick_first_car = (
                pair.pre_tick_first_car
                if dynamic_ray_mode >= 2
                else self.tick_counter
            )
            visit_ordinal = 0 if dynamic_ray_mode >= 2 else -1
            wheel_inputs = [
                    self.tick_counter,
                    self.ray_mesh.id,
                    self.meshes.points_bt,
                    self.meshes.indices,
                    self.meshes.bullet_face_normals,
                    self.meshes.bullet_bvh_rank,
                    self.meshes.face_mesh_index,
                    int(self.variant_level >= 2),
                    dynamic_ray_mode,
                    pre_tick_first_car,
                    visit_ordinal,
                    self.amd_rsqrtss_mantissa,
                    ball_position_bt,
                    state.ball_quat,
                    ball_velocity_bt,
                    state.ball_ang_vel,
                    ball_proxy_min_bt,
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
                    vehicle.pre_tick_forward_speed,
                    vehicle.contact_count,
                    vehicle.world_contact_normal,
                    state.on_ground,
                    state.air_control_disabled,
                    state.boost,
                    state.boosting_time,
                    state.is_boosting,
                    state.has_flipped,
                    state.is_flipping,
                    state.flip_time,
                    state.flip_rel_torque,
                    state.is_auto_flipping,
                    controls.throttle,
                    controls.steer,
                    controls.pitch,
                    controls.yaw,
                    controls.roll,
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
                ]
            wp.launch(
                wheel_pre_tick,
                dim=state.car_count,
                inputs=wheel_inputs,
                device=self.device,
            )
            if dynamic_ray_mode >= 2:
                # Finish each world's selected native-valid order.
                wheel_inputs[10] = 1
                wp.launch(
                    wheel_pre_tick,
                    dim=state.car_count,
                    inputs=wheel_inputs,
                    device=self.device,
                )
            wp.launch(
                apply_resident_mechanics_pre_solve,
                dim=state.car_count,
                inputs=[
                    int(self.resident_full_world_mechanics),
                    self.tick_counter,
                    vehicle.pre_tick_forward_speed,
                    vehicle.contact_count,
                    vehicle.world_contact_normal,
                    state.car_quat,
                    state.on_ground,
                    state.has_jumped,
                    state.is_jumping,
                    state.has_double_jumped,
                    state.has_flipped,
                    state.is_flipping,
                    state.jump_time,
                    state.air_time_since_jump,
                    state.flip_time,
                    state.auto_flip_timer,
                    state.auto_flip_torque_scale,
                    state.is_auto_flipping,
                    state.prev_jump,
                    controls.pitch,
                    controls.yaw,
                    controls.roll,
                    controls.jump,
                    vehicle.solver_velocity,
                    vehicle.solver_angular_velocity,
                    vehicle.rigid_velocity_bt,
                    vehicle.total_force_bt,
                ],
                device=self.device,
            )
            self._after_wheel_pre_tick()
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
            if not getattr(self, "defer_boost_pad_tick", False):
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

    def _after_wheel_pre_tick(self) -> None:
        """Extension point after RocketSim wheel impulses and before Bullet solve."""


class DynamicWorldSim(StaticWorldSim):
    """v0.3 resident Soccar world with the bounded dynamic-contact phases."""

    def __init__(self, *args, enable_car_ball_contacts: bool = True, **kwargs):
        self.enable_car_ball_contacts = bool(enable_car_ball_contacts)
        super().__init__(*args, **kwargs)
        self.enable_ball_wheel_rays = self.enable_car_ball_contacts
        self.defer_ball_physics = True
        self.ball_world = BallWorldState(self.num_envs, self.device)
        self.car_ball = CarBallState(self.num_envs, self.device)
        self._initialize_ball_world_internal()

    @property
    def logical_state_bytes(self) -> int:
        base = super().logical_state_bytes
        ball_world = getattr(self, "ball_world", None)
        car_ball = getattr(self, "car_ball", None)
        return (
            base
            + (0 if ball_world is None else ball_world.logical_bytes)
            + (0 if car_ball is None else car_ball.logical_bytes)
        )

    def reset(self, state: StateSnapshot | None = None, *, seed: int | None = None) -> None:
        super().reset(state, seed=seed)
        if hasattr(self, "ball_world"):
            self.ball_world = BallWorldState(self.num_envs, self.device)
            self.car_ball = CarBallState(self.num_envs, self.device)
            self._initialize_ball_world_internal()

    def _initialize_ball_world_internal(self) -> None:
        wp.launch(
            initialize_ball_world_internal,
            dim=self.num_envs,
            inputs=[
                self.state.ball_pos,
                self.state.ball_vel,
                self.ball_world.position_bt,
                self.ball_world.velocity_bt,
                self.ball_world.broadphase_proxy_min_bt,
            ],
            device=self.device,
        )

    def _launch_tick(self) -> None:
        super()._launch_tick()
        state = self.state
        ball = self.ball_world
        wp.launch(
            ball_world_tick,
            dim=self.num_envs,
            inputs=[
                self.meshes.default.id,
                self.meshes.points_bt,
                self.meshes.indices,
                self.amd_rsqrtss_mantissa,
                self.meshes.internal_edge_face_normals,
                self.meshes.internal_edge_crosses,
                self.meshes.internal_edge_normal_bs,
                self.meshes.internal_edge_angles,
                self.meshes.internal_edge_flags,
                self.meshes.bullet_bvh_rank,
                self.meshes.face_mesh_index,
                state.ball_pos,
                state.ball_vel,
                state.ball_quat,
                state.ball_ang_vel,
                ball.position_bt,
                ball.velocity_bt,
                ball.broadphase_proxy_min_bt,
                ball.contact_count,
                ball.candidate_count,
                ball.candidate_overflow,
                ball.contact_overflow,
                ball.contact_local_a_bt,
                ball.contact_point_b_bt,
                ball.contact_normal,
                ball.contact_distance_bt,
                ball.contact_face,
                ball.contact_mesh,
                ball.contact_lifetime,
                ball.contact_normal_impulse,
                ball.contact_tangent_impulse,
                ball.contact_tangent,
                ball.contact_normal_jacobian,
                ball.contact_tangent_jacobian,
                ball.contact_normal_rhs,
                ball.contact_tangent_rhs,
                ball.contact_push_rhs,
                ball.contact_push_impulse,
                ball.candidate_face,
            ],
            device=self.device,
        )
        if not self.enable_car_ball_contacts:
            return
        pair = self.car_ball
        wp.launch(
            car_ball_tick,
            dim=self.num_envs,
            inputs=[
                self.tick_counter,
                0,
                self.amd_rsqrtss_mantissa,
                self.meshes.rs_static_cell_mask,
                self.meshes.rs_static_aabb_min_bt,
                self.meshes.rs_static_aabb_max_bt,
                self.meshes.rs_equal_island_permutation,
                self.vehicle.total_force_bt,
                self.vehicle.total_torque_bt,
                state.car_pos,
                state.car_vel,
                state.car_quat,
                state.car_ang_vel,
                self.vehicle.rigid_position_bt,
                self.vehicle.rigid_velocity_bt,
                state.ball_pos,
                state.ball_vel,
                state.ball_quat,
                state.ball_ang_vel,
                ball.position_bt,
                ball.velocity_bt,
                self.vehicle.contact_count,
                self.vehicle.contact_local_a,
                self.vehicle.contact_normal,
                self.vehicle.contact_tangent,
                self.vehicle.contact_mesh,
                self.vehicle.contact_normal_jacobian,
                self.vehicle.contact_tangent_jacobian,
                self.vehicle.contact_normal_rhs,
                self.vehicle.contact_tangent_rhs,
                self.vehicle.contact_push_rhs,
                self.vehicle.contact_normal_impulse,
                self.vehicle.contact_tangent_impulse,
                self.vehicle.contact_push_impulse,
                ball.contact_count,
                ball.contact_local_a_bt,
                ball.contact_normal,
                ball.contact_tangent,
                ball.contact_mesh,
                ball.contact_normal_jacobian,
                ball.contact_tangent_jacobian,
                ball.contact_normal_rhs,
                ball.contact_tangent_rhs,
                ball.contact_push_rhs,
                ball.contact_normal_impulse,
                ball.contact_tangent_impulse,
                ball.contact_push_impulse,
                pair.pre_car_position_bt,
                pair.pre_car_velocity_bt,
                pair.pre_car_quaternion,
                pair.pre_car_angular_velocity,
                pair.pre_ball_position_bt,
                pair.pre_ball_velocity_bt,
                pair.pre_ball_quaternion,
                pair.pre_ball_angular_velocity,
                pair.contact_count,
                pair.hit_this_tick,
                pair.algorithm_active,
                pair.contact_point_a_bt,
                pair.contact_point_b_bt,
                pair.contact_normal,
                pair.contact_tangent,
                pair.contact_distance_bt,
                pair.normal_impulse,
                pair.tangent_impulse,
                pair.push_impulse,
                pair.extra_hit_velocity_uu,
                pair.relative_pos_on_ball_uu,
                pair.last_extra_impulse_tick,
                pair.manifold_local_a_bt,
                pair.manifold_local_b_bt,
                pair.manifold_normal,
                pair.manifold_tangent,
                pair.manifold_distance_bt,
                pair.manifold_lifetime,
                pair.manifold_normal_jacobian,
                pair.manifold_tangent_jacobian,
                pair.manifold_normal_rhs,
                pair.manifold_tangent_rhs,
                pair.manifold_push_rhs,
                pair.manifold_normal_impulse,
                pair.manifold_tangent_impulse,
                pair.manifold_push_impulse,
            ],
            device=self.device,
        )

    def _after_wheel_pre_tick(self) -> None:
        if not self.enable_car_ball_contacts:
            return
        state = self.state
        vehicle = self.vehicle
        ball = self.ball_world
        pair = self.car_ball
        wp.launch(
            capture_car_ball_inputs,
            dim=self.num_envs,
            inputs=[
                0,
                vehicle.rigid_position_bt,
                vehicle.rigid_velocity_bt,
                vehicle.solver_orientation,
                vehicle.solver_angular_velocity,
                ball.position_bt,
                ball.velocity_bt,
                state.ball_quat,
                state.ball_ang_vel,
                pair.pre_car_position_bt,
                pair.pre_car_velocity_bt,
                pair.pre_car_quaternion,
                pair.pre_car_angular_velocity,
                pair.pre_ball_position_bt,
                pair.pre_ball_velocity_bt,
                pair.pre_ball_quaternion,
                pair.pre_ball_angular_velocity,
            ],
            device=self.device,
        )


class CarCarWorldSim(StaticWorldSim):
    """v0.3 isolated two-Octane dynamic-contact world."""

    def __init__(
        self,
        *args,
        car_visitation_order: str | int | np.ndarray | None = None,
        car_lifecycle_seed: int | None = None,
        **kwargs,
    ):
        physical_seed = int(kwargs.get("seed", 0))
        self._car_lifecycle_seed = (
            physical_seed if car_lifecycle_seed is None else int(car_lifecycle_seed)
        )
        self._initial_car_visitation_order = car_visitation_order
        super().__init__(*args, **kwargs)
        self.enable_ball_wheel_rays = 2
        self.car_car = CarCarState(
            self.num_envs,
            self.device,
            lifecycle_seed=self._car_lifecycle_seed,
            pre_tick_first_car=self._initial_car_visitation_order,
        )
        self.host_to_device_bytes += self.num_envs * 4

    @property
    def logical_state_bytes(self) -> int:
        pair = getattr(self, "car_car", None)
        return super().logical_state_bytes + (0 if pair is None else pair.logical_bytes)

    def reset(self, state: StateSnapshot | None = None, *, seed: int | None = None) -> None:
        # SetState/kickoff-style physical resets do not mutate Arena::_cars in
        # the pinned source. Preserve the lifecycle-selected visit order and
        # epoch while clearing all contact, solver, and event state.
        lifecycle = (
            self.car_car.lifecycle_copy_kwargs()
            if hasattr(self, "car_car")
            else None
        )
        super().reset(state, seed=seed)
        if lifecycle is not None:
            self.car_car = CarCarState(self.num_envs, self.device, **lifecycle)
            self.host_to_device_bytes += self.num_envs * 4

    def car_membership_changed(
        self,
        car_visitation_order: str | int | np.ndarray | None = None,
    ) -> None:
        """Establish a new per-world order after insertion/removal/reconstruction.

        The selection is internal lifecycle state. An explicit logical order is
        accepted so source-authority tests can exercise both valid branches;
        neither path inspects physical state, case identity, or expected output.
        """

        self.car_car.membership_changed(car_visitation_order)
        self.host_to_device_bytes += self.num_envs * 4
        self._captured_graph = None
        self._captured_graph_ticks = 0

    def reconstruct_car_container(
        self,
        car_visitation_order: str | int | np.ndarray | None = None,
    ) -> None:
        """Model a new arena/fixed-pair construction lifecycle boundary."""

        self.car_membership_changed(car_visitation_order)

    def _launch_tick(self) -> None:
        super()._launch_tick()
        state = self.state
        vehicle = self.vehicle
        pair = self.car_car
        wp.launch(
            car_car_tick,
            dim=self.num_envs,
            inputs=[
                self.tick_counter,
                self.amd_rsqrtss_mantissa,
                self.meshes.rs_static_cell_mask,
                self.meshes.rs_static_aabb_min_bt,
                self.meshes.rs_static_aabb_max_bt,
                self.meshes.rs_equal_island_permutation,
                vehicle.total_force_bt,
                vehicle.total_torque_bt,
                state.car_pos,
                state.car_vel,
                state.car_quat,
                state.car_ang_vel,
                state.on_ground,
                state.is_supersonic,
                state.supersonic_time,
                vehicle.rigid_position_bt,
                vehicle.rigid_velocity_bt,
                vehicle.contact_count,
                vehicle.contact_local_a,
                vehicle.contact_normal,
                vehicle.contact_tangent,
                vehicle.contact_mesh,
                vehicle.contact_normal_jacobian,
                vehicle.contact_tangent_jacobian,
                vehicle.contact_normal_rhs,
                vehicle.contact_tangent_rhs,
                vehicle.contact_push_rhs,
                vehicle.contact_normal_impulse,
                vehicle.contact_tangent_impulse,
                vehicle.contact_push_impulse,
                pair.pre_position_bt,
                pair.pre_velocity_bt,
                pair.pre_quaternion,
                pair.pre_angular_velocity,
                pair.pre_on_ground,
                pair.pre_is_supersonic,
                pair.pre_supersonic_time,
                pair.queued_velocity_bt,
                pair.car_contact_id,
                pair.car_contact_cooldown,
                pair.car_is_demoed,
                pair.contact_count,
                pair.return_code,
                pair.algorithm_active,
                pair.contact_point_b_bt,
                pair.contact_normal,
                pair.contact_distance_bt,
                pair.manifold_local_a_bt,
                pair.manifold_local_b_bt,
                pair.manifold_normal,
                pair.manifold_tangent,
                pair.manifold_distance_bt,
                pair.manifold_normal_jacobian,
                pair.manifold_tangent_jacobian,
                pair.manifold_normal_rhs,
                pair.manifold_tangent_rhs,
                pair.manifold_push_rhs,
                pair.manifold_normal_impulse,
                pair.manifold_tangent_impulse,
                pair.manifold_push_impulse,
                pair.event_count,
                pair.event_bumper,
                pair.event_victim,
                pair.event_is_demo,
            ],
            device=self.device,
        )

    def _after_wheel_pre_tick(self) -> None:
        state = self.state
        vehicle = self.vehicle
        pair = self.car_car
        wp.launch(
            capture_car_car_inputs,
            dim=state.car_count,
            inputs=[
                vehicle.rigid_position_bt,
                vehicle.rigid_velocity_bt,
                vehicle.solver_orientation,
                vehicle.solver_angular_velocity,
                state.on_ground,
                state.is_supersonic,
                state.supersonic_time,
                pair.pre_position_bt,
                pair.pre_velocity_bt,
                pair.pre_quaternion,
                pair.pre_angular_velocity,
                pair.pre_on_ground,
                pair.pre_is_supersonic,
                pair.pre_supersonic_time,
            ],
            device=self.device,
        )


class IntegratedWorldSim(DynamicWorldSim):
    """v0.3 fixed two-Octane/one-ball integrated Soccar world.

    The pairwise kernels first retain the already accepted collision/manifold
    streams.  Phase D's bounded shared-island writeback is layered after those
    source paths; no scenario or authority output is visible here.
    """

    def __init__(
        self,
        *args,
        car_visitation_order: str | int | np.ndarray | None = None,
        car_lifecycle_seed: int | None = None,
        **kwargs,
    ):
        physical_seed = int(kwargs.get("seed", 0))
        self._car_lifecycle_seed = (
            physical_seed if car_lifecycle_seed is None else int(car_lifecycle_seed)
        )
        self._initial_car_visitation_order = car_visitation_order
        super().__init__(*args, **kwargs)
        # Mode 3 visits both cars in lifecycle order and admits both the ball
        # and the other Octane to each native vehicle ray.
        self.enable_ball_wheel_rays = 3
        self.car_ball_b = CarBallState(self.num_envs, self.device)
        self.car_car = CarCarState(
            self.num_envs,
            self.device,
            lifecycle_seed=self._car_lifecycle_seed,
            pre_tick_first_car=self._initial_car_visitation_order,
        )
        self.host_to_device_bytes += self.num_envs * 4
        self._initialize_integrated_broadphase_order()

    @property
    def logical_state_bytes(self) -> int:
        pair_b = getattr(self, "car_ball_b", None)
        pair_car = getattr(self, "car_car", None)
        return (
            super().logical_state_bytes
            + (0 if pair_b is None else pair_b.logical_bytes)
            + (0 if pair_car is None else pair_car.logical_bytes)
            + self.num_envs * 8 * 4
        )

    def _initialize_integrated_broadphase_order(self) -> None:
        # Source proxy construction order is ball, A, B. Ball's construction
        # AABB starts in cell 2862; both identity-transform Octane compounds
        # start in cell 3066. Subsequent updates are device-resident.
        self._dynamic_proxy_cell = wp.array(
            np.tile(np.asarray((2862, 3066, 3066), dtype=np.int32), self.num_envs),
            dtype=wp.int32,
            device=self.device,
        )
        self._dynamic_proxy_move_rank = wp.array(
            np.tile(np.asarray((0, 1, 2), dtype=np.int32), self.num_envs),
            dtype=wp.int32,
            device=self.device,
        )
        self._dynamic_proxy_move_counter = wp.full(
            self.num_envs, 2, dtype=wp.int32, device=self.device
        )
        self._pair_a_before_b = wp.ones(
            self.num_envs, dtype=wp.int32, device=self.device
        )
        self.host_to_device_bytes += self.num_envs * 8 * 4

    def reset(self, state: StateSnapshot | None = None, *, seed: int | None = None) -> None:
        lifecycle = (
            self.car_car.lifecycle_copy_kwargs() if hasattr(self, "car_car") else None
        )
        super().reset(state, seed=seed)
        if hasattr(self, "car_ball_b"):
            self.car_ball_b = CarBallState(self.num_envs, self.device)
        if lifecycle is not None:
            self.car_car = CarCarState(self.num_envs, self.device, **lifecycle)
            self.host_to_device_bytes += self.num_envs * 4

    def car_membership_changed(
        self, car_visitation_order: str | int | np.ndarray | None = None
    ) -> None:
        self.car_car.membership_changed(car_visitation_order)
        self.host_to_device_bytes += self.num_envs * 4
        self._initialize_integrated_broadphase_order()
        self._captured_graph = None
        self._captured_graph_ticks = 0

    def reconstruct_car_container(
        self, car_visitation_order: str | int | np.ndarray | None = None
    ) -> None:
        self.car_membership_changed(car_visitation_order)

    def _launch_tick(self) -> None:
        # DynamicWorldSim retains ball/world and logical-A/ball contacts.
        super()._launch_tick()
        self._launch_car_ball_b()
        self._launch_integrated_car_car()
        self._launch_shared_two_car_ball_island()

    def _launch_shared_two_car_ball_island(self) -> None:
        state = self.state
        vehicle = self.vehicle
        ball = self.ball_world
        pair_a = self.car_ball
        pair_b = self.car_ball_b
        car_pair = self.car_car
        wp.launch(
            integrated_two_car_ball_tick,
            dim=self.num_envs,
            inputs=[
                self.tick_counter,
                self.amd_rsqrtss_mantissa,
                self.meshes.rs_static_cell_mask,
                self.meshes.rs_static_aabb_min_bt,
                self.meshes.rs_static_aabb_max_bt,
                self.meshes.rs_equal_island_permutation,
                self._pair_a_before_b,
                vehicle.total_force_bt,
                vehicle.total_torque_bt,
                state.car_pos,
                state.car_vel,
                state.car_quat,
                state.car_ang_vel,
                state.is_supersonic,
                state.supersonic_time,
                vehicle.rigid_position_bt,
                vehicle.rigid_velocity_bt,
                vehicle.contact_count,
                vehicle.contact_local_a,
                vehicle.contact_normal,
                vehicle.contact_tangent,
                vehicle.contact_mesh,
                vehicle.contact_normal_jacobian,
                vehicle.contact_tangent_jacobian,
                vehicle.contact_normal_rhs,
                vehicle.contact_tangent_rhs,
                vehicle.contact_push_rhs,
                vehicle.contact_normal_impulse,
                vehicle.contact_tangent_impulse,
                vehicle.contact_push_impulse,
                state.ball_pos,
                state.ball_vel,
                state.ball_quat,
                state.ball_ang_vel,
                ball.position_bt,
                ball.velocity_bt,
                ball.contact_count,
                ball.contact_local_a_bt,
                ball.contact_normal,
                ball.contact_tangent,
                ball.contact_mesh,
                ball.contact_normal_jacobian,
                ball.contact_tangent_jacobian,
                ball.contact_normal_rhs,
                ball.contact_tangent_rhs,
                ball.contact_push_rhs,
                ball.contact_normal_impulse,
                ball.contact_tangent_impulse,
                ball.contact_push_impulse,
                car_pair.pre_position_bt,
                car_pair.pre_velocity_bt,
                car_pair.pre_quaternion,
                car_pair.pre_angular_velocity,
                car_pair.pre_is_supersonic,
                car_pair.pre_supersonic_time,
                pair_a.pre_ball_position_bt,
                pair_a.pre_ball_velocity_bt,
                pair_a.pre_ball_quaternion,
                pair_a.pre_ball_angular_velocity,
                pair_a.algorithm_active,
                pair_a.contact_count,
                pair_a.manifold_local_a_bt,
                pair_a.manifold_local_b_bt,
                pair_a.manifold_normal,
                pair_a.manifold_tangent,
                pair_a.manifold_normal_jacobian,
                pair_a.manifold_tangent_jacobian,
                pair_a.manifold_normal_rhs,
                pair_a.manifold_tangent_rhs,
                pair_a.manifold_push_rhs,
                pair_a.manifold_normal_impulse,
                pair_a.manifold_tangent_impulse,
                pair_a.manifold_push_impulse,
                pair_a.extra_hit_velocity_uu,
                pair_b.algorithm_active,
                pair_b.contact_count,
                pair_b.manifold_local_a_bt,
                pair_b.manifold_local_b_bt,
                pair_b.manifold_normal,
                pair_b.manifold_tangent,
                pair_b.manifold_normal_jacobian,
                pair_b.manifold_tangent_jacobian,
                pair_b.manifold_normal_rhs,
                pair_b.manifold_tangent_rhs,
                pair_b.manifold_push_rhs,
                pair_b.manifold_normal_impulse,
                pair_b.manifold_tangent_impulse,
                pair_b.manifold_push_impulse,
                pair_b.extra_hit_velocity_uu,
                car_pair.algorithm_active,
                car_pair.queued_velocity_bt,
            ],
            device=self.device,
        )

    def _launch_car_ball_b(self) -> None:
        state = self.state
        vehicle = self.vehicle
        ball = self.ball_world
        pair = self.car_ball_b
        wp.launch(
            car_ball_tick,
            dim=self.num_envs,
            inputs=[
                self.tick_counter,
                1,
                self.amd_rsqrtss_mantissa,
                self.meshes.rs_static_cell_mask,
                self.meshes.rs_static_aabb_min_bt,
                self.meshes.rs_static_aabb_max_bt,
                self.meshes.rs_equal_island_permutation,
                vehicle.total_force_bt,
                vehicle.total_torque_bt,
                state.car_pos,
                state.car_vel,
                state.car_quat,
                state.car_ang_vel,
                vehicle.rigid_position_bt,
                vehicle.rigid_velocity_bt,
                state.ball_pos,
                state.ball_vel,
                state.ball_quat,
                state.ball_ang_vel,
                ball.position_bt,
                ball.velocity_bt,
                vehicle.contact_count,
                vehicle.contact_local_a,
                vehicle.contact_normal,
                vehicle.contact_tangent,
                vehicle.contact_mesh,
                vehicle.contact_normal_jacobian,
                vehicle.contact_tangent_jacobian,
                vehicle.contact_normal_rhs,
                vehicle.contact_tangent_rhs,
                vehicle.contact_push_rhs,
                vehicle.contact_normal_impulse,
                vehicle.contact_tangent_impulse,
                vehicle.contact_push_impulse,
                ball.contact_count,
                ball.contact_local_a_bt,
                ball.contact_normal,
                ball.contact_tangent,
                ball.contact_mesh,
                ball.contact_normal_jacobian,
                ball.contact_tangent_jacobian,
                ball.contact_normal_rhs,
                ball.contact_tangent_rhs,
                ball.contact_push_rhs,
                ball.contact_normal_impulse,
                ball.contact_tangent_impulse,
                ball.contact_push_impulse,
                pair.pre_car_position_bt,
                pair.pre_car_velocity_bt,
                pair.pre_car_quaternion,
                pair.pre_car_angular_velocity,
                pair.pre_ball_position_bt,
                pair.pre_ball_velocity_bt,
                pair.pre_ball_quaternion,
                pair.pre_ball_angular_velocity,
                pair.contact_count,
                pair.hit_this_tick,
                pair.algorithm_active,
                pair.contact_point_a_bt,
                pair.contact_point_b_bt,
                pair.contact_normal,
                pair.contact_tangent,
                pair.contact_distance_bt,
                pair.normal_impulse,
                pair.tangent_impulse,
                pair.push_impulse,
                pair.extra_hit_velocity_uu,
                pair.relative_pos_on_ball_uu,
                pair.last_extra_impulse_tick,
                pair.manifold_local_a_bt,
                pair.manifold_local_b_bt,
                pair.manifold_normal,
                pair.manifold_tangent,
                pair.manifold_distance_bt,
                pair.manifold_lifetime,
                pair.manifold_normal_jacobian,
                pair.manifold_tangent_jacobian,
                pair.manifold_normal_rhs,
                pair.manifold_tangent_rhs,
                pair.manifold_push_rhs,
                pair.manifold_normal_impulse,
                pair.manifold_tangent_impulse,
                pair.manifold_push_impulse,
            ],
            device=self.device,
        )

    def _launch_integrated_car_car(self) -> None:
        state = self.state
        vehicle = self.vehicle
        pair = self.car_car
        wp.launch(
            car_car_tick,
            dim=self.num_envs,
            inputs=[
                self.tick_counter,
                self.amd_rsqrtss_mantissa,
                self.meshes.rs_static_cell_mask,
                self.meshes.rs_static_aabb_min_bt,
                self.meshes.rs_static_aabb_max_bt,
                self.meshes.rs_equal_island_permutation,
                vehicle.total_force_bt,
                vehicle.total_torque_bt,
                state.car_pos,
                state.car_vel,
                state.car_quat,
                state.car_ang_vel,
                state.on_ground,
                state.is_supersonic,
                state.supersonic_time,
                vehicle.rigid_position_bt,
                vehicle.rigid_velocity_bt,
                vehicle.contact_count,
                vehicle.contact_local_a,
                vehicle.contact_normal,
                vehicle.contact_tangent,
                vehicle.contact_mesh,
                vehicle.contact_normal_jacobian,
                vehicle.contact_tangent_jacobian,
                vehicle.contact_normal_rhs,
                vehicle.contact_tangent_rhs,
                vehicle.contact_push_rhs,
                vehicle.contact_normal_impulse,
                vehicle.contact_tangent_impulse,
                vehicle.contact_push_impulse,
                pair.pre_position_bt,
                pair.pre_velocity_bt,
                pair.pre_quaternion,
                pair.pre_angular_velocity,
                pair.pre_on_ground,
                pair.pre_is_supersonic,
                pair.pre_supersonic_time,
                pair.queued_velocity_bt,
                pair.car_contact_id,
                pair.car_contact_cooldown,
                pair.car_is_demoed,
                pair.contact_count,
                pair.return_code,
                pair.algorithm_active,
                pair.contact_point_b_bt,
                pair.contact_normal,
                pair.contact_distance_bt,
                pair.manifold_local_a_bt,
                pair.manifold_local_b_bt,
                pair.manifold_normal,
                pair.manifold_tangent,
                pair.manifold_distance_bt,
                pair.manifold_normal_jacobian,
                pair.manifold_tangent_jacobian,
                pair.manifold_normal_rhs,
                pair.manifold_tangent_rhs,
                pair.manifold_push_rhs,
                pair.manifold_normal_impulse,
                pair.manifold_tangent_impulse,
                pair.manifold_push_impulse,
                pair.event_count,
                pair.event_bumper,
                pair.event_victim,
                pair.event_is_demo,
            ],
            device=self.device,
        )

    def _after_wheel_pre_tick(self) -> None:
        # Capture every pair from the same pre-Bullet rigid state after the
        # complete serial RocketSim car prepass.
        super()._after_wheel_pre_tick()
        state = self.state
        vehicle = self.vehicle
        ball = self.ball_world
        pair_b = self.car_ball_b
        wp.launch(
            capture_car_ball_inputs,
            dim=self.num_envs,
            inputs=[
                1,
                vehicle.rigid_position_bt,
                vehicle.rigid_velocity_bt,
                vehicle.solver_orientation,
                vehicle.solver_angular_velocity,
                ball.position_bt,
                ball.velocity_bt,
                state.ball_quat,
                state.ball_ang_vel,
                pair_b.pre_car_position_bt,
                pair_b.pre_car_velocity_bt,
                pair_b.pre_car_quaternion,
                pair_b.pre_car_angular_velocity,
                pair_b.pre_ball_position_bt,
                pair_b.pre_ball_velocity_bt,
                pair_b.pre_ball_quaternion,
                pair_b.pre_ball_angular_velocity,
            ],
            device=self.device,
        )
        pair = self.car_car
        wp.launch(
            capture_car_car_inputs,
            dim=state.car_count,
            inputs=[
                vehicle.rigid_position_bt,
                vehicle.rigid_velocity_bt,
                vehicle.solver_orientation,
                vehicle.solver_angular_velocity,
                state.on_ground,
                state.is_supersonic,
                state.supersonic_time,
                pair.pre_position_bt,
                pair.pre_velocity_bt,
                pair.pre_quaternion,
                pair.pre_angular_velocity,
                pair.pre_on_ground,
                pair.pre_is_supersonic,
                pair.pre_supersonic_time,
            ],
            device=self.device,
        )
        wp.launch(
            update_integrated_broadphase_order,
            dim=self.num_envs,
            inputs=[
                self.tick_counter,
                pair.pre_position_bt,
                pair.pre_velocity_bt,
                pair.pre_quaternion,
                pair.pre_angular_velocity,
                pair_b.pre_ball_position_bt,
                pair_b.pre_ball_velocity_bt,
                pair_b.pre_ball_quaternion,
                pair_b.pre_ball_angular_velocity,
                self._dynamic_proxy_cell,
                self._dynamic_proxy_move_rank,
                self._dynamic_proxy_move_counter,
                self._pair_a_before_b,
            ],
            device=self.device,
        )


def make_standard_kickoff_state(
    num_envs: int, kickoff_selector: int | np.ndarray = 0
) -> StateSnapshot:
    """Build the immediate RocketSim readback for standard two-car kickoffs."""

    selector = np.broadcast_to(
        np.asarray(kickoff_selector, dtype=np.int32), (num_envs,)
    ).copy()
    if np.any((selector < 0) | (selector >= 5)):
        raise ValueError("kickoff selector entries must be in [0, 5)")
    state = StateSnapshot.empty(num_envs)
    spawn = np.asarray(
        (
            (-2048.0, -2560.0, np.pi / 4.0),
            (2048.0, -2560.0, 3.0 * np.pi / 4.0),
            (-256.0, -3840.0, np.pi / 2.0),
            (256.0, -3840.0, np.pi / 2.0),
            (0.0, -4608.0, np.pi / 2.0),
        ),
        dtype=np.float32,
    )
    for env, layout in enumerate(selector):
        x, y, yaw = spawn[int(layout)]
        for local_car in range(2):
            orange = local_car == 1
            position = np.asarray(
                ((-x if orange else x), (-y if orange else y), 17.0),
                dtype=np.float32,
            )
            state.car_pos[env, local_car] = (
                position * np.float32(0.02) * np.float32(50.0)
            )
            angle = np.float32(yaw + (np.pi if orange else 0.0))
            state.car_quat[env, local_car] = np.asarray(
                (0.0, 0.0, np.sin(angle * 0.5), np.cos(angle * 0.5)),
                dtype=np.float32,
            )
    state.boost.fill(np.float32(100.0 / 3.0))
    state.on_ground.fill(1)
    ball_z = np.float32(np.float32(93.15) * np.float32(0.02)) * np.float32(50.0)
    state.ball_pos[:] = np.asarray((0.0, 0.0, ball_z), dtype=np.float32)
    return state


class CompleteWorldSim(IntegratedWorldSim):
    """v0.4 complete headless standard two-Octane Soccar transition."""

    def __init__(
        self,
        *args,
        kickoff_selector: int | np.ndarray = 0,
        respawn_selector: int | np.ndarray = 0,
        auto_kickoff: bool = True,
        full_reset_interval_ticks: int = 0,
        **kwargs,
    ):
        self.defer_boost_pad_tick = True
        self._v04_kickoff_selector = kickoff_selector
        self._v04_respawn_selector = respawn_selector
        self._v04_auto_kickoff = bool(auto_kickoff)
        self._v04_full_reset_interval_ticks = int(full_reset_interval_ticks)
        if kwargs.get("initial") is None:
            num_envs = int(args[0]) if args else int(kwargs["num_envs"])
            kwargs["initial"] = make_standard_kickoff_state(
                num_envs, kickoff_selector
            )
        super().__init__(*args, **kwargs)
        self.resident_full_world_mechanics = True
        self.lifecycle = LifecycleState(
            self.num_envs,
            self.device,
            kickoff_selector=kickoff_selector,
            respawn_selector=respawn_selector,
            auto_kickoff=auto_kickoff,
            full_reset_interval_ticks=full_reset_interval_ticks,
        )

    @property
    def logical_state_bytes(self) -> int:
        lifecycle = getattr(self, "lifecycle", None)
        return super().logical_state_bytes + (
            0 if lifecycle is None else lifecycle.logical_bytes
        )

    def reset(self, state: StateSnapshot | None = None, *, seed: int | None = None) -> None:
        """Full-world reset; physical kickoff resets occur inside the GPU path."""

        if state is None:
            state = make_standard_kickoff_state(
                self.num_envs, self._v04_kickoff_selector
            )
        super().reset(state, seed=seed)
        if hasattr(self, "lifecycle"):
            self.lifecycle = LifecycleState(
                self.num_envs,
                self.device,
                kickoff_selector=self._v04_kickoff_selector,
                respawn_selector=self._v04_respawn_selector,
                auto_kickoff=self._v04_auto_kickoff,
                full_reset_interval_ticks=self._v04_full_reset_interval_ticks,
            )

    def lifecycle_snapshot(self) -> LifecycleSnapshot:
        self.synchronize()
        result = self.lifecycle.snapshot(
            pad_cooldown=self.boost_pad_cooldown,
            pad_previous_locked_car=self.boost_pad_previous_locked_car,
            car_is_demoed=self.car_car.car_is_demoed,
        )
        self.device_to_host_bytes += self.lifecycle.logical_bytes
        return result

    def request_demolition(self, local_car: int | np.ndarray) -> None:
        """Request RocketSim Car::Demolish at the next post-physics event point.

        ``local_car`` may be scalar 0/1 or one value per world. This is an
        external lifecycle command, so its small configuration transfer is
        intentionally outside timed resident stepping.
        """

        selected = np.broadcast_to(
            np.asarray(local_car, dtype=np.int32), (self.num_envs,)
        )
        if np.any((selected < 0) | (selected > 1)):
            raise ValueError("local_car entries must be 0 or 1")
        request = np.zeros((self.num_envs, 2), dtype=np.int32)
        request[np.arange(self.num_envs), selected] = 1
        self.lifecycle.demo_request = wp.array(
            request.reshape(-1), dtype=wp.int32, device=self.device
        )
        self.host_to_device_bytes += request.nbytes
        self._captured_graph = None
        self._captured_graph_ticks = 0

    def _launch_tick(self) -> None:
        lifecycle = self.lifecycle
        state = self.state
        pair = self.car_car
        wp.launch(
            lifecycle_pre_tick,
            dim=self.num_envs,
            inputs=[
                lifecycle.demo_respawn_timer,
                lifecycle.demo_held_valid,
                lifecycle.respawn_pending,
                lifecycle.respawn_event,
                lifecycle.respawn_location,
                lifecycle.respawn_selector,
                pair.car_is_demoed,
                state.car_pos,
                state.car_vel,
                state.car_quat,
                state.car_ang_vel,
                self.boost_pad_cooldown,
                lifecycle.pad_cooldown_before,
            ],
            device=self.device,
        )
        super()._launch_tick()
        wp.launch(
            boost_pad_tick_lifecycle,
            dim=self.num_envs,
            inputs=[
                self.boost_pad_positions,
                pair.pre_tick_first_car,
                state.car_pos,
                state.car_quat,
                state.boost,
                lifecycle.pad_boost_gained,
                self.boost_pad_cooldown,
                self.boost_pad_previous_locked_car,
            ],
            device=self.device,
        )
        ball = self.ball_world
        vehicle = self.vehicle
        wp.launch(
            lifecycle_post_tick,
            dim=self.num_envs,
            inputs=[
                lifecycle.world_tick,
                lifecycle.episode_tick,
                lifecycle.blue_score,
                lifecycle.orange_score,
                lifecycle.goal_scored,
                lifecycle.scoring_team,
                lifecycle.kickoff_reset,
                lifecycle.kickoff_layout,
                lifecycle.kickoff_selector,
                lifecycle.full_reset,
                lifecycle.reset_required,
                lifecycle.terminated,
                lifecycle.truncated,
                lifecycle.ball_scored_last,
                lifecycle.auto_kickoff,
                lifecycle.full_reset_interval,
                lifecycle.pad_cooldown_before,
                lifecycle.pad_pickup_car,
                lifecycle.pad_reactivated,
                lifecycle.demo_respawn_timer,
                lifecycle.demo_held_valid,
                lifecycle.demo_request,
                lifecycle.respawn_pending,
                lifecycle.respawn_location,
                lifecycle.held_float,
                lifecycle.held_int,
                pair.car_is_demoed,
                pair.car_contact_id,
                pair.car_contact_cooldown,
                state.car_pos,
                state.car_vel,
                state.car_quat,
                state.car_ang_vel,
                state.boost,
                state.boosting_time,
                state.time_since_boosted,
                state.on_ground,
                state.has_jumped,
                state.is_jumping,
                state.has_double_jumped,
                state.has_flipped,
                state.is_flipping,
                state.sticky_ticks,
                state.jump_time,
                state.air_time,
                state.air_time_since_jump,
                state.flip_time,
                state.flip_rel_torque,
                state.auto_flip_timer,
                state.auto_flip_torque_scale,
                state.is_auto_flipping,
                state.is_boosting,
                state.is_supersonic,
                state.supersonic_time,
                state.prev_throttle,
                state.prev_steer,
                state.prev_pitch,
                state.prev_yaw,
                state.prev_roll,
                state.prev_jump,
                state.prev_boost,
                state.prev_handbrake,
                state.ball_pos,
                state.ball_vel,
                state.ball_quat,
                state.ball_ang_vel,
                ball.position_bt,
                ball.velocity_bt,
                vehicle.rigid_position_bt,
                vehicle.rigid_velocity_bt,
                vehicle.solver_position,
                vehicle.solver_orientation,
                vehicle.solver_velocity,
                vehicle.solver_angular_velocity,
                self.boost_pad_cooldown,
                self.boost_pad_previous_locked_car,
            ],
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
