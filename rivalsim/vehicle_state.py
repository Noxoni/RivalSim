"""Flattened device-resident v0.2 wheel, suspension, and contact state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

WHEELS_PER_CAR = 4
MAX_CONTACTS_PER_CAR = 4


@dataclass(slots=True)
class VehicleSnapshot:
    solver_position: np.ndarray
    rigid_position_bt: np.ndarray
    solver_orientation: np.ndarray
    solver_velocity: np.ndarray
    rigid_velocity_bt: np.ndarray
    solver_angular_velocity: np.ndarray
    auto_roll_acceleration: np.ndarray
    auto_roll_angular_acceleration: np.ndarray
    wheel_ray_start: np.ndarray
    wheel_direction: np.ndarray
    wheel_hit_point: np.ndarray
    wheel_hit_normal: np.ndarray
    wheel_hit_distance: np.ndarray
    wheel_hit_face: np.ndarray
    suspension_length: np.ndarray
    suspension_velocity: np.ndarray
    suspension_clipped_factor: np.ndarray
    suspension_force: np.ndarray
    suspension_pushback: np.ndarray
    wheel_axle: np.ndarray
    wheel_forward: np.ndarray
    wheel_friction_impulse: np.ndarray
    side_impulse: np.ndarray
    rolling_impulse: np.ndarray
    engine_acceleration: np.ndarray
    brake_acceleration: np.ndarray
    steer_angle: np.ndarray
    lateral_friction: np.ndarray
    longitudinal_friction: np.ndarray
    wheel_contact: np.ndarray
    wheel_world_contact: np.ndarray
    handbrake_value: np.ndarray
    wheels_with_contact: np.ndarray
    candidate_count: np.ndarray
    contact_count: np.ndarray
    world_contact_normal: np.ndarray
    candidate_total: np.ndarray
    contact_total: np.ndarray
    candidate_max: np.ndarray
    contact_max: np.ndarray
    penetration_max: np.ndarray
    contact_point: np.ndarray
    contact_normal: np.ndarray
    contact_tangent: np.ndarray
    contact_face: np.ndarray
    contact_distance: np.ndarray
    contact_penetration: np.ndarray
    contact_normal_jacobian: np.ndarray
    contact_tangent_jacobian: np.ndarray
    contact_normal_rhs: np.ndarray
    contact_tangent_rhs: np.ndarray
    contact_normal_impulse: np.ndarray
    contact_tangent_impulse: np.ndarray
    contact_push_impulse: np.ndarray
    contact_lifetime: np.ndarray


class VehicleState:
    """State-of-arrays allocation shared by Gate B and Gate C kernels."""

    def __init__(self, car_count: int, device: str):
        if car_count <= 0:
            raise ValueError("car_count must be positive")
        self.car_count = car_count
        self.wheel_count = car_count * WHEELS_PER_CAR
        self.contact_capacity = car_count * MAX_CONTACTS_PER_CAR
        self.device = device

        self.solver_position = wp.zeros(car_count, dtype=wp.vec3, device=device)
        self.rigid_position_bt = wp.zeros(car_count, dtype=wp.vec3, device=device)
        self.solver_orientation = wp.zeros(car_count, dtype=wp.quat, device=device)
        self.solver_velocity = wp.zeros(car_count, dtype=wp.vec3, device=device)
        self.rigid_velocity_bt = wp.zeros(car_count, dtype=wp.vec3, device=device)
        self.solver_angular_velocity = wp.zeros(car_count, dtype=wp.vec3, device=device)
        self.auto_roll_acceleration = wp.zeros(car_count, dtype=wp.vec3, device=device)
        self.auto_roll_angular_acceleration = wp.zeros(
            car_count, dtype=wp.vec3, device=device
        )

        for name in (
            "wheel_ray_start",
            "wheel_direction",
            "wheel_hit_point",
            "wheel_hit_normal",
            "wheel_axle",
            "wheel_forward",
            "wheel_friction_impulse",
        ):
            setattr(self, name, wp.zeros(self.wheel_count, dtype=wp.vec3, device=device))
        for name in (
            "wheel_hit_distance",
            "suspension_length",
            "suspension_velocity",
            "suspension_clipped_factor",
            "suspension_force",
            "suspension_pushback",
            "side_impulse",
            "rolling_impulse",
            "engine_acceleration",
            "brake_acceleration",
            "steer_angle",
            "lateral_friction",
            "longitudinal_friction",
        ):
            setattr(self, name, wp.zeros(self.wheel_count, dtype=wp.float32, device=device))
        self.wheel_contact = wp.zeros(self.wheel_count, dtype=wp.int32, device=device)
        self.wheel_world_contact = wp.zeros(self.wheel_count, dtype=wp.int32, device=device)
        self.wheel_hit_face = wp.full(self.wheel_count, -1, dtype=wp.int32, device=device)
        self.handbrake_value = wp.zeros(car_count, dtype=wp.float32, device=device)
        self.wheels_with_contact = wp.zeros(car_count, dtype=wp.int32, device=device)
        self.candidate_count = wp.zeros(car_count, dtype=wp.int32, device=device)
        self.contact_count = wp.zeros(car_count, dtype=wp.int32, device=device)
        self.world_contact_normal = wp.zeros(car_count, dtype=wp.vec3, device=device)
        self.candidate_total = wp.zeros(car_count, dtype=wp.float32, device=device)
        self.contact_total = wp.zeros(car_count, dtype=wp.float32, device=device)
        self.candidate_max = wp.zeros(car_count, dtype=wp.int32, device=device)
        self.contact_max = wp.zeros(car_count, dtype=wp.int32, device=device)
        self.penetration_max = wp.zeros(car_count, dtype=wp.float32, device=device)
        self.contact_point = wp.zeros(self.contact_capacity, dtype=wp.vec3, device=device)
        self.contact_normal = wp.zeros(self.contact_capacity, dtype=wp.vec3, device=device)
        self.contact_tangent = wp.zeros(self.contact_capacity, dtype=wp.vec3, device=device)
        self.contact_face = wp.full(
            self.contact_capacity, -1, dtype=wp.int32, device=device
        )
        for name in (
            "contact_distance",
            "contact_penetration",
            "contact_normal_jacobian",
            "contact_tangent_jacobian",
            "contact_normal_rhs",
            "contact_tangent_rhs",
            "contact_normal_impulse",
            "contact_tangent_impulse",
            "contact_push_impulse",
        ):
            setattr(
                self,
                name,
                wp.zeros(self.contact_capacity, dtype=wp.float32, device=device),
            )
        self.contact_lifetime = wp.zeros(self.contact_capacity, dtype=wp.int32, device=device)
        self.plane_support_direction = wp.zeros(
            car_count * 12, dtype=wp.float32, device=device
        )

    @property
    def logical_bytes(self) -> int:
        vec3_count = self.wheel_count * 7 + self.contact_capacity * 3 + self.car_count * 8
        quat_count = self.car_count
        float_count = self.wheel_count * 14 + self.car_count * 16 + self.contact_capacity * 9
        int_count = self.wheel_count * 3 + self.car_count * 5 + self.contact_capacity * 2
        return (vec3_count * 3 + quat_count * 4 + float_count + int_count) * 4

    def snapshot(self) -> VehicleSnapshot:
        car_count = self.car_count
        wheel_shape = (car_count, WHEELS_PER_CAR)
        wheel_vec_shape = (car_count, WHEELS_PER_CAR, 3)
        contact_shape = (car_count, MAX_CONTACTS_PER_CAR)
        contact_vec_shape = (car_count, MAX_CONTACTS_PER_CAR, 3)

        def array(name: str, dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
            return np.asarray(getattr(self, name).numpy(), dtype=dtype).reshape(shape)

        return VehicleSnapshot(
            solver_position=array("solver_position", np.float32, (car_count, 3)),
            rigid_position_bt=array("rigid_position_bt", np.float32, (car_count, 3)),
            solver_orientation=array("solver_orientation", np.float32, (car_count, 4)),
            solver_velocity=array("solver_velocity", np.float32, (car_count, 3)),
            rigid_velocity_bt=array("rigid_velocity_bt", np.float32, (car_count, 3)),
            solver_angular_velocity=array(
                "solver_angular_velocity", np.float32, (car_count, 3)
            ),
            auto_roll_acceleration=array(
                "auto_roll_acceleration", np.float32, (car_count, 3)
            ),
            auto_roll_angular_acceleration=array(
                "auto_roll_angular_acceleration", np.float32, (car_count, 3)
            ),
            wheel_ray_start=array("wheel_ray_start", np.float32, wheel_vec_shape),
            wheel_direction=array("wheel_direction", np.float32, wheel_vec_shape),
            wheel_hit_point=array("wheel_hit_point", np.float32, wheel_vec_shape),
            wheel_hit_normal=array("wheel_hit_normal", np.float32, wheel_vec_shape),
            wheel_hit_distance=array("wheel_hit_distance", np.float32, wheel_shape),
            wheel_hit_face=array("wheel_hit_face", np.int32, wheel_shape),
            suspension_length=array("suspension_length", np.float32, wheel_shape),
            suspension_velocity=array("suspension_velocity", np.float32, wheel_shape),
            suspension_clipped_factor=array("suspension_clipped_factor", np.float32, wheel_shape),
            suspension_force=array("suspension_force", np.float32, wheel_shape),
            suspension_pushback=array("suspension_pushback", np.float32, wheel_shape),
            wheel_axle=array("wheel_axle", np.float32, wheel_vec_shape),
            wheel_forward=array("wheel_forward", np.float32, wheel_vec_shape),
            wheel_friction_impulse=array(
                "wheel_friction_impulse", np.float32, wheel_vec_shape
            ),
            side_impulse=array("side_impulse", np.float32, wheel_shape),
            rolling_impulse=array("rolling_impulse", np.float32, wheel_shape),
            engine_acceleration=array("engine_acceleration", np.float32, wheel_shape),
            brake_acceleration=array("brake_acceleration", np.float32, wheel_shape),
            steer_angle=array("steer_angle", np.float32, wheel_shape),
            lateral_friction=array("lateral_friction", np.float32, wheel_shape),
            longitudinal_friction=array("longitudinal_friction", np.float32, wheel_shape),
            wheel_contact=array("wheel_contact", np.int32, wheel_shape),
            wheel_world_contact=array("wheel_world_contact", np.int32, wheel_shape),
            handbrake_value=array("handbrake_value", np.float32, (car_count,)),
            wheels_with_contact=array("wheels_with_contact", np.int32, (car_count,)),
            candidate_count=array("candidate_count", np.int32, (car_count,)),
            contact_count=array("contact_count", np.int32, (car_count,)),
            world_contact_normal=array(
                "world_contact_normal", np.float32, (car_count, 3)
            ),
            candidate_total=array("candidate_total", np.float32, (car_count,)),
            contact_total=array("contact_total", np.float32, (car_count,)),
            candidate_max=array("candidate_max", np.int32, (car_count,)),
            contact_max=array("contact_max", np.int32, (car_count,)),
            penetration_max=array("penetration_max", np.float32, (car_count,)),
            contact_point=array("contact_point", np.float32, contact_vec_shape),
            contact_normal=array("contact_normal", np.float32, contact_vec_shape),
            contact_tangent=array("contact_tangent", np.float32, contact_vec_shape),
            contact_face=array("contact_face", np.int32, contact_shape),
            contact_distance=array("contact_distance", np.float32, contact_shape),
            contact_penetration=array("contact_penetration", np.float32, contact_shape),
            contact_normal_jacobian=array(
                "contact_normal_jacobian", np.float32, contact_shape
            ),
            contact_tangent_jacobian=array(
                "contact_tangent_jacobian", np.float32, contact_shape
            ),
            contact_normal_rhs=array("contact_normal_rhs", np.float32, contact_shape),
            contact_tangent_rhs=array("contact_tangent_rhs", np.float32, contact_shape),
            contact_normal_impulse=array(
                "contact_normal_impulse", np.float32, contact_shape
            ),
            contact_tangent_impulse=array(
                "contact_tangent_impulse", np.float32, contact_shape
            ),
            contact_push_impulse=array("contact_push_impulse", np.float32, contact_shape),
            contact_lifetime=array("contact_lifetime", np.int32, contact_shape),
        )
