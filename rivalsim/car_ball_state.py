"""GPU-resident v0.3 Octane/ball pair state and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

MAX_CAR_BALL_CONTACTS = 4


@dataclass(slots=True)
class CarBallSnapshot:
    contact_count: np.ndarray
    hit_this_tick: np.ndarray
    contact_point_a_bt: np.ndarray
    contact_point_b_bt: np.ndarray
    contact_normal: np.ndarray
    contact_distance_bt: np.ndarray
    normal_impulse: np.ndarray
    tangent_impulse: np.ndarray
    push_impulse: np.ndarray
    extra_hit_velocity_uu: np.ndarray
    relative_pos_on_ball_uu: np.ndarray
    last_extra_impulse_tick: np.ndarray


class CarBallState:
    """Persistent Bullet manifold for one active Octane/ball pair per world."""

    def __init__(self, num_envs: int, device: str):
        self.num_envs = num_envs
        self.device = device
        for name in ("contact_count", "hit_this_tick", "algorithm_active"):
            setattr(self, name, wp.zeros(num_envs, dtype=wp.int32, device=device))
        self.last_extra_impulse_tick = wp.full(
            num_envs, -1, dtype=wp.int32, device=device
        )
        for name in (
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
        ):
            setattr(self, name, wp.zeros(num_envs, dtype=wp.vec3, device=device))
        manifold_capacity = num_envs * MAX_CAR_BALL_CONTACTS
        for name in (
            "manifold_local_a_bt",
            "manifold_local_b_bt",
            "manifold_normal",
            "manifold_tangent",
        ):
            setattr(
                self,
                name,
                wp.zeros(manifold_capacity, dtype=wp.vec3, device=device),
            )
        for name in (
            "manifold_distance_bt",
            "manifold_normal_jacobian",
            "manifold_tangent_jacobian",
            "manifold_normal_rhs",
            "manifold_tangent_rhs",
            "manifold_push_rhs",
            "manifold_normal_impulse",
            "manifold_tangent_impulse",
            "manifold_push_impulse",
        ):
            setattr(
                self,
                name,
                wp.zeros(manifold_capacity, dtype=wp.float32, device=device),
            )
        self.manifold_lifetime = wp.zeros(
            manifold_capacity, dtype=wp.int32, device=device
        )
        for name in ("pre_car_quaternion", "pre_ball_quaternion"):
            setattr(self, name, wp.zeros(num_envs, dtype=wp.quat, device=device))
        for name in (
            "contact_distance_bt",
            "normal_impulse",
            "tangent_impulse",
            "push_impulse",
        ):
            setattr(self, name, wp.zeros(num_envs, dtype=wp.float32, device=device))

    @property
    def logical_bytes(self) -> int:
        # Per-world diagnostics/pre-state plus the four-point Bullet cache.
        resident = (12 * 3 + 2 * 4 + 4 + 4) * 4
        manifold = MAX_CAR_BALL_CONTACTS * (4 * 3 + 9 + 1) * 4
        return self.num_envs * (resident + manifold)

    def snapshot(self) -> CarBallSnapshot:
        def array(name: str, dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
            return np.asarray(getattr(self, name).numpy(), dtype=dtype).reshape(shape)

        count = self.num_envs
        return CarBallSnapshot(
            contact_count=array("contact_count", np.int32, (count,)),
            hit_this_tick=array("hit_this_tick", np.int32, (count,)),
            contact_point_a_bt=array("contact_point_a_bt", np.float32, (count, 3)),
            contact_point_b_bt=array("contact_point_b_bt", np.float32, (count, 3)),
            contact_normal=array("contact_normal", np.float32, (count, 3)),
            contact_distance_bt=array("contact_distance_bt", np.float32, (count,)),
            normal_impulse=array("normal_impulse", np.float32, (count,)),
            tangent_impulse=array("tangent_impulse", np.float32, (count,)),
            push_impulse=array("push_impulse", np.float32, (count,)),
            extra_hit_velocity_uu=array(
                "extra_hit_velocity_uu", np.float32, (count, 3)
            ),
            relative_pos_on_ball_uu=array(
                "relative_pos_on_ball_uu", np.float32, (count, 3)
            ),
            last_extra_impulse_tick=array(
                "last_extra_impulse_tick", np.int32, (count,)
            ),
        )
