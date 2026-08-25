"""GPU-resident persistent manifold storage for the v0.3 Soccar ball."""

from __future__ import annotations

import warp as wp

MAX_BALL_CONTACTS = 16
# The pinned RocketSim BVH callback has no truncation.  The frozen Phase A
# corpus reaches 155 candidates for one sphere query, so the resident buffer
# must cover the complete source witness stream.
MAX_BALL_MESH_CANDIDATES = 256


class BallWorldState:
    def __init__(self, num_envs: int, device: str):
        self.num_envs = num_envs
        self.device = device
        contact_capacity = num_envs * MAX_BALL_CONTACTS
        candidate_capacity = num_envs * MAX_BALL_MESH_CANDIDATES
        self.contact_count = wp.zeros(num_envs, dtype=wp.int32, device=device)
        self.candidate_count = wp.zeros(num_envs, dtype=wp.int32, device=device)
        self.candidate_overflow = wp.zeros(num_envs, dtype=wp.int32, device=device)
        self.contact_overflow = wp.zeros(num_envs, dtype=wp.int32, device=device)
        self.position_bt = wp.zeros(num_envs, dtype=wp.vec3, device=device)
        self.velocity_bt = wp.zeros(num_envs, dtype=wp.vec3, device=device)
        # Bullet's custom broadphase owns a cached proxy AABB independently of
        # the rigid transform. Wheel rays run before updateAabbs, so they see
        # the proxy minimum left by the preceding simulation step.
        self.broadphase_proxy_min_bt = wp.zeros(
            num_envs, dtype=wp.vec3, device=device
        )
        self.contact_local_a_bt = wp.zeros(contact_capacity, dtype=wp.vec3, device=device)
        self.contact_point_b_bt = wp.zeros(contact_capacity, dtype=wp.vec3, device=device)
        self.contact_normal = wp.zeros(contact_capacity, dtype=wp.vec3, device=device)
        self.contact_distance_bt = wp.zeros(contact_capacity, dtype=wp.float32, device=device)
        self.contact_face = wp.zeros(contact_capacity, dtype=wp.int32, device=device)
        self.contact_mesh = wp.zeros(contact_capacity, dtype=wp.int32, device=device)
        self.contact_lifetime = wp.zeros(contact_capacity, dtype=wp.int32, device=device)
        self.contact_normal_impulse = wp.zeros(
            contact_capacity, dtype=wp.float32, device=device
        )
        self.contact_tangent_impulse = wp.zeros(
            contact_capacity, dtype=wp.float32, device=device
        )
        self.contact_tangent = wp.zeros(contact_capacity, dtype=wp.vec3, device=device)
        self.contact_normal_jacobian = wp.zeros(
            contact_capacity, dtype=wp.float32, device=device
        )
        self.contact_tangent_jacobian = wp.zeros(
            contact_capacity, dtype=wp.float32, device=device
        )
        self.contact_normal_rhs = wp.zeros(
            contact_capacity, dtype=wp.float32, device=device
        )
        self.contact_tangent_rhs = wp.zeros(
            contact_capacity, dtype=wp.float32, device=device
        )
        self.contact_push_rhs = wp.zeros(
            contact_capacity, dtype=wp.float32, device=device
        )
        self.contact_push_impulse = wp.zeros(
            contact_capacity, dtype=wp.float32, device=device
        )
        self.candidate_face = wp.zeros(candidate_capacity, dtype=wp.int32, device=device)

    @property
    def logical_bytes(self) -> int:
        contact_scalars = 13
        contact_vectors = 4
        return (
            self.num_envs * (4 * 4 + 3 * 12)
            + self.num_envs * MAX_BALL_CONTACTS * (contact_scalars * 4 + contact_vectors * 12)
            + self.num_envs * MAX_BALL_MESH_CANDIDATES * 4
        )
