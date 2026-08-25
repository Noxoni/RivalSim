"""Pinned Bullet 3.24 Octane-child-box/standard-sphere narrowphase.

The authority dispatches the Octane compound's one child through
``btConvexConvexAlgorithm`` rather than Bullet's optional sphere/box helper.
This module specializes the already validated GJK/Voronoi/EPA translation to
that exact box-core/sphere-core pair. It is intentionally not a generic
convex-convex dispatcher.
"""

import warp as wp

from rivalsim.kernels.bullet_box_triangle import (
    _BULLET_BOX_TRIANGLE_CLOSEST,
    _BULLET_BOX_TRIANGLE_PENETRATION,
)

# RocketSim constructs the standard-ball radius in UU and converts it through
# its float32 BT scale (91.25f * 0.02f).  The resulting source value is one ULP
# below the directly rounded ``1.825f`` literal.
BALL_RADIUS_BT = 1.8249999284744263


def _replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"box/sphere source specialization expected one match: {old[:80]!r}")
    return source.replace(old, new)


_CLOSEST_TRIANGLE_DECL = """    const BtPairV3 triangle[3] = {
        pair_make(v0_bt[0], v0_bt[1], v0_bt[2]),
        pair_make(v1_bt[0], v1_bt[1], v1_bt[2]),
        pair_make(v2_bt[0], v2_bt[1], v2_bt[2]),
    };"""
_CLOSEST_SPHERE_DECL = """    const BtPairV3 sphere_center = pair_make(
        sphere_center_bt[0], sphere_center_bt[1], sphere_center_bt[2]);
    // btSphereShape::localGetSupportingVertexWithoutMarginNonVirtual returns
    // its zero implicit core. Keep the three aliases only because the shared
    // AABB-center orientation code consumes three support entries.
    const BtPairV3 triangle[3] = {
        pair_make(0.0f, 0.0f, 0.0f),
        pair_make(0.0f, 0.0f, 0.0f),
        pair_make(0.0f, 0.0f, 0.0f),
    };"""

_BULLET_BOX_SPHERE_CLOSEST = _replace_once(
    _BULLET_BOX_TRIANGLE_CLOSEST, _CLOSEST_TRIANGLE_DECL, _CLOSEST_SPHERE_DECL
)
_BULLET_BOX_SPHERE_CLOSEST = _replace_once(
    _BULLET_BOX_SPHERE_CLOSEST,
    "    const BtPairV3 position_offset = pair_vmul(center, 0.5f);",
    "    const BtPairV3 position_offset = pair_vmul(pair_vadd(center, sphere_center), 0.5f);",
)
_BULLET_BOX_SPHERE_CLOSEST = _replace_once(
    _BULLET_BOX_SPHERE_CLOSEST,
    "    const BtPairV3 local_origin_b = pair_vneg(position_offset);",
    "    const BtPairV3 local_origin_b = pair_vsub(sphere_center, position_offset);",
)
_BULLET_BOX_SPHERE_CLOSEST = _replace_once(
    _BULLET_BOX_SPHERE_CLOSEST,
    "    const float margin_b = 0.0f;",
    f"    const float margin_b = {BALL_RADIUS_BT:.9g}f;",
)


@wp.func_native(_BULLET_BOX_SPHERE_CLOSEST)
def bullet_box_sphere_closest(
    body_origin_bt: wp.vec3,
    basis: wp.mat33,
    sphere_center_bt: wp.vec3,
    point_a_bt: wp.ref[wp.vec3],
    point_b_bt: wp.ref[wp.vec3],
    normal_world: wp.ref[wp.vec3],
    distance_bt: wp.ref[wp.float32],
    valid: wp.ref[wp.int32],
    degenerate_status: wp.ref[wp.int32],
): ...


_PENETRATION_TRIANGLE_DECL = """    const BtV3 triangle_world[3] = {
        bt_make(v0_bt[0], v0_bt[1], v0_bt[2]),
        bt_make(v1_bt[0], v1_bt[1], v1_bt[2]),
        bt_make(v2_bt[0], v2_bt[1], v2_bt[2]),
    };"""
_PENETRATION_SPHERE_DECL = """    const BtV3 sphere_center = bt_make(
        sphere_center_bt[0], sphere_center_bt[1], sphere_center_bt[2]);
    const BtV3 triangle_world[3] = {
        bt_make(0.0f, 0.0f, 0.0f),
        bt_make(0.0f, 0.0f, 0.0f),
        bt_make(0.0f, 0.0f, 0.0f),
    };"""
_BULLET_BOX_SPHERE_PENETRATION = _replace_once(
    _BULLET_BOX_TRIANGLE_PENETRATION,
    _PENETRATION_TRIANGLE_DECL,
    _PENETRATION_SPHERE_DECL,
)
_BULLET_BOX_SPHERE_PENETRATION = _replace_once(
    _BULLET_BOX_SPHERE_PENETRATION,
    "    const BtV3 position_offset = bt_vmul(center, 0.5f);",
    "    const BtV3 position_offset = bt_vmul(bt_vadd(center, sphere_center), 0.5f);",
)
_BULLET_BOX_SPHERE_PENETRATION = _replace_once(
    _BULLET_BOX_SPHERE_PENETRATION,
    "    const BtV3 local_origin_b = bt_vsub(bt_make(0.0f, 0.0f, 0.0f), position_offset);",
    "    const BtV3 local_origin_b = bt_vsub(sphere_center, position_offset);",
)
_BULLET_BOX_SPHERE_PENETRATION = _replace_once(
    _BULLET_BOX_SPHERE_PENETRATION,
    "    const float box_margin = 0.0386590995f;",
    f"    const float box_margin = 0.0386590995f;\n    const float sphere_margin = {BALL_RADIUS_BT:.9g}f;",
)
_BULLET_BOX_SPHERE_PENETRATION = _replace_once(
    _BULLET_BOX_SPHERE_PENETRATION,
    """        return bt_triangle_support_world_core(normalized_direction);
    };""",
    """        const BtV3 core = bt_triangle_support_world_core(normalized_direction);
        return bt_vadd(core, bt_vmul(normalized_direction, sphere_margin));
    };""",
)
_BULLET_BOX_SPHERE_PENETRATION = _replace_once(
    _BULLET_BOX_SPHERE_PENETRATION,
    """    BtV3 guesses[9] = {
        safe_normalized(bt_vneg(center)),
        safe_normalized(center),""",
    """    BtV3 guesses[9] = {
        safe_normalized(bt_vsub(sphere_center, center)),
        safe_normalized(bt_vsub(center, sphere_center)),""",
)
_BULLET_BOX_SPHERE_PENETRATION = _replace_once(
    _BULLET_BOX_SPHERE_PENETRATION,
    """                        box_margin);
                    BtV3 pair_normal = bt_sse_normalized(fallback_axis);""",
    """                        bt_add(box_margin, sphere_margin));
                    BtV3 pair_normal = bt_sse_normalized(fallback_axis);""",
)


@wp.func_native(_BULLET_BOX_SPHERE_PENETRATION)
def bullet_box_sphere_penetration(
    body_origin_bt: wp.vec3,
    basis: wp.mat33,
    sphere_center_bt: wp.vec3,
    point_a_bt: wp.ref[wp.vec3],
    point_b_bt: wp.ref[wp.vec3],
    normal_world: wp.ref[wp.vec3],
    distance_bt: wp.ref[wp.float32],
    valid: wp.ref[wp.int32],
): ...
