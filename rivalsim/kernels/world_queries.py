"""Warp kernels for shared static-world triangle queries."""

import warp as wp

SOCCAR_EXTENT_X = 4096.0
SOCCAR_HEIGHT = 2048.0


@wp.kernel
def raycast_mesh(
    mesh_id: wp.uint64,
    origins: wp.array(dtype=wp.vec3),
    directions: wp.array(dtype=wp.vec3),
    max_distances: wp.array(dtype=wp.float32),
    hit: wp.array(dtype=wp.int32),
    distance: wp.array(dtype=wp.float32),
    normal: wp.array(dtype=wp.vec3),
    face: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    result = wp.mesh_query_ray(
        mesh_id,
        origins[tid],
        wp.normalize(directions[tid]),
        max_distances[tid],
    )
    if result.result:
        hit[tid] = 1
        distance[tid] = result.t
        normal[tid] = result.normal
        face[tid] = result.face
    else:
        hit[tid] = 0
        distance[tid] = max_distances[tid]
        normal[tid] = wp.vec3(0.0, 0.0, 0.0)
        face[tid] = -1


@wp.func
def _try_plane(
    origin: wp.vec3,
    direction: wp.vec3,
    point: wp.vec3,
    plane_normal: wp.vec3,
    max_distance: float,
) -> float:
    denominator = wp.dot(direction, plane_normal)
    if wp.abs(denominator) > 1.0e-8:
        candidate = wp.dot(point - origin, plane_normal) / denominator
        if candidate >= 0.0 and candidate <= max_distance:
            return candidate
    return max_distance + 1.0


@wp.kernel
def raycast_soccar(
    mesh_id: wp.uint64,
    origins: wp.array(dtype=wp.vec3),
    directions: wp.array(dtype=wp.vec3),
    max_distances: wp.array(dtype=wp.float32),
    hit: wp.array(dtype=wp.int32),
    distance: wp.array(dtype=wp.float32),
    normal: wp.array(dtype=wp.vec3),
    face: wp.array(dtype=wp.int32),
):
    """Raycast exact CMF triangles and the planes RocketSim adds for Soccar."""

    tid = wp.tid()
    origin = origins[tid]
    direction = wp.normalize(directions[tid])
    maximum = max_distances[tid]
    nearest = maximum
    found = 0
    found_normal = wp.vec3(0.0, 0.0, 0.0)
    found_face = -1

    result = wp.mesh_query_ray(mesh_id, origin, direction, maximum)
    if result.result:
        found = 1
        nearest = result.t
        found_normal = result.normal
        found_face = result.face

    candidate = _try_plane(
        origin, direction, wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 1.0), nearest
    )
    if candidate <= nearest:
        found = 1
        nearest = candidate
        found_normal = wp.vec3(0.0, 0.0, 1.0)
        found_face = -2
    candidate = _try_plane(
        origin,
        direction,
        wp.vec3(0.0, 0.0, SOCCAR_HEIGHT),
        wp.vec3(0.0, 0.0, -1.0),
        nearest,
    )
    if candidate <= nearest:
        found = 1
        nearest = candidate
        found_normal = wp.vec3(0.0, 0.0, -1.0)
        found_face = -3
    candidate = _try_plane(
        origin,
        direction,
        wp.vec3(-SOCCAR_EXTENT_X, 0.0, 0.0),
        wp.vec3(1.0, 0.0, 0.0),
        nearest,
    )
    if candidate <= nearest:
        found = 1
        nearest = candidate
        found_normal = wp.vec3(1.0, 0.0, 0.0)
        found_face = -4
    candidate = _try_plane(
        origin,
        direction,
        wp.vec3(SOCCAR_EXTENT_X, 0.0, 0.0),
        wp.vec3(-1.0, 0.0, 0.0),
        nearest,
    )
    if candidate <= nearest:
        found = 1
        nearest = candidate
        found_normal = wp.vec3(-1.0, 0.0, 0.0)
        found_face = -5

    hit[tid] = found
    distance[tid] = nearest
    normal[tid] = found_normal
    face[tid] = found_face


def query_rays(
    mesh: wp.Mesh,
    origins,
    directions,
    max_distances,
    *,
    device: str,
):
    """Launch ray queries and return device-resident output arrays."""

    count = len(origins)
    origin_array = wp.array(origins, dtype=wp.vec3, device=device)
    direction_array = wp.array(directions, dtype=wp.vec3, device=device)
    max_array = wp.array(max_distances, dtype=wp.float32, device=device)
    hit = wp.empty(count, dtype=wp.int32, device=device)
    distance = wp.empty(count, dtype=wp.float32, device=device)
    normal = wp.empty(count, dtype=wp.vec3, device=device)
    face = wp.empty(count, dtype=wp.int32, device=device)
    wp.launch(
        raycast_mesh,
        dim=count,
        inputs=[mesh.id, origin_array, direction_array, max_array],
        outputs=[hit, distance, normal, face],
        device=device,
    )
    return hit, distance, normal, face


def query_soccar_rays(
    mesh: wp.Mesh,
    origins,
    directions,
    max_distances,
    *,
    device: str,
):
    """Launch full Soccar queries and return device-resident outputs."""

    count = len(origins)
    origin_array = wp.array(origins, dtype=wp.vec3, device=device)
    direction_array = wp.array(directions, dtype=wp.vec3, device=device)
    max_array = wp.array(max_distances, dtype=wp.float32, device=device)
    hit = wp.empty(count, dtype=wp.int32, device=device)
    distance = wp.empty(count, dtype=wp.float32, device=device)
    normal = wp.empty(count, dtype=wp.vec3, device=device)
    face = wp.empty(count, dtype=wp.int32, device=device)
    wp.launch(
        raycast_soccar,
        dim=count,
        inputs=[mesh.id, origin_array, direction_array, max_array],
        outputs=[hit, distance, normal, face],
        device=device,
    )
    return hit, distance, normal, face
