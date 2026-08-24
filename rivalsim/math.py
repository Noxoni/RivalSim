"""FP32 quaternion/vector helpers shared by tests and the CPU reference."""

from __future__ import annotations

import numpy as np


def quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Hamilton product for quaternions stored as ``(x, y, z, w)``."""

    lx, ly, lz, lw = np.moveaxis(left, -1, 0)
    rx, ry, rz, rw = np.moveaxis(right, -1, 0)
    result = np.empty(np.broadcast_shapes(left.shape, right.shape), dtype=np.float32)
    result[..., 0] = lw * rx + lx * rw + ly * rz - lz * ry
    result[..., 1] = lw * ry - lx * rz + ly * rw + lz * rx
    result[..., 2] = lw * rz + lx * ry - ly * rx + lz * rw
    result[..., 3] = lw * rw - lx * rx - ly * ry - lz * rz
    return result


def quat_rotate(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate local vectors into world space."""

    xyz = quat[..., :3]
    vector = np.asarray(vector, dtype=np.float32)
    t = np.float32(2.0) * np.cross(xyz, vector)
    return (vector + quat[..., 3:4] * t + np.cross(xyz, t)).astype(np.float32)


def integrate_quaternion(quat: np.ndarray, ang_vel: np.ndarray, dt: np.float32) -> np.ndarray:
    """Bullet-style world-angular-velocity exponential-map integration."""

    angle = np.linalg.norm(ang_vel, axis=-1).astype(np.float32)
    limited = np.minimum(angle, np.float32((np.pi / 4.0) / float(dt)))
    angle_sq = limited * limited
    small = limited < np.float32(0.001)
    scale = np.empty_like(limited)
    scale[small] = np.float32(0.5) * dt - (
        dt * dt * dt * np.float32(0.020833333333) * angle_sq[small]
    )
    nonzero = ~small
    scale[nonzero] = np.sin(np.float32(0.5) * limited[nonzero] * dt) / angle[nonzero]
    dq = np.empty_like(quat)
    dq[..., :3] = ang_vel * scale[..., None]
    dq[..., 3] = np.cos(np.float32(0.5) * limited * dt)
    result = quat_multiply(dq, quat)
    norm = np.linalg.norm(result, axis=-1, keepdims=True)
    result /= np.maximum(norm, np.float32(1e-20))
    return result.astype(np.float32, copy=False)


def cap_vectors(vectors: np.ndarray, maximum: np.float32) -> None:
    norm_sq = np.sum(vectors * vectors, axis=-1)
    mask = norm_sq > maximum * maximum
    if np.any(mask):
        vectors[mask] *= (maximum / np.sqrt(norm_sq[mask]))[..., None]


def quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Return column-vector rotation matrices with local X/Y/Z as columns."""

    x, y, z, w = np.moveaxis(np.asarray(quat, dtype=np.float32), -1, 0)
    result = np.empty((*quat.shape[:-1], 3, 3), dtype=np.float32)
    result[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    result[..., 0, 1] = 2.0 * (x * y - z * w)
    result[..., 0, 2] = 2.0 * (x * z + y * w)
    result[..., 1, 0] = 2.0 * (x * y + z * w)
    result[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    result[..., 1, 2] = 2.0 * (y * z - x * w)
    result[..., 2, 0] = 2.0 * (x * z - y * w)
    result[..., 2, 1] = 2.0 * (y * z + x * w)
    result[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return result


def matrix_to_quat(matrix: np.ndarray) -> np.ndarray:
    """Convert one right-handed column-basis matrix to normalized ``(x, y, z, w)``."""

    source = np.asarray(matrix, dtype=np.float64)
    if source.shape != (3, 3) or not np.isfinite(source).all():
        raise ValueError("rotation matrix must be one finite 3x3 array")
    trace = float(np.trace(source))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quat = np.asarray(
            (
                (source[2, 1] - source[1, 2]) / scale,
                (source[0, 2] - source[2, 0]) / scale,
                (source[1, 0] - source[0, 1]) / scale,
                0.25 * scale,
            ),
            dtype=np.float64,
        )
    elif source[0, 0] > source[1, 1] and source[0, 0] > source[2, 2]:
        scale = np.sqrt(1.0 + source[0, 0] - source[1, 1] - source[2, 2]) * 2.0
        quat = np.asarray(
            (
                0.25 * scale,
                (source[0, 1] + source[1, 0]) / scale,
                (source[0, 2] + source[2, 0]) / scale,
                (source[2, 1] - source[1, 2]) / scale,
            ),
            dtype=np.float64,
        )
    elif source[1, 1] > source[2, 2]:
        scale = np.sqrt(1.0 + source[1, 1] - source[0, 0] - source[2, 2]) * 2.0
        quat = np.asarray(
            (
                (source[0, 1] + source[1, 0]) / scale,
                0.25 * scale,
                (source[1, 2] + source[2, 1]) / scale,
                (source[0, 2] - source[2, 0]) / scale,
            ),
            dtype=np.float64,
        )
    else:
        scale = np.sqrt(1.0 + source[2, 2] - source[0, 0] - source[1, 1]) * 2.0
        quat = np.asarray(
            (
                (source[0, 2] + source[2, 0]) / scale,
                (source[1, 2] + source[2, 1]) / scale,
                0.25 * scale,
                (source[1, 0] - source[0, 1]) / scale,
            ),
            dtype=np.float64,
        )
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-12:
        raise ValueError("rotation matrix produced a zero quaternion")
    quat /= norm
    if quat[3] < 0.0:
        quat *= -1.0
    return quat.astype(np.float32)


def bullet_matrix_to_quat(matrix: np.ndarray) -> np.ndarray:
    """Mirror pinned Bullet ``btMatrix3x3::getRotation`` SIMD order in FP32.

    The pinned RocketSim binary's traced rigid-body quaternions select Bullet's
    SIMD branch. Unlike the scalar branch, it multiplies all four unscaled
    components by ``0.5 / sqrt(x)``. The source does not normalize the result;
    keeping its branch comparisons and rounding order is necessary when a
    matrix-only binding readback must recover the rigid body's quaternion.
    """

    source = np.asarray(matrix, dtype=np.float32)
    if source.shape != (3, 3) or not np.isfinite(source).all():
        raise ValueError("rotation matrix must be one finite 3x3 array")

    def rn(value: float | np.float32) -> np.float32:
        return np.float32(value)

    trace = rn(rn(source[0, 0] + source[1, 1]) + source[2, 2])
    result = np.empty(4, dtype=np.float32)
    if trace > rn(0.0):
        scale_squared = rn(trace + rn(1.0))
        scale = rn(np.sqrt(scale_squared))
        inverse_scale = rn(rn(0.5) / scale)
        result[0] = rn(rn(source[2, 1] - source[1, 2]) * inverse_scale)
        result[1] = rn(rn(source[0, 2] - source[2, 0]) * inverse_scale)
        result[2] = rn(rn(source[1, 0] - source[0, 1]) * inverse_scale)
        result[3] = rn(scale_squared * inverse_scale)
    else:
        if source[0, 0] < source[1, 1]:
            index = 2 if source[1, 1] < source[2, 2] else 1
        else:
            index = 2 if source[0, 0] < source[2, 2] else 0
        next_index = (index + 1) % 3
        final_index = (index + 2) % 3
        scale_squared = rn(
            rn(
                rn(
                    rn(source[index, index] - source[next_index, next_index])
                    - source[final_index, final_index]
                )
                + rn(1.0)
            )
        )
        scale = rn(np.sqrt(scale_squared))
        inverse_scale = rn(rn(0.5) / scale)
        result[3] = rn(
            rn(source[final_index, next_index] - source[next_index, final_index]) * inverse_scale
        )
        result[next_index] = rn(
            rn(source[next_index, index] + source[index, next_index]) * inverse_scale
        )
        result[final_index] = rn(
            rn(source[final_index, index] + source[index, final_index]) * inverse_scale
        )
        result[index] = rn(scale_squared * inverse_scale)
    return result


def orientation_error_radians(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Shortest orientation difference angle between quaternion arrays."""

    dots = np.abs(np.sum(left * right, axis=-1))
    return (np.float32(2.0) * np.arccos(np.clip(dots, 0.0, 1.0))).astype(np.float32)
