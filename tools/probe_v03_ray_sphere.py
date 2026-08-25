"""One-case CUDA probe for the pinned Bullet point-vs-sphere wheel cast."""

from __future__ import annotations

import numpy as np
import warp as wp

from rivalsim.kernels.rsqrtss_amd import amd_rsqrtss_table
from rivalsim.kernels.vehicle import _bullet_ray_sphere


@wp.kernel(enable_backward=False)
def probe(
    source: wp.array(dtype=wp.vec3),
    target: wp.array(dtype=wp.vec3),
    center: wp.array(dtype=wp.vec3),
    basis: wp.array(dtype=wp.mat33),
    rsqrtss: wp.array(dtype=wp.uint16),
    fraction: wp.array(dtype=wp.float32),
    point: wp.array(dtype=wp.vec3),
    normal: wp.array(dtype=wp.vec3),
    valid: wp.array(dtype=wp.int32),
):
    fraction_value = wp.float32(0.0)
    point_value = wp.vec3(0.0, 0.0, 0.0)
    normal_value = wp.vec3(0.0, 0.0, 0.0)
    valid_value = wp.int32(0)
    _bullet_ray_sphere(
        source[0],
        target[0],
        center[0],
        basis[0],
        1.8249999284744263,
        1.0,
        rsqrtss,
        fraction_value,
        point_value,
        normal_value,
        valid_value,
    )
    fraction[0] = fraction_value
    point[0] = point_value
    normal[0] = normal_value
    valid[0] = valid_value


def main() -> None:
    wp.init()
    device = "cuda:0"
    source = np.asarray([[6.9887042, -13.4346447, 31.9942684]], dtype=np.float32)
    target = np.asarray([[6.03227282, -13.6241655, 31.9823895]], dtype=np.float32)
    center = np.asarray([[4.90817881, -14.3702126, 33.0444679]], dtype=np.float32)
    basis = np.asarray(
        [[
            [0.999898612, 0.00672400557, 0.012551602],
            [-0.00681417342, 0.999951184, 0.00715490431],
            [-0.0125028789, -0.00723970775, 0.999895632],
        ]],
        dtype=np.float32,
    )
    fraction = wp.zeros(1, dtype=wp.float32, device=device)
    point = wp.zeros(1, dtype=wp.vec3, device=device)
    normal = wp.zeros(1, dtype=wp.vec3, device=device)
    valid = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(
        probe,
        dim=1,
        inputs=[
            wp.array(source, dtype=wp.vec3, device=device),
            wp.array(target, dtype=wp.vec3, device=device),
            wp.array(center, dtype=wp.vec3, device=device),
            wp.array(basis, dtype=wp.mat33, device=device),
            wp.array(amd_rsqrtss_table(), dtype=wp.uint16, device=device),
            fraction,
            point,
            normal,
            valid,
        ],
        device=device,
    )
    actual_valid = int(valid.numpy()[0])
    actual_fraction = fraction.numpy()
    actual_point = point.numpy()
    actual_normal = normal.numpy()
    expected_fraction = np.asarray([0.849738598], dtype=np.float32)
    expected_point = np.asarray(
        [[6.17598724, -13.5956879, 31.9841728]], dtype=np.float32
    )
    expected_normal = np.asarray(
        [[0.705824256, 0.417779952, -0.572076857]], dtype=np.float32
    )
    if not (
        actual_valid == 1
        and np.array_equal(actual_fraction, expected_fraction)
        and np.array_equal(actual_point, expected_point)
        and np.array_equal(actual_normal, expected_normal)
    ):
        raise RuntimeError(
            "point-sphere cast differs from the pinned native trace: "
            f"valid={actual_valid}, fraction={actual_fraction.tolist()}, "
            f"point={actual_point.tolist()}, normal={actual_normal.tolist()}"
        )
    print("PASS: pinned B-00005 wheel-0 cast is bit-exact")


if __name__ == "__main__":
    main()
