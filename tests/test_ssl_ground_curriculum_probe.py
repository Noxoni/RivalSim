import numpy as np

from rivalsim.fresh_ground_30hz import scenario_hash, scenarios
from rivalsim.ssl_ground_curriculum_probe import PROBE_SEED, probe_scenarios


def test_determinism_untouched_fraction_and_physical_geometry():
    n = 1000
    a, b = probe_scenarios(n), probe_scenarios(n)
    assert scenario_hash(a) == scenario_hash(b)
    original = scenarios(n, PROBE_SEED)
    order = np.random.default_rng(PROBE_SEED + 1).permutation(n)
    changed, unchanged = order[:800], order[800:]
    for name in original.state.__dataclass_fields__:
        np.testing.assert_array_equal(
            getattr(a.state, name)[unchanged], getattr(original.state, name)[unchanged]
        )
    for name in ("family", "focal_side", "kickoff_indicator", "kickoff_layout"):
        np.testing.assert_array_equal(
            getattr(a, name)[unchanged], getattr(original, name)[unchanged]
        )
    side = a.focal_side[changed].astype(int)
    other = 1 - side
    pos = a.state.car_pos[changed, side]
    delta = a.state.ball_pos[changed, :2] - pos[:, :2]
    distance = np.linalg.norm(delta, axis=-1)
    assert distance.min() >= 349.99 and distance.max() <= 850.01
    assert (
        np.linalg.norm(a.state.car_pos[changed, other, :2] - a.state.ball_pos[changed, :2], axis=-1)
        > 1500
    ).all()
    for car in (0, 1):
        q = a.state.car_quat[changed, car]
        np.testing.assert_allclose(np.linalg.norm(q, axis=-1), 1, atol=1e-6)
        yaw = 2 * np.arctan2(q[:, 2], q[:, 3])
        forward = np.stack([np.cos(yaw), np.sin(yaw)], axis=-1)
        vel = a.state.car_vel[changed, car, :2]
        assert (np.sum(forward * vel, axis=-1) >= -1e-3).all()
        assert np.max(np.abs(forward[:, 0] * vel[:, 1] - forward[:, 1] * vel[:, 0])) < 1e-3
    yaw = 2 * np.arctan2(a.state.car_quat[changed, side, 2], a.state.car_quat[changed, side, 3])
    error = np.arctan2(
        np.sin(yaw - np.arctan2(delta[:, 1], delta[:, 0])),
        np.cos(yaw - np.arctan2(delta[:, 1], delta[:, 0])),
    )
    assert np.max(np.abs(error[::2])) <= 0.30001
    assert np.max(np.abs(error[1::2])) <= 0.70001
    assert np.max(np.abs(error)) > 0.6
    assert (a.kickoff_indicator[changed] == 0).all()
    assert (a.state.on_ground[changed] == 1).all()
    a.state.validate()
