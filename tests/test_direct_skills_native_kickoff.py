import ast
from pathlib import Path

import numpy as np
import pytest

from benchmarks.diagnose_direct_skills_native_kickoff import ARMS, OUTPUT, FIELDS, reduce_archive, source_map


def test_layout_side_mapping_not_world_index_assumption():
    np.testing.assert_array_equal(source_map(np.tile([1, 2, 3, 4, 0], 2), np.repeat([0, 1], 5)),
                                  [1, 2, 3, 4, 0, 6, 7, 8, 9, 5])


def test_invalid_layout_rejected():
    with pytest.raises(AssertionError):
        source_map(np.full(10, -1), np.zeros(10))


def test_interventions_remain_explicit_and_bounded():
    assert len(ARMS) == 10
    assert ARMS['initial_cold'] == ('initial', 0, ())
    assert ARMS['initial_warm'] == ('initial', 1, ())
    assert ARMS['postgoal_warm'] == ('postgoal', 1, ())
    for _, _, fields in ARMS.values():
        assert all(f.startswith('vehicle.') or f == 'state.car_quat' for f in fields)


def test_no_optimizer_and_lease_present():
    text = Path('benchmarks/diagnose_direct_skills_native_kickoff.py').read_text()
    tree = ast.parse(text)
    calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert 'gpu_lease' in calls and 'owned_match_stream' in calls
    assert not any('optimizer' in c.lower() or 'backward' in c.lower() for c in calls)


@pytest.fixture(scope='module')
def actual():
    with np.load(OUTPUT / 'capture/native_replay.npz', allow_pickle=False) as archive:
        return {k: archive[k] for k in archive.files}


def test_actual_reduction_is_deterministic(actual):
    import json
    assert reduce_archive(actual) == json.loads((OUTPUT / 'capture/report.json').read_text())


def test_baseline_replay_and_actual_controls_exact(actual):
    for phase, name in ((0, 'initial'), (1, 'postgoal')):
        for field in FIELDS:
            np.testing.assert_array_equal(actual[f'baseline_{phase}/{field}'], actual[f'captured_{name}/{field}'])
    np.testing.assert_array_equal(actual['initial_controls'][:, actual['source_map']], actual['postgoal_controls'])


def test_handbrake_intervention_changes_only_cars_with_residual_handbrake(actual):
    residual = actual['origin_postgoal/vehicle.handbrake_value'] != 0
    delta = abs(actual['postgoal_handbrake/car_pos'] - actual['postgoal_warm/car_pos']).reshape(32, 10, 2, 3)
    np.testing.assert_array_equal(delta.any(axis=(0, 3)), residual)
    assert residual.sum() == 5


def test_clock_only_arm_is_not_valid_warm_initialized_reference(actual):
    # No false claim that these huge deltas describe real kickoff states:
    # tick0 is responsible for filling initially-zero rigid body positions.
    assert not actual['origin_initial/vehicle.rigid_position_bt'].any()
    assert actual['origin_initial/state.car_pos'].any()
    assert abs(actual['initial_warm/car_pos'] - actual['initial_cold/car_pos']).max() > 1000


def test_archive_checkpoint_and_source_integrity():
    import hashlib
    import json
    from benchmarks.diagnose_direct_skills_native_kickoff import CHECKPOINT, ROOT, sha
    manifest = json.loads((OUTPUT / 'capture/manifest.json').read_text())
    assert sha(OUTPUT / 'capture/native_replay.npz') == manifest['archive_sha256']
    assert sha(CHECKPOINT) == manifest['checkpoint_sha256']
    for name, digest in manifest['frozen_sources'].items():
        assert hashlib.sha256((ROOT / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest().upper() == digest
