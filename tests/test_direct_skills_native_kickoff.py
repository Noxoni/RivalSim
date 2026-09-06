import ast
from pathlib import Path

import numpy as np
import pytest

from benchmarks.diagnose_direct_skills_native_kickoff import ARMS, source_map


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
