import numpy as np
import pytest
import torch

from benchmarks.report_rival2_direct_skills_kickoff_inputs import paired_indices, restore_wheel_inputs


def test_wheel_intervention_only_changes_eight_copied_fields():
    observations = torch.arange(364).reshape(2, 182).float()
    initial = -observations
    original = observations.clone()
    result, columns = restore_wheel_inputs(observations, initial)
    other = [i for i in range(182) if i not in columns]
    assert len(columns) == 8
    assert torch.equal(result[:, columns], initial[:, columns])
    assert torch.equal(result[:, other], original[:, other])
    assert torch.equal(observations, original)


def test_matching_uses_layout_side_and_only_age_zero():
    a = dict(eligible=np.ones((3, 2), dtype=bool),
             episode_ticks=np.array([[0, 0], [4, 4], [0, 0]]),
             rival_side=np.array([[0, 1]] * 3),
             match_goal_count=np.array([[0, 0], [0, 0], [1, 1]]),
             match_starting_layout=np.array([[3, 4]] * 3),
             lifecycle_layout=np.array([[3, 4]] * 3))
    assert paired_indices(a) == [((0, 0, 0), (2, 0, 0)), ((0, 1, 1), (2, 1, 1))]
    a['lifecycle_layout'][2, 0] = 4
    with pytest.raises(KeyError):
        paired_indices(a)


def test_no_later_samples_fail_closed():
    a = {key: np.zeros((1, 1), dtype=int) for key in (
        'episode_ticks', 'rival_side', 'match_goal_count', 'match_starting_layout', 'lifecycle_layout')}
    a['eligible'] = np.ones((1, 1), dtype=bool)
    with pytest.raises(ValueError, match='no post-goal'):
        paired_indices(a)
