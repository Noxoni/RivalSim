import numpy as np
import pytest

from benchmarks.summarize_direct_skills_value_residuals_v1 import residual_stats


def test_exact_prediction():
    result = residual_stats([1, 2, 3], [1, 2, 3])
    assert result['mse'] == 0 and result['explained_variance'] == 1


def test_constant_bias_is_not_hidden_by_explained_variance():
    result = residual_stats([1, 2, 3], [3, 4, 5])
    assert result['residual_mean'] == -2
    assert result['mse'] == 4 and result['explained_variance'] == 1


def test_zero_variance_target_has_no_explained_variance_claim():
    assert residual_stats([1, 1], [0, 2])['explained_variance'] is None


@pytest.mark.parametrize('target,value', [([], []), ([1, 2], [1]), ([1, np.nan], [1, 2])])
def test_bad_arrays_rejected(target, value):
    with pytest.raises(ValueError):
        residual_stats(target, value)
