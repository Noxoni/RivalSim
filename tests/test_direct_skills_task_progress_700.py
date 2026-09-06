import pytest

from benchmarks.summarize_direct_skills_task_progress_700 import COUNTS, aggregate


def record(**values):
    return dict.fromkeys(COUNTS, 0) | values


def test_thirty_hz_exposure_and_pooled_rates():
    result = aggregate([record(samples=900, touches=20), record(samples=2700, touches=10)])
    assert result['learner_minutes'] == 2
    assert result['per_learner_minute']['touches'] == 15
    assert result['success_fraction_of_observed_endings'] is None


def test_observed_endings_and_events_are_not_same_cohort():
    result = aggregate([record(samples=1800, first_touch=10, episode_endings=3,
                               goals=1, concedes=1, success_endings=2)])
    assert result['totals']['first_touch'] == 10
    assert result['success_fraction_of_observed_endings'] == 2/3


@pytest.mark.parametrize('bad', [-1, .5, float('nan'), float('inf')])
def test_reject_invalid_counts(bad):
    with pytest.raises(ValueError, match='Invalid count'):
        aggregate([record(samples=1800, touches=bad)])


def test_reject_missing_exposure_or_inconsistent_goals():
    with pytest.raises(ValueError, match='No records'):
        aggregate([])
    with pytest.raises(ValueError, match='No learner exposure'):
        aggregate([record()])
    with pytest.raises(ValueError, match='More terminal goals'):
        aggregate([record(samples=1800, goals=2, episode_endings=1)])
