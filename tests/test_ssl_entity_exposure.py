import json

import pytest

from benchmarks.report_rival2_ssl_entity_exposure import CAPACITY, decompose, report, summarize


def fixture(selfplay=100, wins=7, losses=19, nexto=10000):
    return dict(
        physical_goals=selfplay + wins + losses,
        goals=selfplay + wins,
        concedes=selfplay + losses,
        resets=selfplay + wins + losses + 5,
        trainable_agent_samples=CAPACITY - nexto,
        nexto_training_sample_count=nexto,
    )


def test_exact_goal_family_conservation():
    result = decompose(fixture())
    assert result["selfplay_physical_goal_endings"] == 100
    assert result["nexto_learner_goal_endings"] == 7
    assert result["nexto_learner_concede_endings"] == 19


def test_pure_selfplay_not_claimed_as_nexto_scoring():
    result = decompose(fixture(wins=0, losses=0, nexto=0))
    assert result["nexto_learner_goal_endings"] == result["nexto_learner_concede_endings"] == 0


@pytest.mark.parametrize(
    "key,value",
    [
        ("goals", 200),
        ("physical_goals", -1),
        ("resets", 0),
        ("concedes", 1.5),
        ("nexto_training_sample_count", 0),
    ],
)
def test_invalid_ledger_rejected(key, value):
    training = fixture()
    training[key] = value
    with pytest.raises(ValueError):
        decompose(training)


def test_ratios_count_weighted_and_not_match_win_rate():
    rows = [
        dict(accepted_updates=151, training=fixture(wins=1, losses=1)),
        dict(accepted_updates=152, training=fixture(wins=9, losses=99)),
    ]
    result = summarize(rows)
    assert result["nexto_fraction_of_goal_endings_scored_by_learner"] == 10 / 110
    assert result["regulation_match_win_rate"] is None


def test_deterministic_prefix_and_missing_boundary_rejected():
    rows = [
        dict(accepted_updates=i, training=fixture(wins=0, losses=0, nexto=0))
        for i in range(101, 151)
    ]
    prefix = ("\n".join(json.dumps(r) for r in rows) + "\n").encode()
    assert report(prefix, 150) == report(prefix, 150)
    assert report(prefix, 150)["total"]["nexto_fraction_of_goal_endings_scored_by_learner"] is None
    with pytest.raises(ValueError):
        report(prefix, 200)
