import pytest

from benchmarks.report_rival2_ssl_entity_progress import summarize_case


def case(touches, goals=0, no_touch=63, seconds=1.0):
    return dict(
        worlds=64,
        scenario_sha256="same",
        focal_touch_fraction=touches / 64,
        goals_for=goals,
        goals_against=0,
        no_touch_truncations=no_touch,
        touches_per_minute=1.0,
        median_first_touch_seconds_if_touched=seconds,
    )


def test_report_separates_projection_baseline_parent_and_conditional_time():
    row = summarize_case(case(13, 3, 61, 0.5), case(11, 3, 61, 1.0), case(16, 2, 61, 2.0))
    assert row["touch_count_change_vs_initial"] == 2
    assert row["touch_count_change_vs_parent"] == -3
    assert row["no_touch_truncations"] == 61  # Not 64-13; overlap is allowed.
    assert row["full_match_winrate_available"] is False
    assert row["median_first_touch_seconds_if_touched"] == 0.5


def test_report_refuses_changed_evaluation_corpus():
    altered = case(30)
    altered["scenario_sha256"] = "different"
    with pytest.raises(ValueError, match="identical scenario"):
        summarize_case(altered, case(11), case(16))
