import pytest

from benchmarks.request_rival2_long_trace_eval20 import comparison, should_request_pause


def test_pause_requires_exact_inflight_boundary_and_correct_stack():
    stack = [
        {
            "thread_name": "MainThread",
            "frames": [
                {
                    "name": "collect_rollout",
                    "filename": "rivalsim/rival2_ssl_foundation_training.py",
                }
            ],
        }
    ]
    assert should_request_pause(19, stack, 19)
    assert not should_request_pause(18, stack, 18)
    assert not should_request_pause(19, stack, 20)
    assert not should_request_pause(20, stack, 20)
    assert not should_request_pause(19, [], 19)
    assert not should_request_pause(
        19,
        [
            {
                "thread_name": "MainThread",
                "frames": [
                    {"name": "save_checkpoint", "filename": "rivalsim/rival2_recurrent_training.py"}
                ],
            }
        ],
        19,
    )


def test_comparison_requires_same_protocol_and_reports_signed_changes():
    metrics = dict(
        goals_for=10,
        goals_against=20,
        touches_per_minute=12.0,
        no_touch_resets=5,
        goalward_touch_fraction=0.6,
        mean_speed_uu_per_second=1000.0,
    )
    before = dict(
        accepted_updates=10,
        worlds=1024,
        ticks=3600,
        deterministic_policy=True,
        opponents={"nexto": metrics},
    )
    after = {
        **before,
        "accepted_updates": 20,
        "opponents": {"nexto": {**metrics, "goals_for": 12, "goals_against": 18}},
    }
    report = comparison(before, after)["nexto"]
    assert report["goal_differential"]["delta"] == 4
    assert report["goals_against"]["delta"] == -2
    with pytest.raises(ValueError):
        comparison(before, {**after, "ticks": 128})
