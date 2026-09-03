from __future__ import annotations

from benchmarks.run_rival2_offensive_demo_probe_v1 import summarize
from rivalsim.rival2_offensive_demo_v1 import ROUTE_NAMES


def _row(route: str, value: float) -> dict:
    return {
        "route": route,
        "fractions": {
            "actual_demo": value,
            "offensive_context_demo": value,
            "post_demo_touch": value / 2.0,
            "post_demo_goalward_progress": value / 3.0,
            "post_demo_goal": value / 4.0,
            "productive_conversion": value / 2.0,
            "expired_without_conversion": 1.0 - value,
        },
        "mean_offensive_context_closure_uu_per_active_tick": value * 10.0,
    }


def test_summary_preserves_routes_and_averages_sides() -> None:
    rows = [
        _row(route, value)
        for route in ROUTE_NAMES
        for value in (0.2, 0.4)
    ]
    result = summarize(rows)
    assert tuple(result) == ROUTE_NAMES
    for route in ROUTE_NAMES:
        assert result[route]["rows"] == 2
        assert abs(result[route]["actual_demo"] - 0.3) < 1.0e-7
        assert abs(result[route]["productive_conversion"] - 0.15) < 1.0e-7
        assert (
            abs(
                result[route][
                    "mean_offensive_context_closure_uu_per_active_tick"
                ]
                - 3.0
            )
            < 1.0e-7
        )


def test_summary_fails_closed_when_a_route_is_missing() -> None:
    try:
        summarize([_row(ROUTE_NAMES[0], 0.5)])
    except ValueError as error:
        assert "missing route" in str(error)
    else:
        raise AssertionError("incomplete route evidence was accepted")
