from __future__ import annotations

import numpy as np

from benchmarks.evaluate_rival2_ground_to_air_selfplay_v12 import _option_summary


def test_option_summary_counts_authoritative_events_by_route() -> None:
    shape = (4, 2)
    trace = {
        "match_active": np.ones(shape, dtype=np.int16),
        "option_active": np.asarray(
            [[1, 0], [1, 1], [1, 1], [0, 1]], dtype=np.int16
        ),
        "option_activated": np.asarray(
            [[1, 0], [0, 1], [0, 0], [0, 0]], dtype=np.int16
        ),
        "option_route": np.asarray(
            [[0, -1], [0, 2], [0, 2], [-1, 2]], dtype=np.int16
        ),
        "option_contact": np.asarray(
            [[1, 0], [0, 1], [1, 1], [0, 0]], dtype=np.int16
        ),
        "option_entry_contact": np.asarray(
            [[1, 0], [0, 1], [0, 0], [0, 0]], dtype=np.int16
        ),
        "option_second_contact": np.asarray(
            [[0, 0], [0, 0], [1, 1], [0, 0]], dtype=np.int16
        ),
        "option_productive_contact": np.asarray(
            [[1, 0], [0, 1], [1, 0], [0, 0]], dtype=np.int16
        ),
        "option_goal": np.asarray(
            [[0, 0], [0, 0], [1, 0], [0, 0]], dtype=np.int16
        ),
        "option_ground_failure": np.zeros(shape, dtype=np.int16),
    }
    summary = _option_summary(trace)
    assert summary["activations"] == 2
    assert summary["contacts"] == 4
    assert summary["second_airborne_contacts"] == 2
    assert summary["goals_within_six_contacts"] == 1
    assert summary["per_route"]["assisted_low_bounce"]["contacts"] == 2
    assert summary["per_route"]["rising_double_jump"]["contacts"] == 2
