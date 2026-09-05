import numpy as np
import torch

from benchmarks.evaluate_rival2_ssl_entity_full_match import (
    assignments,
    reset_hidden,
    spec,
    summarize,
)


def test_balanced_frozen_match_protocol():
    layout, side = assignments()
    assert set(zip(layout.tolist(), side.tolist(), strict=True)) == {
        (layout_index, s) for layout_index in range(5) for s in range(2)
    }
    assert len(side) == 10
    protocol = spec()
    assert protocol["regulation_ticks"] == 36000
    assert protocol["candidate_selection"] is False
    assert protocol["action_sampling"] is False


def test_recurrent_reset_only_for_reset_lanes():
    hidden = torch.randn(1, 10, 256)
    mask = torch.arange(10) % 3 == 0
    actual = reset_hidden(hidden, mask)
    assert torch.equal(actual[:, ~mask], hidden[:, ~mask])
    assert torch.count_nonzero(actual[:, mask]) == 0


def test_match_score_perspective_and_unresolved_not_win():
    raw = {
        "match.rival_side": np.array([0, 1, 1]),
        "match.done": np.array([1, 1, 0]),
        "match.winner": np.array([0, 0, -1]),
        "match.blue_score": np.array([3, 4, 2]),
        "match.orange_score": np.array([1, 1, 2]),
        "match.total_ticks": np.array([36000, 36000, 50400]),
        "touch_count": np.array([[10, 12], [15, 0], [6, 3]]),
        "kickoff_first_touch_count": np.array([[1, 2], [3, 0], [1, 1]]),
    }
    result = summarize(raw)
    assert (result["wins"], result["losses"], result["unresolved"]) == (1, 1, 1)
    assert (result["goals_for"], result["goals_against"]) == (6, 7)
    assert result["touches"] == 13
    assert result["matches_without_rival_touch"] == 1
    assert result["no_touch_truncations"] == 0
