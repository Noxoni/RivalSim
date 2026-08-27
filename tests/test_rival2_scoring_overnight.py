from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from benchmarks.run_rival2_scoring_overnight import (
    AUTHORIZED_DEADLINE_LOCAL,
    crosses_deadline,
    is_snapshot_iteration,
    parse_authorized_deadline,
)


def test_authorized_deadline_is_explicit_eastern_time() -> None:
    deadline = parse_authorized_deadline(AUTHORIZED_DEADLINE_LOCAL)
    assert deadline.isoformat() == "2026-08-27T07:00:00-04:00"
    assert deadline.utcoffset() == timedelta(hours=-4)


@pytest.mark.parametrize(
    "value",
    (
        "2026-08-27T07:00:00",
        "2026-08-27T07:00:01-04:00",
        "2026-08-27T07:00:00+00:00",
    ),
)
def test_deadline_cannot_drift(value: str) -> None:
    with pytest.raises(ValueError):
        parse_authorized_deadline(value)


def test_snapshot_cadence_continues_from_update_240() -> None:
    assert not is_snapshot_iteration(240)
    assert not is_snapshot_iteration(299)
    assert is_snapshot_iteration(300)
    assert is_snapshot_iteration(360)
    assert not is_snapshot_iteration(361)


def test_deadline_crossing_selects_first_completed_update() -> None:
    deadline = datetime.fromisoformat(AUTHORIZED_DEADLINE_LOCAL)
    before = deadline - timedelta(milliseconds=1)
    after = deadline + timedelta(seconds=4)
    assert not crosses_deadline(before, before, deadline)
    assert crosses_deadline(before, deadline, deadline)
    assert crosses_deadline(before, after, deadline)
    assert not crosses_deadline(deadline, after, deadline)
