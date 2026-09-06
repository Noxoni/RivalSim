import copy
import json
from pathlib import Path

import pytest

from benchmarks import report_rival2_ssl_entity_match_followup as report


@pytest.fixture
def recorded():
    path = (
        Path(__file__).resolve().parents[1]
        / "results/rival2/ssl_entity_continuation_v1/full_match_000200.json"
    )
    return json.loads(path.read_text())


def test_existing_counters_and_positive_score_supported(recorded, tmp_path, monkeypatch):
    monkeypatch.setattr(report, "ROOT", tmp_path)
    path = tmp_path / "match.json"
    path.write_text(json.dumps(recorded))
    result = report.reduce(path)
    assert result["contacts"]["rival"] == 238
    assert result["contacts"]["same_player_followup"] == 22
    assert result["summary"]["goals_against"] == 243
    # The reducer conserves total goals without hard-coding zero learner goals.
    changed = copy.deepcopy(recorded)
    changed["summary"].update(goals_for=1, goals_against=242)
    changed["raw"]["match.blue_score"][0] += 1
    changed["raw"]["match.orange_score"][0] -= 1
    changed["raw"]["goal_scorer"][0][0] = 0
    path.write_text(json.dumps(changed))
    assert report.reduce(path)["summary"]["goals_for"] == 1


@pytest.mark.parametrize(
    "fault", ["reset", "goal", "overflow", "learning", "unfinished", "scoreboard"]
)
def test_integrity_failure_is_rejected(recorded, tmp_path, monkeypatch, fault):
    monkeypatch.setattr(report, "ROOT", tmp_path)
    if fault == "reset":
        recorded["hidden_resets"][0] += 1
    elif fault == "goal":
        recorded["summary"]["goals_against"] += 1
    elif fault == "overflow":
        recorded["raw"]["goal_overflow"][0] = 1
    elif fault == "learning":
        recorded["optimizer_steps"] = 1
    elif fault == "scoreboard":
        recorded["raw"]["match.blue_score"][0] += 1
    else:
        recorded["raw"]["match.done"][0] = 0
    path = tmp_path / "match.json"
    path.write_text(json.dumps(recorded))
    with pytest.raises(AssertionError):
        report.reduce(path)
