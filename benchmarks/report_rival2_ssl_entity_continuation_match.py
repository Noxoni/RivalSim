"""CPU-only integrity and counter reduction of the frozen +100/+200 matches."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/rival2/ssl_entity_continuation_v1"


def digest(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest().upper()


def summarize(path):
    source = json.loads(path.read_text())
    raw = source["raw"]
    sides = raw["match.rival_side"]
    goals = raw["match.goal_count"]
    checks = {
        "ten_full_regulation_matches": len(sides) == 10
        and all(t == 36000 for t in raw["match.total_ticks"]),
        "all_goals_valid": all(
            sum(raw["goal_entry_valid"][i][:n]) == n for i, n in enumerate(goals)
        ),
        "hidden_reset_per_goal": source["hidden_resets"] == goals,
        "no_goal_overflow": not any(raw["goal_overflow"]),
        "model_unchanged": source["model_unchanged"],
        "checkpoint_unchanged": source["checkpoint_unchanged"],
        "zero_optimizer_steps": source["optimizer_steps"] == 0,
    }
    assert all(checks.values()), checks

    def total(key, opponent=False):
        return sum(raw[key][i][1 - side if opponent else side] for i, side in enumerate(sides))

    def directions(key):
        return {
            name: sum(raw[key][i][side][j] for i, side in enumerate(sides))
            for j, name in enumerate(("backward", "neutral", "forward"))
        }

    same, opponent, resolved = (
        total(key) for key in ("possession_same", "possession_opponent", "possession_total")
    )
    assert same + opponent == resolved
    assert sum(directions("direction_count").values()) == total("touch_count")
    assert source["summary"]["goals_for"] == 0
    assert sum(goals) == source["summary"]["goals_against"]
    return {
        "source": path.relative_to(ROOT).as_posix(),
        "source_text_sha256": digest(path),
        "checkpoint": source["checkpoint"],
        "summary": source["summary"],
        "integrity": checks,
        "conceded_by_match": goals,
        "contacts": {
            "rival": total("touch_count"),
            "nexto": total("touch_count", opponent=True),
            "same_player_followup": same,
            "opponent_followup": opponent,
            "resolved_followups": resolved,
            "same_player_followup_fraction": same / resolved if resolved else None,
            "ball_velocity_change": directions("direction_count"),
            "ball_displacement_until_next_contact_or_goal": directions("displacement_count"),
            "conceded_with_at_most_one_contact_since_reset": total(
                "kickoff_goal_count", opponent=True
            ),
            "native_rival_demo_events": total("demo_count"),
        },
    }


def main():
    comparisons = [
        summarize(ROOT / "results/rival2/ssl_entity_joint_control_v1/full_match_entity100.json"),
        summarize(RESULTS / "full_match_000200.json"),
    ]
    protocol = json.loads((RESULTS / "full_match_000200_protocol.json").read_text())
    result = json.loads((RESULTS / "full_match_000200.json").read_text())
    assert protocol["baseline_text_sha256"] == comparisons[0]["source_text_sha256"]
    assert result["protocol_text_sha256"] == digest(RESULTS / "full_match_000200_protocol.json")
    assert result["baseline_summary"] == comparisons[0]["summary"]
    assert protocol["spec"]["candidate_accepted_updates"] == 200
    assert result["checkpoint"]["sha256"] == (
        "2A01B1280BD1385FDD56FEC0FB65C7C8432FB3B0EE2668EBD1B00288EEC99652"
    )
    report = {
        "verdict": "PASS",
        "meaning": "Match integrity only, not scoring, competitive or SSL capability",
        "optimizer_steps_in_audit": 0,
        "source": "Existing native counters only; no new simulation or event detector",
        "comparisons": comparisons,
        "semantics": {
            "followup": "Next distinct contact identity, not possession time or controllability",
            "velocity": "Canonical ball-velocity change at contact, not resulting ball velocity",
            "displacement": "Canonical net ball y until next contact or goal; +/-100 uu bins",
            "kickoff_goal": "At most one distinct contact since reset, not elapsed time",
            "demo": "Native demolition count; no inferred intent",
            "no_touch": "No no-touch resets in full matches; use matches_without_rival_touch",
        },
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (RESULTS / "full_match_000200_integrity.json").write_text(encoded, encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
