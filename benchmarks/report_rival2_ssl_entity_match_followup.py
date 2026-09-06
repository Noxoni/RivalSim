"""CPU-only reduction of already recorded matches; no evaluation or learning."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/rival2/ssl_entity_continuation_v1"


def digest(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest().upper()


def reduce(path):
    source = json.loads(path.read_text())
    raw, summary = source["raw"], source["summary"]
    sides, goals = raw["match.rival_side"], raw["match.goal_count"]
    scores = list(zip(raw["match.blue_score"], raw["match.orange_score"], strict=True))
    checks = {
        "ten_complete_matches": len(sides) == 10 and all(raw["match.done"]),
        "regulation_completed": all(t >= 36000 for t in raw["match.total_ticks"]),
        "all_goals_valid": all(
            sum(raw["goal_entry_valid"][i][:n]) == n for i, n in enumerate(goals)
        ),
        "hidden_reset_per_goal": source["hidden_resets"] == goals,
        "no_goal_overflow": not any(raw["goal_overflow"]),
        "model_unchanged": source["model_unchanged"],
        "checkpoint_unchanged": source["checkpoint_unchanged"],
        "zero_optimizer_steps": source["optimizer_steps"] == 0,
        "scoring_conserved": sum(goals) == summary["goals_for"] + summary["goals_against"],
        "scoreboard_matches_summary": (
            sum(scores[i][side] for i, side in enumerate(sides)) == summary["goals_for"]
            and sum(scores[i][1 - side] for i, side in enumerate(sides)) == summary["goals_against"]
        ),
        "scorers_match_scoreboard": all(
            raw["goal_scorer"][i][:n].count(team) == scores[i][team]
            for i, n in enumerate(goals)
            for team in (0, 1)
        ),
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
        total(k) for k in ("possession_same", "possession_opponent", "possession_total")
    )
    assert same + opponent == resolved
    assert sum(directions("direction_count").values()) == total("touch_count")
    return dict(
        source=path.relative_to(ROOT).as_posix(),
        source_text_sha256=digest(path),
        checkpoint=source["checkpoint"],
        summary=summary,
        integrity=checks,
        contacts=dict(
            rival=total("touch_count"),
            nexto=total("touch_count", True),
            same_player_followup=same,
            opponent_followup=opponent,
            resolved_followups=resolved,
            same_player_followup_fraction=same / resolved if resolved else None,
            ball_velocity_change=directions("direction_count"),
            ball_displacement_until_next_contact_or_goal=directions("displacement_count"),
            conceded_with_at_most_one_contact_since_reset=total("kickoff_goal_count", True),
            native_rival_demo_events=total("demo_count"),
        ),
    )


def main(target, baseline):
    output = RESULTS / f"full_match_{target:06d}_integrity.json"
    target_path = RESULTS / f"full_match_{target:06d}.json"
    source = json.loads(target_path.read_text())
    protocol = RESULTS / f"full_match_{target:06d}_protocol.json"
    comparisons = [reduce(RESULTS / f"full_match_{n:06d}.json") for n in (baseline, target)]
    assert source["protocol_text_sha256"] == digest(protocol)
    assert source["baseline_text_sha256"] == comparisons[0]["source_text_sha256"]
    assert source["baseline_summary"] == comparisons[0]["summary"]
    assert source["training_pause_checkpoint_unchanged"]
    assert source["checkpoint"]["sha256"] == json.loads(protocol.read_text())["checkpoint_sha256"]
    result = dict(
        verdict="PASS",
        meaning="Recorded match integrity only, not competitive or SSL capability",
        comparisons=comparisons,
        optimizer_steps_in_audit=0,
        training_pause_checkpoint=source["training_pause_checkpoint"],
        semantics=dict(
            followup="Identity of next distinct contact, not possession time",
            velocity="Canonical velocity change at contact, not resulting ball velocity",
            displacement="Canonical net ball y until next contact/goal; +/-100 uu bins",
            kickoff_goal="At most one distinct contact since reset, not elapsed time",
            no_touch="No no-touch resets; report matches without Rival contact",
            demo="Native demolition count, not intent",
        ),
    )
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if output.exists():
        assert output.read_text() == text, "Existing report differs"
    else:
        output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--baseline", type=int, required=True)
    args = parser.parse_args()
    main(args.target, args.baseline)
