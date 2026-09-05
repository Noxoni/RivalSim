"""CPU-only reduction of existing full-match counters; no new event detectors."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/rival2/ssl_entity_joint_control_v1"


def digest(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest().upper()


def summarize(path):
    source = json.loads(path.read_text())
    raw = source["raw"]
    sides = raw["match.rival_side"]
    assert source["model_unchanged"] and source["checkpoint_unchanged"]
    assert source["optimizer_steps"] == 0
    assert all(x == 36000 for x in raw["match.total_ticks"])

    def scalar(key, opponent=False):
        return sum(raw[key][i][1 - side if opponent else side] for i, side in enumerate(sides))

    def directions(key):
        return {
            name: sum(raw[key][i][side][j] for i, side in enumerate(sides))
            for j, name in enumerate(("backward", "neutral", "forward"))
        }

    contacts = scalar("touch_count")
    resolved = scalar("possession_total")
    same, opponent = scalar("possession_same"), scalar("possession_opponent")
    assert same + opponent == resolved
    impulse = directions("direction_count")
    displacement = directions("displacement_count")
    assert sum(impulse.values()) == contacts
    conceded = source["summary"]["goals_against"]
    one_contact_goals = scalar("kickoff_goal_count", opponent=True)
    goal_flags = sum(
        sum(raw["goal_kickoff"][i][:count]) for i, count in enumerate(raw["match.goal_count"])
    )
    assert source["summary"]["goals_for"] == 0
    assert one_contact_goals == goal_flags
    return dict(
        source=path.relative_to(ROOT).as_posix(),
        source_text_sha256=digest(path),
        checkpoint=source["checkpoint"],
        protocol_sha256=source["protocol_sha256"],
        contacts=contacts,
        contact_ball_velocity_change=impulse,
        ball_net_displacement_after_contact=displacement,
        followup_same_player=same,
        followup_opponent=opponent,
        followup_resolved_total=resolved,
        followup_same_fraction=same / resolved,
        goals_conceded=conceded,
        goals_conceded_with_at_most_one_contact_since_reset=one_contact_goals,
        fraction_conceded_with_at_most_one_contact=one_contact_goals / conceded,
        native_demo_events=scalar("demo_count"),
        wall_surface_intervals=scalar("wall_continuation_count"),
        backboard_surface_intervals=scalar("backboard_continuation_count"),
    )


def main():
    policies = [
        summarize(RESULTS / f"full_match_{name}.json") for name in ("parent597", "entity100")
    ]
    assert policies[0]["protocol_sha256"] == policies[1]["protocol_sha256"]
    report = dict(
        version="ENTITY100_POSTHOC_CONTACT_BREAKDOWN_V1",
        computation="Existing committed counters only; CPU; zero new simulation or training",
        counter_source="rivalsim/full_match.py",
        counter_source_text_sha256=digest(ROOT / "rivalsim/full_match.py"),
        policies=policies,
        semantics={
            "velocity": "Canonical signed ball-velocity CHANGE at contact, +/-100 uu/s; "
            "forward does not mean the ball has positive resulting velocity or will score.",
            "displacement": "Signed net ball y displacement before next contact or goal, "
            "+/-100 uu. Not path length, velocity or goal probability.",
            "followup": "Identity of next distinct contact after Rival contact, excluding "
            "goal/end censoring. Not measured controllability or possession duration.",
            "kickoff_goal": "At most one registered distinct contact since kickoff reset, "
            "not every goal soon after kickoff and not an elapsed-time threshold.",
            "wall": "Ball contacted surface after this player's contact before interval end; "
            "does not establish a car wall mechanic or aerial.",
            "demo": "Native demolition events; no evidence of tactical intent.",
        },
        use="Posthoc diagnosis only; no checkpoint selection, reward change or new detector",
    )
    output = RESULTS / "full_match_contact_breakdown.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
