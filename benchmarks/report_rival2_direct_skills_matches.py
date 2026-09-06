"""Compare completed same-method Direct Skills matches; CPU only, no learning.

Old and corrected reset results must not be combined into a training trend.
This is a reporting guard, never a PPO acceptance/rejection mechanism.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.report_rival2_ssl_entity_match_followup import reduce  # noqa: E402

METHOD = "RIVAL2_STANDARD_KICKOFF_CONTACT_CACHE_RESET_V2"


def validate_comparability(baseline, target):
    for field in ("match_reset_version", "runtime_package_sha256", "authority_sha256"):
        if not baseline.get(field) or baseline[field] != target.get(field):
            raise ValueError(f"Not a same-method learning comparison: {field}")
    if baseline["match_reset_version"] != METHOD:
        raise ValueError("Use the corrected-reset baseline, not untagged historical matches")
    if target["accepted_updates"] <= baseline["accepted_updates"]:
        raise ValueError("Target must follow baseline; no backward/identical comparison")
    for field in ("match.rival_side", "match.starting_layout"):
        if baseline["raw"][field] != target["raw"][field]:
            raise ValueError(f"Match cases differ: {field}")


def checkpoint_integrity(source):
    identity = source["checkpoint"]
    path = Path(identity["path"])
    if not path.is_absolute():
        path = ROOT / path
    actual = hashlib.sha256(path.read_bytes()).hexdigest().upper()
    if actual != identity["sha256"]:
        raise ValueError(f"Checkpoint hash mismatch: {path}")
    if identity["accepted_updates"] != source["accepted_updates"]:
        raise ValueError("Checkpoint/evaluation offset mismatch")


def match_rows(source):
    raw = source["raw"]
    result = []
    for i, side in enumerate(raw["match.rival_side"]):
        scores = [raw["match.blue_score"][i], raw["match.orange_score"][i]]
        goals, concedes = scores[side], scores[1 - side]
        scorers = raw["goal_scorer"][i][: raw["match.goal_count"][i]]
        opening_for = int(bool(scorers) and scorers[0] == side)
        opening_against = int(bool(scorers) and scorers[0] != side)
        result.append(dict(
            match=i, side=side, layout=raw["match.starting_layout"][i],
            goals=goals, concedes=concedes, goal_difference=goals - concedes,
            contacts=raw["touch_count"][i][side],
            opening_goal_for=opening_for,
            second_goal_for=int(len(scorers) > 1 and scorers[1] == side),
            later_goals=goals - opening_for, later_concedes=concedes - opening_against,
        ))
    return result


def compare(baseline_path, target_path):
    paths = [baseline_path.resolve(), target_path.resolve()]
    sources = [json.loads(path.read_text()) for path in paths]
    validate_comparability(*sources)
    for source in sources:
        checkpoint_integrity(source)
    reductions = [reduce(path) for path in paths]
    rows = [match_rows(source) for source in sources]
    differences = [dict(
        match=a["match"], side=a["side"], layout=a["layout"], baseline=a, target=b,
        delta={k: b[k] - a[k] for k in ("goals", "concedes", "goal_difference", "contacts",
                                        "opening_goal_for", "second_goal_for", "later_goals", "later_concedes")},
    ) for a, b in zip(*rows, strict=True)]
    outcomes = [row["delta"]["goal_difference"] for row in differences]
    total = [dict(
        summary=reduction["summary"], contacts=reduction["contacts"],
        opening_goals=sum(r["opening_goal_for"] for r in group),
        second_goals=sum(r["second_goal_for"] for r in group),
        later_goals=sum(r["later_goals"] for r in group),
        later_concedes=sum(r["later_concedes"] for r in group),
    ) for reduction, group in zip(reductions, rows, strict=True)]
    return dict(
        schema="RIVAL2_DIRECT_SKILLS_SAME_METHOD_MATCH_COMPARISON_V1",
        match_reset_version=METHOD, runtime_package_sha256=sources[0]["runtime_package_sha256"],
        baseline_update=sources[0]["accepted_updates"], target_update=sources[1]["accepted_updates"],
        source_identities=[{k: r[k] for k in ("source", "source_text_sha256", "checkpoint")} for r in reductions],
        source_integrity=[r["integrity"] for r in reductions],
        checkpoint_hashes_verified=True, same_method_verified=True,
        baseline=total[0], target=total[1], per_match=differences,
        goal_difference_cases=dict(improved=sum(x > 0 for x in outcomes),
                                   unchanged=sum(x == 0 for x in outcomes), worsened=sum(x < 0 for x in outcomes)),
        optimizer_steps=0, automatic_training_change=False,
        meaning="Matched fixed development cases, not ranked skill or SSL proof. Same-player follow-ups "
        "are not possession duration. Later-goal counts have unequal segment exposure. No-touch resets "
        "are disabled by protocol. Only comparisons with the same tagged reset/runtime method are allowed.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.baseline, args.target)
    encoded = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.output.exists():
        assert json.loads(args.output.read_text()) == result, "Never replace different evidence"
    else:
        args.output.write_text(encoded)
    print(json.dumps({k: result[k] for k in ("baseline_update", "target_update", "goal_difference_cases", "same_method_verified")}))
