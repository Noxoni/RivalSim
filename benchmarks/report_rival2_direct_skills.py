"""CPU-only comparison of completed fixed-case evaluations; never runs a policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/rival2/direct_skills_v1"
FAMILIES = {"natural", "challenge", "finishing", "defense", "kickoff"}
COUNTS = ("goals_for", "goals_against", "touched_worlds", "touches", "truncation_endings")


def verify_cases(item):
    n, raw = item["worlds"], item["raw"]
    assert n == 64 and item["unfinished"] == 0
    assert all(len(raw[k]) == n for k in ("touches", "goals", "concedes", "endings", "seconds"))
    assert all(math.isfinite(v) and v >= 0 for values in raw.values() for v in values)
    assert all(v in (0, 1) for k in ("goals", "concedes") for v in raw[k])
    assert all(e in (1, 2) for e in raw["endings"])
    assert all(
        g + c == (e == 1)
        for g, c, e in zip(raw["goals"], raw["concedes"], raw["endings"], strict=True)
    )
    assert sum(raw["goals"]) == item["goals_for"]
    assert sum(raw["concedes"]) == item["goals_against"]
    assert sum(raw["touches"]) == item["touches"]
    assert sum(t > 0 for t in raw["touches"]) == item["touched_worlds"]
    assert raw["endings"].count(2) == item["truncation_endings"]
    assert raw["endings"].count(1) == item["goal_endings"]
    assert all(s > 0 for s in raw["seconds"])
    assert math.isclose(
        item["touches_per_minute"],
        sum(raw["touches"]) / sum(raw["seconds"]) * 60,
        rel_tol=2e-6,
        abs_tol=1e-6,
    )
    assert all(0 <= v <= n and int(v) == v for v in item["events"].values())


def compare_documents(baseline, candidate):
    assert baseline["authority_sha256"] == candidate["authority_sha256"]
    assert baseline["optimizer_steps"] == candidate["optimizer_steps"] == 0
    assert baseline["accepted_updates"] <= candidate["accepted_updates"]
    assert set(baseline["skills"]) == set(candidate["skills"]) == FAMILIES
    report = {}
    for family in sorted(FAMILIES):
        before, after = baseline["skills"][family], candidate["skills"][family]
        verify_cases(before)
        verify_cases(after)
        assert before["scenario_sha256"] == after["scenario_sha256"]
        assert before["event_semantics"] == after["event_semantics"]
        assert before["events"].keys() == after["events"].keys()
        scalar = {
            key: dict(before=before[key], after=after[key], change=after[key] - before[key])
            for key in COUNTS
        }
        scalar["task_proxy_events"] = {
            key: dict(before=value, after=after["events"][key], change=after["events"][key] - value)
            for key, value in before["events"].items()
        }
        outcomes = [
            [g - c for g, c in zip(x["raw"]["goals"], x["raw"]["concedes"], strict=True)]
            for x in (before, after)
        ]
        paired = [a - b for b, a in zip(*outcomes, strict=True)]
        scalar["paired_goal_outcome"] = dict(
            improved=sum(x > 0 for x in paired),
            worsened=sum(x < 0 for x in paired),
            unchanged=paired.count(0),
        )
        scalar["conditional_first_touch_seconds"] = {
            "before": before["conditional_first_touch_seconds"],
            "after": after["conditional_first_touch_seconds"],
            "warning": "Touched subsets can differ; this is not paired acquisition latency",
        }
        report[family] = scalar
    return report


def main(offset):
    paths = [RESULTS / f"evaluation_{i:06d}.json" for i in (0, offset)]
    docs = [json.loads(p.read_text()) for p in paths]
    assert docs[1]["accepted_updates"] == offset
    report = dict(
        accepted_updates=offset,
        baseline_updates=0,
        sources={
            p.name: hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest().upper()
            for p in paths
        },
        checkpoints=[d["checkpoint"] for d in docs],
        families=compare_documents(*docs),
        interpretation=(
            "Fixed initial episodes, not full-match wins. Task events are declared proxies; "
            "not proof of intended fakes, lasting possession or a saved match. "
            "No automatic acceptance threshold or reward tuning."
        ),
        optimizer_steps=0,
        policy_evaluations_run=0,
    )
    path = RESULTS / f"comparison_{offset:06d}.json"
    if path.exists():
        assert json.loads(path.read_text()) == report
    else:
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                k: {m: v for m, v in values.items() if m in COUNTS or m == "paired_goal_outcome"}
                for k, values in report["families"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", required=True, type=int)
    main(parser.parse_args().update)
