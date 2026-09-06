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


def load_pair(offset, baseline_offset):
    assert 0 <= baseline_offset <= offset
    paths = [RESULTS / f"evaluation_{i:06d}.json" for i in (baseline_offset, offset)]
    docs = [json.loads(p.read_text()) for p in paths]
    assert [d["accepted_updates"] for d in docs] == [baseline_offset, offset]
    return paths, docs


def output_path(prefix, offset, baseline_offset):
    suffix = f"{offset:06d}" if baseline_offset == 0 else f"{baseline_offset:06d}_to_{offset:06d}"
    return RESULTS / f"{prefix}_{suffix}.json"


def main(offset, baseline_offset=0):
    paths, docs = load_pair(offset, baseline_offset)
    report = dict(
        accepted_updates=offset,
        baseline_updates=baseline_offset,
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
    path = output_path("comparison", offset, baseline_offset)
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


def group_start_cases(item, kickoff, layouts, sides):
    """Partition existing results by initial state, never by successful outcome."""
    verify_cases(item)
    assert len(kickoff) == len(layouts) == len(sides) == item["worlds"]
    groups = {}
    for i, (is_kickoff, layout, side) in enumerate(zip(kickoff, layouts, sides, strict=True)):
        assert is_kickoff in (0, 1) and side in (0, 1)
        assert layout in range(5) if is_kickoff else layout == -1
        label = f"kickoff_layout_{layout}_side_{side}" if is_kickoff else "ongoing_ground"
        groups.setdefault(label, []).append(i)
    raw = item["raw"]
    return {
        name: dict(
            cases=len(indices),
            case_indices=indices,
            goals_for=int(sum(raw["goals"][i] for i in indices)),
            goals_against=int(sum(raw["concedes"][i] for i in indices)),
            touches=int(sum(raw["touches"][i] for i in indices)),
            touched_cases=sum(raw["touches"][i] > 0 for i in indices),
            timeouts=sum(raw["endings"][i] == 2 for i in indices),
        )
        for name, indices in sorted(groups.items())
    }


def start_groups(offset, baseline_offset=0):
    # The imported generator constructs NumPy initial states only. No simulator,
    # policy, optimizer or CUDA rollout is constructed or evaluated here.
    from rivalsim.direct_skills_v1 import NAMES, SEED, scenarios
    from rivalsim.fresh_ground_30hz import scenario_hash

    paths, docs = load_pair(offset, baseline_offset)
    compare_documents(*docs)
    families = {}
    for family in (0, 4):
        name = NAMES[family]
        bank = scenarios(64, SEED + 1000 + family, family_only=family)
        digest = scenario_hash(bank)
        assert all(d["skills"][name]["scenario_sha256"] == digest for d in docs)
        metadata = [
            a.tolist() for a in (bank.kickoff_indicator, bank.kickoff_layout, bank.focal_side)
        ]
        families[name] = dict(
            scenario_sha256=digest,
            baseline=group_start_cases(docs[0]["skills"][name], *metadata),
            candidate=group_start_cases(docs[1]["skills"][name], *metadata),
        )
    result = dict(
        accepted_updates=offset,
        baseline_updates=baseline_offset,
        authority_sha256=docs[0]["authority_sha256"],
        checkpoints=[d["checkpoint"] for d in docs],
        sources={
            p.name: hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest().upper()
            for p in paths
        },
        families=families,
        interpretation=(
            "Recorded cases grouped by hash-verified initial-state metadata. Repeated "
            "standard kickoff layouts are not independent scenario generalization. "
            "A gain on kickoff starts must not be called a gain on ongoing-ground starts."
        ),
        policy_evaluations_run=0,
        optimizer_steps=0,
    )
    output = output_path("start_groups", offset, baseline_offset)
    if output.exists():
        assert json.loads(output.read_text()) == result
    else:
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", required=True, type=int)
    parser.add_argument(
        "--baseline", default=0, type=int, help="Completed comparison offset (default 0)"
    )
    parser.add_argument("--start-groups", action="store_true")
    args = parser.parse_args()
    if args.start_groups:
        start_groups(args.update, args.baseline)
    else:
        main(args.update, args.baseline)
