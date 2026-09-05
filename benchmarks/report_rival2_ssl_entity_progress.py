"""CPU-only comparison of completed entity-pilot evaluations, never a selector."""

# ruff: noqa: E402
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.run_rival2_fresh_ground_30hz_v1 import sha, utc, write_json
from benchmarks.run_rival2_ssl_entity_joint_control import BOUNDARIES, CHECKPOINTS, RESULTS, verify

CASES = ("acquisition_selfplay", "finishing_selfplay", "standard_kickoff_nexto")


def summarize_case(current, initial, parent):
    for other in (initial, parent):
        if (
            current["scenario_sha256"] != other["scenario_sha256"]
            or current["worlds"] != other["worlds"]
        ):
            raise ValueError("comparison requires identical scenario corpus and count")
    n = current["worlds"]

    def touched(report):
        return round(report["focal_touch_fraction"] * n)

    return dict(
        worlds=n,
        touched=touched(current),
        initial_touched=touched(initial),
        parent_touched=touched(parent),
        touch_count_change_vs_initial=touched(current) - touched(initial),
        touch_count_change_vs_parent=touched(current) - touched(parent),
        goals_for=current["goals_for"],
        initial_goals_for=initial["goals_for"],
        parent_goals_for=parent["goals_for"],
        goals_against=current["goals_against"],
        no_touch_truncations=current["no_touch_truncations"],
        initial_no_touch=initial["no_touch_truncations"],
        parent_no_touch=parent["no_touch_truncations"],
        touches_per_minute=current["touches_per_minute"],
        median_first_touch_seconds_if_touched=current["median_first_touch_seconds_if_touched"],
        full_match_winrate_available=False,
    )


def main():
    verify(published=True)
    initial = json.loads((RESULTS / "evaluation_000.json").read_text())
    parent = json.loads(
        (
            ROOT / "results/rival2/ssl_development_exploration_v1/control/evaluation_000.json"
        ).read_text()
    )
    evaluations = []
    files = {}
    for path in sorted(RESULTS.glob("evaluation_*.json")):
        report = json.loads(path.read_text())
        offset = report["accepted_updates"]
        assert offset in BOUNDARIES
        checkpoint = CHECKPOINTS / (
            "launch_zero_before_recovery.pt" if offset == 0 else f"plus_{offset:03d}.pt"
        )
        assert sha(checkpoint) == report["checkpoint"]["sha256"], checkpoint
        cases = {case: summarize_case(report[case], initial[case], parent[case]) for case in CASES}
        evaluations.append(
            dict(accepted_updates=offset, checkpoint=str(checkpoint.relative_to(ROOT)), cases=cases)
        )
        files[str(path.relative_to(ROOT))] = sha(path)
    result = dict(
        utc=utc(),
        status="INITIAL_BUDGET_COMPLETE_REQUIRES_REVIEW"
        if evaluations[-1]["accepted_updates"] == 100
        else "PILOT_IN_PROGRESS",
        latest_completed_evaluation=evaluations[-1]["accepted_updates"],
        evaluations=evaluations,
        evidence_hashes=files,
        ssl_capability_proven=False,
        interpretation="All cases are original fixed development scenarios, not complete "
        "match win rates or a held-out acceptance test. Candidate head initialization differs "
        "from the hybrid parent, so report both baselines. No-touch truncations can overlap "
        "earlier touches. Compare coverage together with conditional touch speed and scoring; "
        "do not select a lucky intermediate boundary.",
    )
    write_json(RESULTS / "progress_report.json", result)
    print(json.dumps(dict(status=result["status"], latest=evaluations[-1]), indent=2))


if __name__ == "__main__":
    main()
