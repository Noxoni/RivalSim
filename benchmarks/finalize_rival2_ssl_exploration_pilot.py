"""Consolidate completed, immutable pilot evidence; never train or select a model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.run_rival2_ssl_exploration_comparison import (
    ARMS,
    CHECKPOINTS,
    PARENT,
    PARENT_SHA,
    RESULTS,
    sha,
    utc,
    write_json,
)


def main():
    assert sha(PARENT) == PARENT_SHA
    pilot = json.loads((RESULTS / "comparison_report.json").read_text())
    independent = {
        arm: json.loads((RESULTS / "independent" / f"{arm}.json").read_text())
        for arm in ("parent", *ARMS)
    }
    rows = {}
    for arm, report in independent.items():
        path = PARENT if arm == "parent" else CHECKPOINTS / arm / "plus_030.pt"
        assert sha(path) == report["checkpoint"]["sha256"]
        rows[arm] = {}
        for case, result in report["cases"].items():
            parent = independent["parent"]["cases"][case]
            assert result["scenario_sha256"] == parent["scenario_sha256"]
            assert len(result["per_case"]) == len(parent["per_case"])
            gained = lost = 0
            for a, b in zip(result["per_case"], parent["per_case"], strict=True):
                assert (a["case"], a["side"]) == (b["case"], b["side"])
                gained += a["first_touch_seconds"] is not None and b["first_touch_seconds"] is None
                lost += a["first_touch_seconds"] is None and b["first_touch_seconds"] is not None
            rows[arm][case] = dict(
                worlds=result["worlds"],
                touched=result["focal_cases_with_touch"],
                gained_touch_cases_vs_parent=gained,
                lost_touch_cases_vs_parent=lost,
                goals_for=result["totals"]["goals_for"],
                goals_against=result["totals"]["goals_against"],
                no_touch_truncations=result["totals"]["no_touch"],
                speed=result["mean_speed"],
                median_first_touch_seconds_if_touched=result[
                    "median_first_touch_seconds_if_touched"
                ],
            )
    reference = json.loads((RESULTS / "diagnostics" / "ground_steering_reference.json").read_text())
    gradients = json.loads((RESULTS / "diagnostics" / "entropy_gradient_audit.json").read_text())
    report = dict(
        utc=utc(),
        pilot_verdict=pilot["verdict"],
        independent=rows,
        scripted_reference=reference,
        exploration_gradient_audit=gradients,
        capability_verdict="SSL gameplay NOT demonstrated; no pilot checkpoint promoted",
        interpretation=(
            "Thirty updates per arm test one local analog-noise change, not all exploration "
            "designs or the attainable skill ceiling. Matched scenario outcomes are not full "
            "match win rates. Conditional touch time excludes failures; read touch coverage "
            "alongside it. Scripted reference is not learned Rival or training data."
        ),
        evidence_hashes={
            str(p.relative_to(ROOT)).replace("\\", "/"): sha(p)
            for p in [
                PARENT,
                *(CHECKPOINTS / arm / "plus_030.pt" for arm in ARMS),
                *(RESULTS / "independent" / f"{arm}.json" for arm in independent),
                RESULTS / "diagnostics" / "ground_steering_reference.json",
                RESULTS / "diagnostics" / "entropy_gradient_audit.json",
            ]
        },
    )
    write_json(RESULTS / "final_pilot_report.json", report)
    print(json.dumps(dict(pilot_verdict=pilot["verdict"], independent=rows), indent=2))


if __name__ == "__main__":
    main()
