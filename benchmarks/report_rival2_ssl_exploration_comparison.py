"""CPU-only evidence consolidation; never selects from incomplete pilot results."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.run_rival2_ssl_exploration_comparison import (
    ARMS,
    CHECKPOINTS,
    PARENT,
    PARENT_SHA,
    RESULTS,
    authority,
    sha,
    utc,
    write_json,
)


def main():
    assert sha(PARENT) == PARENT_SHA
    arms = {}
    for arm in ARMS:
        evaluations = {}
        for offset in (0, 10, 20, 30):
            path = RESULTS / arm / f"evaluation_{offset:03d}.json"
            if not path.exists():
                continue
            report = json.loads(path.read_text())
            if offset:
                snapshot = CHECKPOINTS / arm / f"plus_{offset:03d}.pt"
                assert snapshot.exists()
                assert sha(snapshot) == report["checkpoint"]["sha256"], (
                    "evaluation/checkpoint mismatch"
                )
            evaluations[str(offset)] = report
        curve_path = RESULTS / arm / "training_curve.jsonl"
        curve = []
        if curve_path.exists():
            for line in curve_path.read_text().splitlines():
                try:
                    curve.append(json.loads(line))
                except json.JSONDecodeError:
                    break
        arms[arm] = dict(
            evaluations=evaluations,
            accepted_additional_updates=max(
                [row["additional_updates"] for row in curve], default=0
            ),
            curve=curve,
        )
    complete = all("30" in row["evaluations"] for row in arms.values())
    verdict = "IN_PROGRESS"
    checks = {}
    if complete:
        c = arms["control"]["evaluations"]
        h = arms["half_sigma"]["evaluations"]
        checks = {
            "acquisition_better_at_20": h["20"]["acquisition_selfplay"]["focal_touch_fraction"]
            >= c["20"]["acquisition_selfplay"]["focal_touch_fraction"] + 4 / 64,
            "acquisition_better_at_30": h["30"]["acquisition_selfplay"]["focal_touch_fraction"]
            >= c["30"]["acquisition_selfplay"]["focal_touch_fraction"] + 4 / 64,
            "acquisition_improves_parent": h["30"]["acquisition_selfplay"]["focal_touch_fraction"]
            >= h["0"]["acquisition_selfplay"]["focal_touch_fraction"] + 4 / 64,
            "finishing_touch_nonregression": h["30"]["finishing_selfplay"]["focal_touch_fraction"]
            >= c["30"]["finishing_selfplay"]["focal_touch_fraction"] - 4 / 64,
            "finishing_goal_nonregression": h["30"]["finishing_selfplay"]["goals_for"]
            >= c["30"]["finishing_selfplay"]["goals_for"] - 3,
        }
        verdict = (
            "PROMISING_HALF_SIGMA_PILOT_REQUIRES_INDEPENDENT_CONFIRMATION"
            if all(checks.values())
            else "INCONCLUSIVE_OR_NEGATIVE_HALF_SIGMA_PILOT"
        )
    report = dict(
        utc=utc(),
        verdict=verdict,
        complete=complete,
        checks=checks,
        ssl_capability_demonstrated=False,
        authority=authority(),
        arms=arms,
    )
    write_json(RESULTS / "comparison_report.json", report)
    print(
        json.dumps(
            dict(
                verdict=verdict,
                checks=checks,
                arms={
                    arm: {
                        k: dict(
                            acquisition=r["acquisition_selfplay"]["focal_touch_fraction"],
                            finishing_goals=r["finishing_selfplay"]["goals_for"],
                            nexto_goals_for=r["standard_kickoff_nexto"]["goals_for"],
                            nexto_goals_against=r["standard_kickoff_nexto"]["goals_against"],
                        )
                        for k, r in info["evaluations"].items()
                    }
                    for arm, info in arms.items()
                },
            )
        )
    )


if __name__ == "__main__":
    main()
