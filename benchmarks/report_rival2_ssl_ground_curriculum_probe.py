"""CPU-only evidence summary for the frozen reset-only probe."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.run_rival2_fresh_ground_30hz_v1 import sha, utc, write_json

RESULTS = ROOT / "results/rival2/ssl_ground_curriculum_probe_v1"
CONTROL = ROOT / "results/rival2/ssl_development_exploration_v1/control"
CHECKPOINTS = ROOT / "checkpoints/rival2/ssl_ground_curriculum_probe_v1"


def decision(probe, control):
    if not all(i in probe and i in control for i in (0, 20, 30)):
        return "INCOMPLETE", {}

    def acquisition(report):
        return report["acquisition_selfplay"]["focal_touch_fraction"]

    checks = {
        "acquisition_better_at_20": acquisition(probe[20]) - acquisition(control[20]) >= 4 / 64,
        "acquisition_better_at_30": acquisition(probe[30]) - acquisition(control[30]) >= 4 / 64,
        "acquisition_improves_parent": acquisition(probe[30]) - acquisition(probe[0]) >= 4 / 64,
        "finishing_goal_nonregression": probe[30]["finishing_selfplay"]["goals_for"]
        >= control[30]["finishing_selfplay"]["goals_for"] - 3,
    }
    return (
        "PROMISING_PILOT_REQUIRES_TRANSFER_CONFIRMATION"
        if all(checks.values())
        else "INCONCLUSIVE_OR_NEGATIVE_CURRICULUM_PILOT"
    ), checks


def main():
    probe, control = {}, {}
    rows = []
    for path in sorted(RESULTS.glob("evaluation_*.json")):
        report = json.loads(path.read_text())
        offset = report["additional_updates"]
        if offset:
            checkpoint = CHECKPOINTS / f"plus_{offset:03d}.pt"
            assert sha(checkpoint) == report["checkpoint"]["sha256"]
        reference = json.loads((CONTROL / f"evaluation_{offset:03d}.json").read_text())
        probe[offset], control[offset] = report, reference
        for case in ("acquisition_selfplay", "finishing_selfplay", "standard_kickoff_nexto"):
            current, prior = report[case], reference[case]
            assert current["scenario_sha256"] == prior["scenario_sha256"]
            if offset == 0:
                assert current == prior, "parent baseline failed exact deterministic parity"
            rows.append(
                dict(
                    offset=offset,
                    case=case,
                    touched=current["focal_touch_fraction"] * current["worlds"],
                    control_touched=prior["focal_touch_fraction"] * prior["worlds"],
                    goals_for=current["goals_for"],
                    goals_against=current["goals_against"],
                    control_goals_for=prior["goals_for"],
                    control_goals_against=prior["goals_against"],
                    no_touch=current["no_touch_truncations"],
                    control_no_touch=prior["no_touch_truncations"],
                )
            )
    verdict, checks = decision(probe, control)
    report = dict(
        utc=utc(),
        verdict=verdict,
        checks=checks,
        rows=rows,
        ssl_capability_demonstrated=False,
        authority_sha256=sha(RESULTS / "authority.json"),
        interpretation="Original fixed development scenarios; no match win-rate or SSL claim. "
        "Complete the frozen experiment; do not select a lucky intermediate boundary.",
    )
    write_json(RESULTS / "comparison_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
