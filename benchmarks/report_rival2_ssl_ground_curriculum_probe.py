"""CPU-only evidence summary for the frozen reset-only probe."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.run_rival2_fresh_ground_30hz_v1 import sha, tensor_hash, utc, write_json
from benchmarks.run_rival2_ssl_exploration_comparison import PARENT, PARENT_SHA

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
    if 30 in probe:
        parent = torch.load(PARENT, map_location="cpu", weights_only=False)
        path = CHECKPOINTS / "plus_030.pt"
        final = torch.load(path, map_location="cpu", weights_only=False)
        curve = [json.loads(s) for s in (RESULTS / "training_curve.jsonl").read_text().splitlines()]
        parent_steps = [float(s["step"]) for s in parent["optimizer"]["state"].values()]
        final_steps = [float(s["step"]) for s in final["optimizer"]["state"].values()]
        checks = dict(
            parent_unchanged=sha(PARENT) == PARENT_SHA,
            complete_30_rows=len(curve) == 30,
            final_offset=final["additional_updates"] == 30,
            all_model_tensors_finite=all(
                bool(torch.isfinite(v).all()) for v in final["model"].values()
            ),
            all_adam_tensors_finite=all(
                bool(torch.isfinite(v).all())
                for s in final["optimizer"]["state"].values()
                for v in s.values()
                if torch.is_tensor(v)
            ),
            optimizer_steps_complete=len(parent_steps) == len(final_steps) == 24
            and all(b - a == 30 * 182 for a, b in zip(parent_steps, final_steps, strict=True)),
            checkpoint_matches_evaluation=sha(path) == probe[30]["checkpoint"]["sha256"],
        )
        write_json(
            RESULTS / "final_integrity.json",
            dict(
                utc=utc(),
                checks=checks,
                verdict="PASS" if all(checks.values()) else "FAIL",
                checkpoint=str(path.relative_to(ROOT)),
                sha256=sha(path),
                model_tensor_sha256=tensor_hash(final["model"]),
                training_curve_sha256=sha(RESULTS / "training_curve.jsonl"),
                capability_verdict=verdict,
                promoted=False,
            ),
        )
        assert all(checks.values()), checks
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
