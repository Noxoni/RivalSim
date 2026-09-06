"""Preserve closed block evidence after the operator verifies its worker exited.

No environment, policy or optimizer is constructed. The existing CPU report
checks the checkpoint and existing completed evaluation; no match is rerun.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks import report_direct_skills_native_nexto_v1 as report


def validate_terminal(state, latest, rows, authority):
    limit = authority["child_updates"]
    if state["status"] != "complete_review" or state["branch_updates"] != limit:
        raise ValueError("Campaign has not completed its frozen block")
    if latest != state["latest_checkpoint"]:
        raise ValueError("Latest checkpoint and final state disagree")
    if latest["branch_updates"] != limit or latest["accepted_updates"] != authority["parent"]["accepted_updates"]+limit:
        raise ValueError("Wrong final checkpoint counters")
    if len(rows) != limit or [r["branch_updates"] for r in rows] != list(range(1,limit+1)):
        raise ValueError("Final training curve is incomplete or noncontiguous")
    if state["cumulative_optimizer_steps"] != rows[-1]["cumulative_optimizer_steps"]:
        raise ValueError("Final optimizer counter disagrees with training curve")
    if any(r["ppo"]["kl_rejections"] for r in rows):
        raise ValueError("Frozen KL-telemetry semantics were violated")


def finalize():
    out, external = report.OUT, report.RUN
    authority = report.read(out / "training_authority.json")
    state, latest = report.read(external / "campaign_state.json"), report.read(external / "latest.json")
    prefix, rows = report.training_prefix(external / "training_curve.jsonl", authority["child_updates"])
    validate_terminal(state, latest, rows, authority)
    assert not (external / "failure.json").exists(), "Preserve and investigate failure evidence instead"
    assert prefix == (external / "training_curve.jsonl").read_bytes(), "Unexpected records after frozen boundary"
    latest_path = Path(latest["path"])
    assert latest_path.resolve().parent == external.resolve()
    assert report.sha(latest_path) == latest["sha256"]
    # This verifies the already frozen local/remote source and evidence hashes,
    # but does not call run(), collect(), backward(), step() or a GPU preflight.
    from benchmarks.run_direct_skills_native_nexto_v1 import verify
    package = verify()
    report.report(authority["child_updates"])
    final_path = out / f"child_{authority['child_updates']:06d}.json"
    final = report.read(final_path)
    early_path = out / "child_000010.json"
    early = report.read(early_path)
    # Existing reducer validates each complete raw scoreboard/contact record.
    report.reduce(early_path); report.reduce(final_path)
    change_from_early = report.compare_records(early, final)
    captures = {
        "campaign_state.json": "final_campaign_state.json",
        "latest.json": "final_latest.json",
        "stdout.log": "stdout.log",
        "stderr.log": "stderr.log",
    }
    captured = {}
    for source, destination in captures.items():
        data = (external / source).read_bytes()
        report.save_once(out / destination, data)
        captured[destination] = report.sha(out / destination)
    result = dict(
        version="RIVAL2_DIRECT_SKILLS_NATIVE_NEXTO_COMPLETION_V1",
        status="complete_review_not_SSL", accepted_updates=authority["child_updates"],
        ancestry_update=state["accepted_updates"], authority_sha256=report.canonical(authority),
        frozen_source_and_evidence_hashes_verified=True,
        frozen_package=package, final_checkpoint=final["checkpoint"], rolling_checkpoint=latest,
        captures=captured, training_prefix_sha256=report.sha(out / "through_000025.jsonl"),
        complete_progress_sha256=report.sha(out / "progress_000025.json"),
        early_progress_sha256=report.sha(out / "progress_000010.json"),
        early_vs_final=change_from_early,
        final_contacts=report.reduce(final_path)["contacts"],
        optimizer_steps_in_finalization=0, no_new_evaluation=True,
        safety="Final checkpoint finite/counter checks pass; no failure artifact. Runtime nonfinite protection unchanged; KL telemetry only.",
        interpretation="Frozen development block completed. Gameplay and further learning decisions require review; not native-game or SSL proof.")
    report.save_once(out / "completion.json", (json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+"\n").encode())
    print(json.dumps(dict(status=result["status"], final_checkpoint=result["final_checkpoint"],
                          early_vs_final=change_from_early["summary"])))


if __name__ == "__main__":
    finalize()
