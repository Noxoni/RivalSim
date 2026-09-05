"""Prospectively bind measured execution-only changes at accepted update50."""

import argparse
import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402

from benchmarks import run_rival2_ssl_foundation_v5_long_trace_v1 as c  # noqa: E402
from benchmarks.validate_rival2_execution_optimization import (  # noqa: E402
    CANDIDATE_PATHS,
    OLD_AUTHORITY,
)


def rebind(payload, authority, launch):
    result = copy.deepcopy(payload)
    result["source"].update(authority_sha256=authority, schedule_authority_sha256=launch)
    result["phase_transition"].update(authority_sha256=authority, schedule_authority_sha256=launch)
    result["phase_transition"]["credit_assignment_amendment"]["authority_sha256"] = authority
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", required=True)
    parser.add_argument("--implementation-commit", required=True)
    args = parser.parse_args()
    c.configure_engine()
    assert (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        == args.implementation_commit
    )
    assert c.engine.sha256_file(c.AUTHORITY) == OLD_AUTHORITY
    before = torch.load(args.resume, map_location="cpu", weights_only=False)
    c.validate_resume_payload(before)
    assert before["accepted_updates_total"] == before["policy_version"] == 50
    summary = json.loads((c.RESULTS / "training_summary.json").read_text())
    assert summary["accepted_updates"] == 50 and summary["hard_failure"] is None
    assert summary["last_evaluation"]["accepted_updates"] == 50
    stop = json.loads((c.RUN_DIR / "STOP_REQUESTED").read_text())
    assert stop["owner"] == "long-trace-execution-optimization-u50-20260904"
    evidence = c.RESULTS / "execution_optimization/exact_scale.json"
    report = json.loads(evidence.read_text())
    assert report["verdict"] == "PASS" and all(report["checks"].values())
    assert report["no_optimizer_step"] and report["worlds"] == 32768 and report["horizon"] == 360
    assert report["source_sha256"] == c.engine.sha256_file(Path(args.resume))
    assert report["implementation_sha256"] == {
        p: c.engine.sha256_file(ROOT / p) for p in sorted(CANDIDATE_PATHS)
    }
    old = json.loads(c.AUTHORITY.read_text())
    new = c.authority_payload(args.implementation_commit, c.engine.utc_now())
    for key in (
        "ppo",
        "reward",
        "source",
        "reset_curriculum",
        "opponents",
        "exploration",
        "campaign",
        "credit_assignment_amendment",
    ):
        assert old[key] == new[key], key
    archive = c.RESULTS / "execution_optimization/original"
    archive.mkdir(parents=True, exist_ok=False)
    for p in (
        c.AUTHORITY,
        c.LAUNCH,
        c.RESULTS / "memory_preflight.json",
        c.RESULTS / "resume_preflight.json",
        c.RESULTS / "training_summary.json",
        c.RESULTS / "snapshot_manifest.json",
        c.RUN_DIR / "campaign_state.json",
    ):
        shutil.copy2(p, archive / p.name)
    c.engine.write_json(c.AUTHORITY, new)
    c.engine.write_json(c.LAUNCH, c.launch_payload())
    ah, lh = c.engine.sha256_file(c.AUTHORITY), c.engine.sha256_file(c.LAUNCH)
    after = rebind(before, ah, lh)
    unchanged = {
        k: c.tree_sha256(before[k]) == c.tree_sha256(after[k])
        for k in before
        if k not in {"source", "phase_transition"}
    }
    assert all(unchanged.values())
    c.validate_resume_payload(after)
    dest = c.RUN_DIR / "execution_optimized_u0050.pt"
    assert not dest.exists()
    torch.save(after, dest)
    record = {
        "source": str(Path(args.resume).resolve()),
        "source_sha256": report["source_sha256"],
        "resume": str(dest),
        "resume_sha256": c.engine.sha256_file(dest),
        "accepted_update": 50,
        "authority_sha256": ah,
        "launch_authority_sha256": lh,
        "unchanged_checkpoint_fields": unchanged,
        "model_sha256": c.tree_sha256(after["model"]),
        "optimizer_sha256": c.tree_sha256(after["optimizer"]),
        "optimizer_step_taken": False,
        "benchmark_sha256": c.engine.sha256_file(evidence),
    }
    c.engine.write_json(c.RESULTS / "execution_optimization/transition.json", record)
    for p in (c.RUN_DIR / "campaign_state.json", c.RESULTS / "snapshot_manifest.json"):
        data = json.loads(p.read_text())
        data.update(authority_sha256=ah, schedule_authority_sha256=lh)
        c.engine.write_json(p, data)
    memory = copy.deepcopy(report["production_preflight"])
    memory.update(
        authority_sha256=ah,
        verdict="PASS",
        optimizer_step_taken=False,
        source_checkpoint_sha256=report["source_sha256"],
        rollout_horizon=360,
        rollout_logical_gib=report["rollout_logical_gib"],
        cuda_peak_allocated_gib=report["cuda_peak_allocated_gib"],
        cuda_peak_reserved_gib=report["cuda_peak_reserved_gib"],
        evidence="execution_optimization/exact_scale.json",
        evidence_sha256=c.engine.sha256_file(evidence),
    )
    memory["checks"].update(report["checks"])
    c.engine.write_json(c.RESULTS / "memory_preflight.json", memory)
    assert (
        json.loads((archive / "campaign_state.json").read_text())["deadline_unix"]
        == json.loads((c.RUN_DIR / "campaign_state.json").read_text())["deadline_unix"]
    )
    c.load_authority()
    c.load_launch_authority()
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
