"""Publish existing completed update-100 evidence; CPU reads, no policy evaluation."""

import json
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402

from benchmarks import run_rival2_ssl_foundation_strong_exploration as s  # noqa: E402


def finite_tree(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite_tree(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def read(path):
    return json.loads(path.read_text())


def recover_publication(manifest):
    """Recover only the hash-bound completed run; never touch the locked old file."""
    authority_path = s.EVIDENCE / "publication_recovery_authority.json"
    authority = read(authority_path)
    targets = {
        "stderr": s.c.RUN_DIR / "campaign.strong_exploration.stderr.log",
        "old_final": s.c.CHECKPOINT,
        "rolling": s.c.RUN_DIR / "rolling.pt",
        "snapshot": s.c.RUN_DIR / "snapshots/ssl_foundation_u0100.pt",
    }
    for name, path in targets.items():
        if s.c.engine.sha256_file(path) != authority[name + "_sha256"]:
            raise ValueError(f"publication recovery input changed: {name}")
    final_path = ROOT / authority["final_path"]
    if s.c.engine.sha256_file(final_path) != authority["rolling_sha256"]:
        raise ValueError("published rolling100 bytes do not match recovery authority")
    payload = torch.load(final_path, map_location="cpu", weights_only=False)
    snapshot = torch.load(targets["snapshot"], map_location="cpu", weights_only=False)
    s.validate_resume(payload)
    if not finite_tree(payload) or payload["accepted_updates_total"] != 100:
        raise ValueError("invalid final learning state")
    if s.c.tree_sha256(snapshot) != s.c.tree_sha256(payload):
        raise ValueError("snapshot and rolling100 payloads differ")
    completed = [e for e in manifest["evaluations"] if e["accepted_updates"] == 100]
    if len(completed) != 1 or not finite_tree(completed):
        raise ValueError("missing/duplicate/nonfinite completed evaluation")
    if (s.c.RESULTS / "hard_failure.json").exists():
        raise ValueError("training guard evidence present; not a publication-only recovery")
    for name, path in {
        "stderr": targets["stderr"],
        "stdout": s.c.RUN_DIR / "campaign.strong_exploration.stdout.log",
    }.items():
        archived = s.EVIDENCE / f"publication_failure.{name}.log"
        if not archived.exists():
            shutil.copyfile(path, archived)
    old_summary = s.EVIDENCE / "publication_failure_old_summary.json"
    if not old_summary.exists():
        shutil.copyfile(s.c.RESULTS / "training_summary.json", old_summary)
    record = {
        "accepted_updates": 100,
        "bytes": final_path.stat().st_size,
        "path": authority["final_path"],
        "policy_version": payload["policy_version"],
        "schedule_authority_sha256": s.c.engine.sha256_file(s.c.LAUNCH),
        "sha256": authority["rolling_sha256"],
        "total_agent_samples": payload["total_agent_samples"],
    }
    manifest["final"] = record
    summary = {
        "format": f"{s.c.FORMAT}_TRAINING_SUMMARY",
        "created_utc": s.c.engine.utc_now(),
        "verdict": "PASS",
        "verdict_scope": "completed campaign; not gameplay improvement",
        "stop_reason": "continuation_review_marker",
        "accepted_updates": 100,
        "final_checkpoint": record,
        "hard_failure": None,
        "authority_sha256": s.c.engine.sha256_file(s.c.AUTHORITY),
        "schedule_authority_sha256": s.c.engine.sha256_file(s.c.LAUNCH),
        "source_sha256": s.c.engine.SOURCE_SHA256,
        "last_evaluation": completed[0],
        "operational_recovery": {
            "authority_sha256": s.c.engine.sha256_file(authority_path),
            "reason": authority["error"],
            "checkpoint_origin": "exact accepted rolling100, pre-evaluation RNG preserved",
            "additional_optimizer_steps": 0,
            "additional_evaluations": 0,
            "old_final_unchanged": True,
            "snapshot_and_rolling_payload_exact_parity": True,
        },
    }
    s.c.engine.write_json(s.c.RESULTS / "snapshot_manifest.json", manifest)
    s.c.engine.write_json(s.c.RESULTS / "training_summary.json", summary)
    return summary


def main():
    torch.set_num_threads(2)
    s.no_training_process()
    s.configure()
    s.load_authority()
    s.load_launch()
    summary = read(s.c.RESULTS / "training_summary.json")
    manifest = read(s.c.RESULTS / "snapshot_manifest.json")
    production = read(s.EVIDENCE / "production_verification.json")
    if "--recover-publication" in sys.argv:
        summary = recover_publication(manifest)
    if summary["accepted_updates"] != 100 or production["accepted_update"] != 100:
        raise ValueError("final update 100 and its CPU production audit must already be complete")
    final_record = summary["final_checkpoint"]
    final_path = ROOT / final_record["path"]
    final = torch.load(final_path, map_location="cpu", weights_only=False)
    rolling = torch.load(s.c.RUN_DIR / "rolling.pt", map_location="cpu", weights_only=False)
    s.validate_resume(final)
    rows = [
        json.loads(line) for line in (s.c.RESULTS / "training_curve.jsonl").read_text().splitlines()
    ]
    strong_rows = [r for r in rows if 84 < r["accepted_update"] <= 100]
    evaluations = {e["accepted_updates"]: e for e in manifest["evaluations"]}
    final_eval = evaluations[100]
    fields = (
        "goals_for",
        "goals_against",
        "goal_differential",
        "touches_per_minute",
        "mean_speed_uu_per_second",
        "no_touch_resets",
        "goalward_touch_fraction",
    )
    comparison = {}
    for opponent in ("nexto", "frozen_unified_v5"):
        values = {}
        for update in (10, 20, 50, 84, 100):
            metrics = dict(evaluations[update]["opponents"][opponent])
            metrics["goal_differential"] = metrics["goals_for"] - metrics["goals_against"]
            values[str(update)] = {k: metrics[k] for k in fields}
        comparison[opponent] = {
            "evaluations": values,
            "deltas_100_minus": {
                str(base): {k: values["100"][k] - values[str(base)][k] for k in fields}
                for base in (10, 50, 84)
            },
        }
    publication = []
    for record in (final_record, *manifest["snapshots"]):
        path = ROOT / record["path"]
        if record["accepted_updates"] != 100:
            continue
        digest = s.c.engine.sha256_file(path)
        if digest != record["sha256"]:
            raise ValueError(f"checkpoint manifest hash mismatch: {path}")
        publication.append({**record, "verified_sha256": digest})
    checks = {
        "completed_review_marker100": summary["stop_reason"] == "continuation_review_marker",
        "no_failure": summary["hard_failure"] is None and summary["verdict"] == "PASS",
        "production_audit_pass": production["verdict"] == "PASS"
        and all(production["checks"].values()),
        "final_finite": finite_tree(final),
        "curve_and_evaluations_finite": finite_tree(rows) and finite_tree(manifest["evaluations"]),
        "final_policy_version100": final["accepted_updates_total"]
        == final["policy_version"]
        == 100,
        "no_updates_past100": max(r["accepted_update"] for r in rows) == 100,
        "16_stronger_updates": [r["accepted_update"] for r in strong_rows] == list(range(85, 101)),
        "final_exploration_exact": final["exploration"] == s.exploration_for_update(99).as_dict(),
        "final_matches_accepted_rolling_model_and_adam": all(
            s.c.tree_sha256(final[k]) == s.c.tree_sha256(rolling[k]) for k in ("model", "optimizer")
        ),
        "evaluation_once_at100": sum(e["accepted_updates"] == 100 for e in manifest["evaluations"])
        == 1,
        "matched_protocol": all(
            evaluations[u]["ticks"] == 3600
            and evaluations[u]["worlds"] == 1024
            and evaluations[u]["deterministic_policy"]
            for u in (10, 20, 50, 84, 100)
        ),
        "summary_evaluation_matches_manifest": summary["last_evaluation"] == final_eval,
        "authority_bound": summary["authority_sha256"] == s.c.engine.sha256_file(s.c.AUTHORITY),
        "launch_bound": summary["schedule_authority_sha256"] == s.c.engine.sha256_file(s.c.LAUNCH),
        "original_v5_unchanged": s.c.engine.sha256_file(s.c.engine.SOURCE)
        == s.c.engine.SOURCE_SHA256,
    }
    report = {
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "verdict_scope": "campaign completion and checkpoint integrity, not improved gameplay",
        "checks": checks,
        "created_utc": s.c.engine.utc_now(),
        "final_checkpoint": final_record,
        "operational_recovery": summary.get("operational_recovery"),
        "checkpoint_hash_verification": publication,
        "accepted_updates": 100,
        "stronger_exploration_updates": 16,
        "stronger_trainable_samples": sum(
            r["rollout"]["trainable_agent_samples"] for r in strong_rows
        ),
        "total_trainable_samples": final["total_agent_samples"],
        "physical_ticks": final["physical_physics_ticks_experienced"],
        "stronger_update_wall_seconds": sum(r["wall_seconds"] for r in strong_rows),
        "stronger_optimizer_steps": sum(int(r["ppo"]["optimizer_steps"]) for r in strong_rows),
        "stronger_ppo_telemetry": {
            k: {
                "minimum": min(r["ppo"][k] for r in strong_rows),
                "maximum": max(r["ppo"][k] for r in strong_rows),
            }
            for k in (
                "completed_update_mean_kl",
                "optimizer_post_step_approx_kl_max",
                "post_clip_gradient_norm",
                "cuda_peak_allocated_gib",
            )
        },
        "comparison": comparison,
        "protocol_limits": [
            "1024 scenarios total, 512 per opponent, 3600 ticks; not full-match win rates",
            "fixed scenario corpus and side assignment; matched before/after, "
            "not broad generalization",
            "goalward touch is ball velocity toward opponent end after contact, "
            "not confirmed aimed shot",
            "no possession duration, reverse-driving or mechanic competence is inferred",
            "stochastic training is not directly comparable with deterministic evaluation",
            "only 16 updates used stronger exploration; restart also initializes fresh episodes",
        ],
        "training_or_evaluation_launched_by_this_audit": False,
    }
    s.c.engine.write_json(s.EVIDENCE / "completion_audit.json", report)
    print(json.dumps(report, indent=2))
    if not all(checks.values()):
        raise RuntimeError("completion audit failed")


if __name__ == "__main__":
    main()
