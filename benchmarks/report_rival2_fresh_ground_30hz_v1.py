"""CPU-only, read-only checkpoint audit plus evidence publication for the monitor.

Never imports old weights, changes training, or launches competing GPU work.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from benchmarks.run_rival2_fresh_ground_30hz_v1 import RESULTS, write_json, utc, tensor_hash
from rivalsim.fresh_ground_30hz import VERSION, CHECKPOINT_FORMAT, authority, content_hash


def finite_tree(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite_tree(v) for v in value)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="G:/dev/RivalSim-runs/fresh-ground-30hz-v1")
    args = p.parse_args()
    run = Path(args.run_dir)
    # Atomic writer publishes data before its pointer. A two-slot rotation can
    # race a reader; retry rather than mmap or lock a live checkpoint.
    for _ in range(3):
        latest = json.loads((run / "latest.json").read_text())
        data = Path(latest["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest().upper() == latest["sha256"]:
            break
    else:
        raise RuntimeError("checkpoint rotated during all read attempts")
    payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
    offset = payload["accepted_updates_total"]
    package = json.loads((RESULTS / "package.json").read_text())
    opt_groups = payload["optimizer"]["param_groups"]
    checks = {
        "checkpoint_pointer_matches": offset == latest["accepted_updates"],
        "fresh_lineage": payload["format"] == CHECKPOINT_FORMAT and payload["lineage"] == VERSION
                         and payload["source"]["parent"] is None,
        "frozen_authority": payload["authority_sha256"] == content_hash(authority()),
        "cadence": payload["policy_hz"] == 30 and payload["physics_hz"] == 120,
        "finite_model": finite_tree(payload["model"]),
        "finite_adam": finite_tree(payload["optimizer"]),
        "actor_critic_lrs": [g["lr"] for g in opt_groups] == [1e-4, 3e-4],
        "no_old_policy_opponents": payload["opponents"]["historical"] is False,
        "kl_telemetry_only": payload["kl_policy"]["telemetry_only"]
                             and not payload["kl_policy"]["kl_rollback"],
        "nonfinite_rollback_preserved": payload["kl_policy"]["nonfinite_transactional_rollback"],
        "policy_has_learned_new_weights": offset == 0 or tensor_hash(payload["model"]) != package["initial_model_sha256"],
        "adam_has_real_steps": offset == 0 or (bool(payload["optimizer"]["state"])
                and all(float(v["step"]) > 0 for v in payload["optimizer"]["state"].values())),
    }
    report = {"verdict": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
              "utc": utc(), "checkpoint": latest, "accepted_updates": offset,
              "total_trainable_samples": payload["total_agent_samples"],
              "physical_world_ticks": payload["physical_physics_ticks_experienced"],
              "optimizer_parameter_state_count": len(payload["optimizer"]["state"]),
              "last_rollout_metrics": payload["last_rollout_metrics"],
              "campaign_state_at_audit": json.loads((run / "campaign_state.json").read_text()),
              "learning_verdict": "not determined by finite-state audit; inspect deterministic evaluations"}
    if (run / "latest_evaluation.json").exists():
        report["latest_completed_evaluation"] = json.loads((run / "latest_evaluation.json").read_text())
    if (run / "training_curve.jsonl").exists():
        # Retain only complete, accepted records if the writer is concurrently appending.
        records = []
        for line in (run / "training_curve.jsonl").read_text().splitlines(keepends=True):
            if not line.endswith("\n"):
                continue
            row = json.loads(line)
            if row["update"] <= offset:
                records.append(row)
        write_json(RESULTS / "monitoring" / f"curve_through_u{offset:06d}.json", records)
    write_json(RESULTS / "monitoring" / f"u{offset:06d}.json", report)
    print(json.dumps({k: report[k] for k in ("verdict", "accepted_updates", "total_trainable_samples", "checkpoint")}))
    if report["verdict"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
