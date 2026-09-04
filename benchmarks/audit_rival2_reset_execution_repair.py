"""Read-only audit of a saved, accepted production update after execution repair."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_rival2_ssl_foundation_v5_long_trace_v1 as campaign  # noqa: E402


def run(args):
    torch.set_num_threads(2)
    checkpoint = Path(args.checkpoint)
    before = campaign.engine.sha256_file(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    campaign.configure_engine()
    campaign.load_authority()
    campaign.load_launch_authority()
    campaign.validate_resume_payload(payload)
    startup = torch.load(campaign.STARTUP, map_location="cpu", weights_only=False)
    original = torch.load(
        campaign.RUN_DIR / "performance_repair_original/transition_u0010.pt",
        map_location="cpu",
        weights_only=False,
    )
    records = [
        json.loads(line)
        for line in (campaign.RESULTS / "training_curve.jsonl").read_text().splitlines()
    ]
    accepted = payload["accepted_updates_total"]
    records = [r for r in records if r["accepted_update"] <= accepted]
    expected_steps = sum(int(r["ppo"]["optimizer_steps"]) for r in records)
    step_deltas = {
        str(key): int(state["step"]) - int(startup["optimizer"]["state"][key]["step"])
        for key, state in payload["optimizer"]["state"].items()
    }
    protected = (
        "model",
        "optimizer",
        "policy_generator_state",
        "shuffle_generator_state",
        "torch_cpu_rng_state",
        "torch_cuda_rng_state",
        "total_agent_samples",
        "accepted_updates_total",
        "policy_version",
        "ppo_config",
        "contract_hashes",
    )
    current_state = json.loads((campaign.RUN_DIR / "campaign_state.json").read_text())
    original_state = json.loads(
        (campaign.RESULTS / "performance_repair/original/campaign_state.json").read_text()
    )
    checks = {
        "source_checkpoint_unchanged": campaign.engine.sha256_file(campaign.PARENT)
        == campaign.PARENT_SHA256,
        "protected_startup_state_unchanged": all(
            campaign.tree_sha256(startup[k]) == campaign.tree_sha256(original[k]) for k in protected
        ),
        "model_finite": all(bool(torch.isfinite(t).all()) for t in payload["model"].values()),
        "optimizer_finite": all(
            bool(torch.isfinite(t).all())
            for state in payload["optimizer"]["state"].values()
            for t in state.values()
            if isinstance(t, torch.Tensor)
        ),
        "optimizer_steps_exact": all(delta == expected_steps for delta in step_deltas.values()),
        "accepted_boundaries_contiguous": [r["accepted_update"] for r in records]
        == list(range(11, accepted + 1)),
        "samples_accounted": payload["total_agent_samples"]
        == startup["total_agent_samples"]
        + sum(r["rollout"]["trainable_agent_samples"] for r in records),
        "physics_ticks_accounted": payload["physical_physics_ticks_experienced"]
        == startup["physical_physics_ticks_experienced"] + len(records) * 32768 * 360,
        "original_deadline_unchanged": current_state["deadline_unix"]
        == original_state["deadline_unix"],
        "no_hard_failure": not (campaign.RESULTS / "hard_failure.json").exists(),
        "no_kl_rejection_or_rollback": payload["kl_policy"]["telemetry_only"]
        and not payload["kl_policy"]["kl_rollback"],
        "checkpoint_unchanged_during_audit": before == campaign.engine.sha256_file(checkpoint),
    }
    report = {
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": before,
        "accepted_updates": accepted,
        "total_agent_samples": payload["total_agent_samples"],
        "authority_sha256": campaign.engine.sha256_file(campaign.AUTHORITY),
        "optimizer_steps_since_repair": expected_steps,
        "optimizer_step_deltas": step_deltas,
        "production_updates": [
            {
                "accepted_update": r["accepted_update"],
                "wall_seconds": r["wall_seconds"],
                "peak_allocated_gib": r["ppo"]["cuda_peak_allocated_gib"],
                "peak_reserved_gib": r["ppo"]["cuda_peak_reserved_gib"],
                "optimizer_steps": r["ppo"]["optimizer_steps"],
            }
            for r in records
        ],
        "no_learning_in_audit": True,
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["verdict"] == "PASS" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    raise SystemExit(run(parser.parse_args()))
