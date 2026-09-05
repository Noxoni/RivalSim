"""Read-only learning-state audit and bounded publication after optimized resume."""

import hashlib
import io
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks import run_rival2_ssl_foundation_v5_long_trace_v1 as c  # noqa: E402


def main():
    c.configure_engine()
    c.load_authority()
    c.load_launch_authority()
    transition = json.loads((c.RESULTS / "execution_optimization/transition.json").read_text())
    parent_path = Path(transition["resume"])
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    raw = (c.RUN_DIR / "rolling.pt").read_bytes()
    checkpoint = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    c.validate_resume_payload(checkpoint)
    accepted = checkpoint["accepted_updates_total"]
    assert 52 <= accepted <= 100
    digest = hashlib.sha256(raw).hexdigest().upper()
    artifact = (
        ROOT
        / "checkpoints/rival2/ssl_foundation_v5_long_trace_v1"
        / f"execution_verified_u{accepted:04d}.pt"
    )
    if artifact.exists():
        assert c.engine.sha256_file(artifact) == digest
    else:
        artifact.write_bytes(raw)
    rows = [
        json.loads(x)
        for x in (c.RESULTS / "training_curve.jsonl").read_text().splitlines(keepends=True)
        if x.endswith("\n")
    ]
    rows = [r for r in rows if r["accepted_update"] <= accepted]
    before = [r for r in rows if 44 <= r["accepted_update"] <= 50]
    after = [r for r in rows if 51 <= r["accepted_update"] <= accepted]
    assert len(after) == accepted - 50
    optimizer = checkpoint["optimizer"]["state"]
    old_optimizer = parent["optimizer"]["state"]
    deltas = {str(k): float(v["step"] - old_optimizer[k]["step"]) for k, v in optimizer.items()}
    contracts = (
        "policy_config",
        "ppo_config",
        "contract_hashes",
        "lineage",
        "source",
        "policy_hz",
        "physics_hz",
        "opponents",
    )
    checks = {
        "finite_model": all(bool(torch.isfinite(v).all()) for v in checkpoint["model"].values()),
        "finite_optimizer": all(
            bool(torch.isfinite(v).all())
            for state in optimizer.values()
            for v in state.values()
            if isinstance(v, torch.Tensor)
        ),
        "same_model_parameter_keys": checkpoint["model"].keys() == parent["model"].keys(),
        "every_adam_counter_exact": all(d == (accepted - 50) * 722 for d in deltas.values()),
        "all_update_step_counts_exact": all(r["ppo"]["optimizer_steps"] == 722 for r in after),
        "learning_contracts_unchanged": all(
            c.tree_sha256(checkpoint[k]) == c.tree_sha256(parent[k]) for k in contracts
        ),
        "preserved_parent_unchanged": c.engine.sha256_file(parent_path)
        == transition["resume_sha256"],
        "original_update50_unchanged": c.engine.sha256_file(Path(transition["source"]))
        == transition["source_sha256"],
        "deadline_unchanged": json.loads((c.RUN_DIR / "campaign_state.json").read_text())[
            "deadline_unix"
        ]
        == json.loads(
            (c.RESULTS / "execution_optimization/original/campaign_state.json").read_text()
        )["deadline_unix"],
        "no_hard_failure": not (c.RESULTS / "hard_failure.json").exists(),
        "stderr_empty": (c.RUN_DIR / "campaign.optimized.stderr.log").stat().st_size == 0,
    }
    old_seconds = statistics.mean(r["wall_seconds"] for r in before)
    new_seconds = statistics.mean(r["wall_seconds"] for r in after)
    report = {
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "created_utc": c.engine.utc_now(),
        "checks": checks,
        "accepted_update": accepted,
        "checkpoint_path": artifact.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": digest,
        "parent_update50_sha256": transition["resume_sha256"],
        "source_original_update50_sha256": transition["source_sha256"],
        "authority_sha256": c.engine.sha256_file(c.AUTHORITY),
        "launch_authority_sha256": c.engine.sha256_file(c.LAUNCH),
        "adam_step_deltas": deltas,
        "total_trainable_samples": checkpoint["total_agent_samples"],
        "physical_ticks": checkpoint["physical_physics_ticks_experienced"],
        "baseline_update_rows": before,
        "optimized_update_rows": after,
        "baseline_mean_seconds": old_seconds,
        "optimized_mean_seconds": new_seconds,
        "observed_elapsed_reduction_percent": 100 * (1 - new_seconds / old_seconds),
        "observed_throughput_factor": old_seconds / new_seconds,
        "timing_caveat": (
            "Sequential live windows, not paired identical rollouts. "
            "Use exact_scale.json for matched-input measurements."
        ),
        "resume_semantics": (
            "Existing fresh simulator episodes/zero hidden; "
            "saved model/Adam/counters/RNG preserved at transition."
        ),
        "large_kl_is_failure": False,
    }
    c.engine.write_json(c.RESULTS / "execution_optimization/production_verification.json", report)
    manifest = json.loads((c.RESULTS / "snapshot_manifest.json").read_text())
    evaluations = {
        r["accepted_updates"]: r
        for r in manifest["evaluations"]
        if r["accepted_updates"] in (10, 20, 50)
    }
    comparison = {
        "protocol": "1024 scenario worlds, 3600 ticks, deterministic; not full-match win rates",
        "evaluations": evaluations,
        "update50_checkpoint": transition["source"],
        "update50_sha256": transition["source_sha256"],
        "before_execution_optimization": True,
        "deltas": {},
    }
    for index in (10, 20):
        comparison["deltas"][f"{index}_to_50"] = {}
        for opponent, current in evaluations[50]["opponents"].items():
            baseline = evaluations[index]["opponents"][opponent]
            comparison["deltas"][f"{index}_to_50"][opponent] = {
                k: current[k] - baseline[k]
                for k in (
                    "goals_for",
                    "goals_against",
                    "touches_per_minute",
                    "no_touch_resets",
                    "mean_speed_uu_per_second",
                    "goalward_touch_fraction",
                )
            }
    c.engine.write_json(c.RESULTS / "evaluation_u0050_comparison.json", comparison)
    print(json.dumps({k: v for k, v in report.items() if not k.endswith("_rows")}, indent=2))
    assert all(checks.values())


if __name__ == "__main__":
    main()
