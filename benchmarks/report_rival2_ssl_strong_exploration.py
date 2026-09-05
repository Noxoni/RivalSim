"""CPU-only audit/publication of accepted stronger-exploration learning state."""

import hashlib
import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402

from benchmarks import run_rival2_ssl_foundation_strong_exploration as s  # noqa: E402


def curriculum_contract_preserved(parent, checkpoint):
    # The per-world family is live episode state: valid resets must change it.
    # Compare the frozen generation/scheduling contract, not episode assignments.
    old = parent["reset_curriculum"]
    new = checkpoint["reset_curriculum"]
    family = new["scenario_family"]
    return (
        set(old) == set(new)
        and all(
            s.c.tree_sha256(new[k]) == s.c.tree_sha256(old[k])
            for k in old
            if k != "scenario_family"
        )
        and family.shape == old["scenario_family"].shape
        and family.dtype == old["scenario_family"].dtype
        and bool(((family >= 0) & (family < len(new["summary"]["counts"]))).all())
    )


def main():
    torch.set_num_threads(2)
    s.configure()
    s.load_authority()
    s.load_launch()
    transition = json.loads((s.EVIDENCE / "transition.json").read_text())
    parent = torch.load(transition["resume"], map_location="cpu", weights_only=False)
    rolling = s.c.RUN_DIR / "rolling.pt"
    for _ in range(5):
        before = rolling.stat()
        raw = rolling.read_bytes()
        after = rolling.stat()
        if (before.st_mtime_ns, before.st_size) == (after.st_mtime_ns, after.st_size):
            checkpoint = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
            break
        time.sleep(0.2)
    else:
        raise RuntimeError("rolling checkpoint is being replaced; retry read-only audit")
    s.validate_resume(checkpoint)
    accepted = checkpoint["accepted_updates_total"]
    anchor = transition["accepted_update"]
    if not anchor < accepted <= 100:
        raise ValueError("wait for a stronger-exploration accepted update")
    rows = [
        json.loads(line)
        for line in (s.c.RESULTS / "training_curve.jsonl").read_text().splitlines(keepends=True)
        if line.endswith("\n")
    ]
    rows = [row for row in rows if anchor < row["accepted_update"] <= accepted]
    expected_steps = sum(int(row["ppo"]["optimizer_steps"]) for row in rows)
    adam = checkpoint["optimizer"]["state"]
    old_adam = parent["optimizer"]["state"]
    deltas = {str(key): float(value["step"] - old_adam[key]["step"]) for key, value in adam.items()}
    old_state = json.loads((s.EVIDENCE / "original/campaign_state.json").read_text())
    state = json.loads((s.c.RUN_DIR / "campaign_state.json").read_text())
    same = (
        "lineage",
        "source",
        "phase_transition",
        "policy_config",
        "ppo_config",
        "contract_hashes",
        "physics_hz",
        "policy_hz",
        "opponents",
        "kl_policy",
    )
    checks = {
        "contiguous_accepted_updates": [r["accepted_update"] for r in rows]
        == list(range(anchor + 1, accepted + 1)),
        "correct_exploration_every_update": all(
            r["exploration"] == s.exploration_for_update(r["accepted_update"] - 1).as_dict()
            for r in rows
        ),
        "722_adam_steps_per_update": all(r["ppo"]["optimizer_steps"] == 722 for r in rows),
        "all_adam_counters_preserved_and_advanced": all(
            d == expected_steps for d in deltas.values()
        ),
        "learning_contracts_preserved": all(
            s.c.tree_sha256(checkpoint[key]) == s.c.tree_sha256(parent[key]) for key in same
        ),
        "reset_curriculum_contract_preserved": curriculum_contract_preserved(parent, checkpoint),
        "model_finite": all(bool(torch.isfinite(v).all()) for v in checkpoint["model"].values()),
        "adam_finite": all(
            bool(torch.isfinite(v).all())
            for st in adam.values()
            for v in st.values()
            if isinstance(v, torch.Tensor)
        ),
        "original_deadline_preserved": state["deadline_unix"] == old_state["deadline_unix"],
        "total100_bound": state["continuation_review_marker"] == 100,
        "source_checkpoint_unchanged": s.c.engine.sha256_file(s.ROOT / transition["source"])
        == transition["source_sha256"],
        "rebound_checkpoint_unchanged": s.c.engine.sha256_file(Path(transition["resume"]))
        == transition["resume_sha256"],
        "no_hard_failure": not (s.c.RESULTS / "hard_failure.json").exists(),
        "stderr_empty": (s.c.RUN_DIR / "campaign.strong_exploration.stderr.log").stat().st_size
        == 0,
    }
    artifact = s.c.CHECKPOINT.parent / f"strong_exploration_verified_u{accepted:04d}.pt"
    digest = hashlib.sha256(raw).hexdigest().upper()
    if artifact.exists():
        if s.c.engine.sha256_file(artifact) != digest:
            raise ValueError("refuse to overwrite different accepted checkpoint")
    else:
        artifact.write_bytes(raw)
    report = {
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "accepted_update": accepted,
        "stronger_exploration_updates": accepted - anchor,
        "checkpoint": artifact.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": digest,
        "authority_sha256": s.c.engine.sha256_file(s.c.AUTHORITY),
        "launch_authority_sha256": s.c.engine.sha256_file(s.c.LAUNCH),
        "adam_step_deltas": deltas,
        "training_rows": rows,
        "total_agent_samples": checkpoint["total_agent_samples"],
        "physical_ticks": checkpoint["physical_physics_ticks_experienced"],
        "interpretation": (
            "accepted learning/exploration integrity; gameplay improvement not yet evaluated"
        ),
    }
    s.c.engine.write_json(s.EVIDENCE / "production_verification.json", report)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "training_rows"}, indent=2
        )
    )
    if not all(checks.values()):
        raise RuntimeError("production verification failed")


if __name__ == "__main__":
    main()
