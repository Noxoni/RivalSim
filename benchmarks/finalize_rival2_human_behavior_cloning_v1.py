"""Finalize human-BC V1 evidence without another model or optimizer mutation."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT_PATH = Path(__file__).resolve().parents[1]
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

from benchmarks.run_rival2_human_behavior_cloning_v1 import (  # noqa: E402
    RESULT_ROOT,
    ROOT,
    _artifact_manifest,
    _write_json,
)
from rivalsim.human_demo.missing_feature_distillation import file_sha256  # noqa: E402


def _rename_normalized_speed(value: dict[str, Any]) -> None:
    for label in ("bootstrap", "selected"):
        row = value[label]
        if "mean_self_speed_uu_per_s" in row:
            row["mean_normalized_self_speed"] = row.pop("mean_self_speed_uu_per_s")


def main() -> int:
    evidence_path = ROOT / RESULT_ROOT / "evidence.json"
    sanity_path = ROOT / RESULT_ROOT / "gameplay_sanity.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    sanity = json.loads(sanity_path.read_text(encoding="utf-8"))
    checkpoint = ROOT / evidence["checkpoint"]["path"]
    checkpoint_before = file_sha256(checkpoint)
    if checkpoint_before != evidence["checkpoint"]["sha256"]:
        raise ValueError("selected human-BC checkpoint changed before evidence finalization")
    _rename_normalized_speed(sanity)
    _rename_normalized_speed(evidence["gameplay_sanity"])
    attempts = evidence["training"]["interval_attempts"]
    rejected = [row for row in attempts if not row["retention_guard"]["accepted"]]
    evidence["post_training_evidence_finalization"] = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "optimizer_steps": 0,
        "model_mutation": False,
        "test_reopened": False,
        "closed_loop_rerun": False,
        "checkpoint_sha256_before_after": [checkpoint_before, file_sha256(checkpoint)],
        "unit_correction": {
            "from": "mean_self_speed_uu_per_s",
            "to": "mean_normalized_self_speed",
            "values_changed": False,
            "reason": "RIVAL2_OBS_V2_120HZ self velocity is normalized by the car speed scale",
        },
        "guard_stop_audit": {
            "selected_checkpoint_precedes_rejected_boundary": (
                evidence["training"]["selected_accepted_step"]
                < min(row["attempted_accepted_step"] for row in rejected)
            ),
            "rejected_boundary": min(row["attempted_accepted_step"] for row in rejected),
            "all_rejections_only_critic_max_absolute_drift": all(
                not row["retention_guard"]["checks"]["critic_max_absolute_drift"]
                and all(
                    passed
                    for name, passed in row["retention_guard"]["checks"].items()
                    if name != "critic_max_absolute_drift"
                )
                for row in rejected
            ),
            "retry_learning_rates": [row["learning_rate"] for row in rejected],
            "duplicate_backoff_retries_disclosed": True,
            "note": (
                "Rollback restored the interval-start optimizer LR before retries 2 and 3, "
                "so they repeated retry 1 at 1.5e-5. The hard guard remained enforced, the "
                "campaign stopped, and the selected step-160 checkpoint predates the rejected "
                "step-192 boundary. No limit was weakened."
            ),
        },
    }
    _write_json(sanity_path, sanity)
    _write_json(evidence_path, evidence)
    artifact_paths = [
        Path(evidence["checkpoint"]["path"]),
        RESULT_ROOT / "REVIEW.md",
        RESULT_ROOT / "closed_loop_mechanic_evaluation.json",
        RESULT_ROOT / "corpus_manifest.json",
        RESULT_ROOT / "evidence.json",
        RESULT_ROOT / "final_test_metrics.json",
        RESULT_ROOT / "frozen_config.json",
        RESULT_ROOT / "gameplay_sanity.json",
        RESULT_ROOT / "pre_step_authority.json",
        RESULT_ROOT / "pre_step_preflight.json",
        RESULT_ROOT / "simulator_retention_test.json",
        RESULT_ROOT / "training_curve.json",
        RESULT_ROOT / "verification_evidence.json",
    ]
    _write_json(ROOT / RESULT_ROOT / "artifact_manifest.json", _artifact_manifest(artifact_paths))
    if file_sha256(checkpoint) != checkpoint_before:
        raise RuntimeError("evidence finalization mutated the selected checkpoint")
    print(
        json.dumps(
            {
                "verdict": evidence["verdict"],
                "checkpoint_sha256": checkpoint_before,
                "optimizer_steps": 0,
                "test_reopened": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
