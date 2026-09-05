"""Preserve the zero-accepted-update device-check fix before resuming PPO."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from benchmarks.run_rival2_fresh_ground_30hz_v1 import sha, tensor_hash, utc, write_json
from benchmarks.run_rival2_ssl_entity_joint_control import (
    RESULTS,
    EXTERNAL,
    INITIAL,
    authority,
    content_hash,
)


def main():
    destination = RESULTS / "operational_recovery_v1.json"
    assert not destination.exists(), "Recovery already frozen"
    package = json.loads((RESULTS / "package.json").read_text())
    failure = json.loads((RESULTS / "failure_initial_cpu_adam_check.json").read_text())
    latest = json.loads((EXTERNAL / "latest.json").read_text())
    path = Path(latest["path"])
    assert failure["accepted_updates"] == latest["accepted_updates"] == 0
    assert sha(path) == latest["sha256"] == failure["latest_checkpoint"]["sha256"]
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    initial = torch.load(INITIAL, map_location="cpu", weights_only=False)
    assert (
        tensor_hash(checkpoint["model"])
        == tensor_hash(initial["model"])
        == package["initial_model_sha256"]
    )
    assert checkpoint["optimizer"]["state"] == initial["optimizer"]["state"] == {}
    changed = [
        "rivalsim/ssl_entity_training.py",
        "tests/test_ssl_entity_training.py",
        "benchmarks/run_rival2_ssl_entity_joint_control.py",
    ]
    write_json(
        destination,
        dict(
            utc=utc(),
            version="ENTITY_CPU_ADAM_FINITE_CHECK_RECOVERY_V1",
            base_package_sha256=sha(RESULTS / "package.json"),
            authority_sha256=content_hash(authority()),
            accepted_updates_before_recovery=0,
            last_accepted_checkpoint=latest,
            initialized_model_unchanged=True,
            initialized_optimizer_empty=True,
            numerical_checks_preserved=True,
            change="Group finite checks by device. Check all CUDA parameters/moments AND CPU Adam step counters. No guard, loss, learning-rate, reward or model change. Runtime source override is explicitly limited to these three committed files.",
            source_changes={
                n: dict(before_sha256=package["sources"][n], after_sha256=sha(ROOT / n))
                for n in changed
            },
            tests_sha256=sha(RESULTS / "operational_recovery_tests.xml"),
            failure_sha256=sha(RESULTS / "failure_initial_cpu_adam_check.json"),
            rollback="Update exception handler completed model/Adam/shuffle-RNG restoration; no accepted update. Immutable saved zero-update checkpoint exactly matches initialization and empty Adam.",
            diagnostic_only=True,
        ),
    )
    print(json.dumps(dict(status="recovery_frozen_requires_push", checkpoint=latest)))


if __name__ == "__main__":
    main()
