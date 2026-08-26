import hashlib
import json
from pathlib import Path

import torch

from benchmarks.run_rival2_campaign01 import (
    EXPECTED_CONTRACT_HASHES,
    EXPECTED_POLICY_CONFIG_HASH,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "rival2" / "campaign01"
CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "rival2"
    / "campaign01"
    / "rival2_campaign01_100m_resume.pt"
)
LABELS = ("000m", "010m", "025m", "050m", "100m")


def _json(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def test_campaign01_published_execution_and_behavior_are_independent() -> None:
    summary = _json("summary.json")
    assert summary["execution_status"] == "COMPLETE"
    assert summary["behavioral_result"] == "DEGRADED"
    assert summary["selected_worlds"] == 131072
    assert summary["final_iteration"] == 12
    assert summary["final_agent_decision_samples"] == 100663296
    assert summary["checkpoint_reload_verdict"] == "PASS_GREEN"
    assert summary["update_integrity_pass_count"] == 12
    assert summary["update_integrity_failure_count"] == 0
    assert summary["fixed_evaluation_pass_count"] == 5
    assert summary["prior_results_v01_through_v05_unchanged"]
    assert summary["v05_pass_green_unchanged"]
    assert not summary["v06_begun"]


def test_campaign01_published_authority_and_evaluations_are_fixed() -> None:
    configuration = _json("config.json")
    assert configuration["contract_hashes"] == EXPECTED_CONTRACT_HASHES
    assert configuration["policy_config_hash"] == EXPECTED_POLICY_CONFIG_HASH
    assert configuration["ppo_config"]["entropy_coefficient"] == 0.01
    assert configuration["ppo_config"]["rollout_horizon"] == 32
    protocol = configuration["evaluation"]["protocol_sha256"]
    samples = [0, 16777216, 25165824, 50331648, 100663296]
    for label, expected_samples in zip(LABELS, samples, strict=True):
        evaluation = _json(f"evaluation_{label}.json")
        assert evaluation["verdict"] == "PASS_GREEN"
        assert evaluation["agent_decision_samples"] == expected_samples
        assert evaluation["evaluation_protocol_sha256"] == protocol
        assert all(mode["verdict"] == "PASS_GREEN" for mode in evaluation["modes"].values())


def test_campaign01_final_checkpoint_custody_and_format() -> None:
    summary = _json("summary.json")
    assert CHECKPOINT.stat().st_size == 21126324
    assert CHECKPOINT.stat().st_size <= 25 * 1024**2
    assert _sha256(CHECKPOINT) == summary["published_checkpoint_sha256"]
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    assert payload["format"] == "RIVAL2_CHECKPOINT_V1"
    assert payload["contract_hashes"] == EXPECTED_CONTRACT_HASHES
    assert payload["policy_config_hash"] == EXPECTED_POLICY_CONFIG_HASH
    assert payload["policy_version"] == 12
    assert payload["iteration"] == 12
    assert payload["total_agent_samples"] == 100663296
    assert [item["version"] for item in payload["historical_opponents"]] == [0, 2, 3, 6, 12]
