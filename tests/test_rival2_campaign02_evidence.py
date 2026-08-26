import hashlib
import json
from pathlib import Path

import torch

from benchmarks.run_rival2_campaign01 import EXPECTED_CONTRACT_HASHES
from benchmarks.run_rival2_campaign02 import (
    EXPECTED_INITIALIZATION_SHA256,
    ppo_configuration_differences,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "rival2" / "campaign02"
CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "rival2"
    / "campaign02"
    / "rival2_campaign02_100m_resume.pt"
)
LABELS = ("000m", "010m", "025m", "050m", "100m")


def _json(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def test_campaign02_published_control_and_boundary() -> None:
    summary = _json("summary.json")
    assert summary["execution_status"] == "COMPLETE"
    assert summary["behavioral_result"] == "IMPROVED"
    assert summary["selected_worlds"] == 131072
    assert summary["final_iteration"] == 12
    assert summary["final_agent_decision_samples"] == 100663296
    assert summary["initialization_model_sha256"] == EXPECTED_INITIALIZATION_SHA256
    assert summary["initialization_control_verdict"] == "PASS_GREEN"
    assert summary["initialization_evaluation_exact"]
    assert summary["checkpoint_reload_verdict"] == "PASS_GREEN"
    assert summary["entropy_coefficient"] == 0.0
    assert summary["entropy_optimization_contribution"] == 0.0
    assert summary["update_integrity_pass_count"] == 12
    assert summary["fixed_evaluation_pass_count"] == 5
    assert summary["prior_v05_and_campaign01_artifacts_unchanged"]
    assert summary["v05_pass_green_unchanged"]
    assert not summary["v06_begun"]


def test_campaign02_configuration_diff_and_initialization_are_exact() -> None:
    configuration = _json("config.json")
    differences = ppo_configuration_differences(
        configuration["campaign01_ppo_config"], configuration["ppo_config"]
    )
    assert differences == {
        "entropy_coefficient": {"campaign01": 0.01, "campaign02": 0.0}
    }
    assert configuration["contract_hashes"] == EXPECTED_CONTRACT_HASHES
    control = _json("initialization_control.json")
    assert control["actual_model_sha256"] == EXPECTED_INITIALIZATION_SHA256
    assert control["model_sha256_exact"]
    assert control["evaluation_control"]["semantic_metrics_exact"]
    assert control["evaluation_control"]["semantic_difference_count"] == 0


def test_campaign02_evaluations_and_prospective_classification() -> None:
    configuration = _json("config.json")
    protocol = configuration["evaluation"]["protocol_sha256"]
    samples = [0, 16777216, 25165824, 50331648, 100663296]
    for label, expected_samples in zip(LABELS, samples, strict=True):
        evaluation = _json(f"evaluation_{label}.json")
        assert evaluation["verdict"] == "PASS_GREEN"
        assert evaluation["agent_decision_samples"] == expected_samples
        assert evaluation["evaluation_protocol_sha256"] == protocol
        assert all(mode["verdict"] == "PASS_GREEN" for mode in evaluation["modes"].values())
    comparison = _json("comparison_campaign01.json")
    classification = comparison["behavioral_classification"]
    assert classification["behavioral_result"] == "IMPROVED"
    assert classification["improved_vs_initialization_count"] == 2
    assert classification["worse_vs_initialization_count"] == 1
    assert classification["not_worse_than_campaign01_on_all_three"]


def test_campaign02_optimizer_diagnosis_and_checkpoint_custody() -> None:
    optimizer = _json("optimizer_diagnosis.json")
    assert optimizer["entropy_coefficient"] == 0.0
    assert optimizer["entropy_optimization_contribution"] == 0.0
    assert optimizer["maximum_approximate_kl"] == {
        "value": 0.00819435529410839,
        "iteration": 2,
    }
    assert optimizer["maximum_clip_fraction"] == {
        "value": 0.0875341072678566,
        "iteration": 6,
    }
    assert optimizer["flagged_updates"] == []
    assert not optimizer["campaign01_update4_style_instability_recurred"]
    assert not optimizer["analog_standard_deviation"]["trended_toward_ceiling"]
    summary = _json("summary.json")
    assert CHECKPOINT.stat().st_size == 21126324
    assert CHECKPOINT.stat().st_size <= 25 * 1024**2
    assert _sha256(CHECKPOINT) == summary["published_checkpoint_sha256"]
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    assert payload["format"] == "RIVAL2_CHECKPOINT_V1"
    assert payload["contract_hashes"] == EXPECTED_CONTRACT_HASHES
    assert payload["ppo_config"]["entropy_coefficient"] == 0.0
    assert payload["policy_version"] == payload["iteration"] == 12
    assert payload["total_agent_samples"] == 100663296
    assert [item["version"] for item in payload["historical_opponents"]] == [0, 2, 3, 6, 12]
