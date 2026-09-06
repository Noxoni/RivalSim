import json

import torch

from benchmarks import audit_direct_skills_native_nexto_entry as audit


def test_real_entry_and_first_artifact_bindings():
    result = json.loads((audit.OUT / "first_update_audit.json").read_text())
    assert len(result["checks"]) == 21 and all(result["checks"].values())
    assert result["optimizer_steps_in_audit"] == 0 and result["audit_device"] == "cpu"
    for name in ("parent", "entry", "first"):
        item = result[name]
        assert audit.sha(audit.ROOT / item["path"]) == item["sha256"]
    spec = json.loads((audit.OUT / "training_authority.json").read_text())
    assert audit.canonical(spec) == result["authority_sha256"]
    first = torch.load(audit.ROOT / result["first"]["path"], map_location="cpu", weights_only=False)
    assert audit.tensor_hash(first["model"]) == result["first_model_sha256"]
    assert first["native_nexto_branch_updates"] == 1 and first["accepted_updates"] == 651
    assert first["opponent_state"]["native_nexto"]["sampling_mode"] == "native_v5"


def test_first_update_was_real_ppo_with_correct_sample_mask():
    row = json.loads((audit.OUT / "first_update_audit.json").read_text())["first_update"]
    assert row["branch_updates"] == 1
    assert row["ppo"]["optimizer_steps"] > 0 and row["ppo"]["kl_rejections"] == 0
    assert row["training"]["trainable_agent_samples"] == 4423680
    assert row["training"]["nexto_training_sample_count"] == 1474560
    assert row["training"]["physical_goals"] > 0


def test_baseline_selection_preserves_both_negative_results():
    result = json.loads((audit.OUT / "selection.json").read_text())
    assert result["selected_name"] == "finishing650"
    for name in ("reference600", "finishing650"):
        path = audit.OUT / (name + ".json")
        assert audit.sha(path) == result["results"][name]["file_sha256"]
        record = json.loads(path.read_text())
        assert record["summary"]["wins"] == 0 and record["summary"]["losses"] == 10
        assert record["optimizer_steps"] == 0 and record["model_unchanged"]
