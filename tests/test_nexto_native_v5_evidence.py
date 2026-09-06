"""Checks of completed no-learning validation, separate from gameplay promotion."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/rival2/nexto_native_controller_v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def test_gpu_protocol_and_sources_remain_frozen():
    protocol = json.loads((OUT / "gpu_protocol.json").read_text())
    started = json.loads((OUT / "gpu_run_started.json").read_text())
    result = json.loads((OUT / "gpu_results.json").read_text())
    assert started["commit"] == "bbafdd99fcac865b879822e3c11e9ba459537a1b"
    assert started["protocol_sha256"] == result["protocol_sha256"] == sha(OUT / "gpu_protocol.json")
    assert sha(ROOT / protocol["checkpoint"]) == protocol["checkpoint_sha256"]
    for path, expected in protocol["sources"].items(): assert sha(ROOT / path) == expected, path
    assert result["optimizer_steps"] == result["backward_calls"] == 0
    assert result["checkpoint_unchanged"]


def test_gpu_controller_and_real_model_replay_completed():
    result = json.loads((OUT / "gpu_results.json").read_text())
    assert result["cuda_oracle_world_ticks"] == 4096
    assert set(result["pinned_model_checkpoint_replay"]) == {"native_v5", "deterministic_argmax"}
    for arm in result["pinned_model_checkpoint_replay"].values():
        assert arm["exact_controls_indices_state_and_rng"] and arm["physics_calls"] == 17


def test_full_scale_rollout_masks_and_immutable_models():
    row = json.loads((OUT / "gpu_results.json").read_text())["full_scale_collector"]
    assert (row["worlds"], row["decisions"]) == (32768, 90)
    metrics = row["metrics"]
    assert metrics["trainable_agent_samples"] == 4423680
    assert metrics["nexto_training_sample_count"] * 3 == metrics["trainable_agent_samples"]
    assert metrics["physical_physics_ticks"] == 11796480
    assert metrics["physical_goals"] > 0 and metrics["resets"] >= metrics["physical_goals"]
    for key in ("actor_unchanged", "nexto_unchanged", "no_gradients", "inactive_world_controls_zero",
                "serialized_controller_and_rng_exact"):
        assert row[key]
    assert row["peak_allocated_bytes"] < row["peak_reserved_bytes"] < 32 * 1024**3


def test_bounded_match_smoke_is_not_a_completed_match_or_capability_claim():
    result = json.loads((OUT / "gpu_results.json").read_text())["match_smoke"]
    data = np.load(OUT / "gpu_match_smoke.npz")
    assert data["actions"].shape == (2400, 10, 2, 8)
    assert data["poses"].shape == (600, 10, 9)
    assert np.isfinite(data["actions"]).all() and np.isfinite(data["poses"]).all()
    assert result["trace_sha256"] == sha(OUT / "gpu_match_smoke.npz")
    assert result["mode"] == "deterministic_argmax" and result["summary"]["unresolved"] == 10
    assert (result["summary"]["goals_for"], result["summary"]["goals_against"]) == (2, 10)
    assert result["summary"]["touches"] == 23
