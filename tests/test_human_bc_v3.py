from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from benchmarks.run_rival2_human_bc_continuation_v1 import (
    _configure_trainable_parameters,
    _model_partition_hashes,
)
from benchmarks.run_rival2_human_bc_v3 import FROZEN_CONFIG_SHA256
from rivalsim.human_demo.bc_v3_retention import (
    build_retention_pools,
    detailed_retention_guard,
    encoded_rows_for_worlds,
    evaluate_detailed_retention,
    gather_encoded_rows,
    int64_sha256,
    role_masks_for_encoded,
    sample_retention_rows,
    tail_aware_actor_retention_loss,
    verify_retention_pools,
)
from rivalsim.human_demo.missing_feature_distillation import file_sha256
from rivalsim.rival2_contracts import OBS_DIM
from rivalsim.rival2_policy import Rival2ActorCritic, Rival2PolicyConfig

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "results/rival2/human_bc_v3/frozen_config.json"
STRATA_PATH = ROOT / "results/rival2/human_bc_v3/retention_strata_manifest.json"
TEST_AUTHORITY_PATH = ROOT / "results/rival2/human_bc_v3/new_simulator_test_authority.json"


def _policy_config() -> Rival2PolicyConfig:
    return Rival2PolicyConfig(hidden_dim=16, hidden_layers=1)


def _model() -> Rival2ActorCritic:
    torch.manual_seed(11)
    return Rival2ActorCritic(_policy_config())


def _observations(worlds: int) -> torch.Tensor:
    values = torch.arange(128 * worlds * 2 * OBS_DIM, dtype=torch.float32)
    return (values.remainder(101) / 100.0).reshape(128, worlds, 2, OBS_DIM)


def test_v3_authority_is_prospective_and_keeps_v1_hard_guard() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    test_authority = json.loads(TEST_AUTHORITY_PATH.read_text(encoding="utf-8"))
    assert file_sha256(CONFIG_PATH) == FROZEN_CONFIG_SHA256
    assert config["authority"]["required_parent"] == ("b9140d96f73a78a539df7ebd019a8f9670bc34e7")
    assert config["authority"]["source_checkpoint_sha256"] == (
        "560C2414C17039DC920126EA148BF73FE6CC4677EE440F043599A7E1C76D2874"
    )
    assert config["authority"]["v2_checkpoint_training_prohibited"]
    assert config["validation"]["hard_guard"]["actor_max_sample_kl"] == 2.0
    assert config["simulator_authority"]["complete_validation_worlds"] == 3277
    assert config["simulator_authority"]["new_untouched_test_worlds"] == 3277
    assert config["test_discipline"]["selection_reopened_after_test"] is False
    assert test_authority["optimizer_steps_before_binding"] == 0
    assert test_authority["student_evaluations_before_final_selection"] == 0
    assert (
        test_authority["disjointness"]["new_corpus_seed"]
        != (test_authority["disjointness"]["old_corpus_seed"])
    )
    assert test_authority["disjointness"]["seed_namespace_distinct"]


def test_actor_only_boundary_keeps_trunk_and_critic_byte_identical() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    model = _model()
    before = _model_partition_hashes(model)
    parameters, names = _configure_trainable_parameters(model, config)
    assert names == ("actor.weight", "actor.bias")
    assert [name for name, value in model.named_parameters() if value.requires_grad] == [
        "actor.weight",
        "actor.bias",
    ]
    optimizer = torch.optim.AdamW(parameters, lr=3e-5)
    assert optimizer.state_dict()["state"] == {}
    optimizer.zero_grad(set_to_none=True)
    model(torch.zeros(4, OBS_DIM))[0].square().mean().backward()
    optimizer.step()
    after = _model_partition_hashes(model)
    assert after["frozen_trunk_and_critic"] == before["frozen_trunk_and_critic"]
    assert after["actor"] != before["actor"]


def test_encoded_rows_round_trip_and_role_masks() -> None:
    observations = _observations(3)
    encoded = encoded_rows_for_worlds(np.asarray([2, 0], dtype=np.int64))
    gathered = gather_encoded_rows(observations, encoded)
    expected = torch.cat(
        [observations[:, 2].reshape(-1, OBS_DIM), observations[:, 0].reshape(-1, OBS_DIM)]
    )
    assert torch.equal(gathered, expected)
    assert int64_sha256(encoded) == int64_sha256(encoded.clone())
    family = torch.tensor([0, 1, 1])
    rival_side = torch.tensor([0, 1, 0])
    selected = torch.tensor(
        [0 * 256 + 0, 1 * 256 + 0, 1 * 256 + 1, 2 * 256 + 1],
        dtype=torch.int64,
    )
    roles = role_masks_for_encoded(selected, family, rival_side)
    assert roles["current_policy_applicable"].tolist() == [True, False, True, False]
    assert roles["historical_opponent"].tolist() == [False, True, False, True]
    assert roles["counterfactual_opponent"].tolist() == [False, True, False, True]


def test_teacher_only_strata_and_sampling_are_deterministic() -> None:
    teacher = _model().eval().requires_grad_(False)
    observations = _observations(3)
    family = torch.tensor([0, 1, 1])
    rival_side = torch.tensor([0, 1, 0])
    pools = build_retention_pools(
        teacher,
        observations,
        np.asarray([0, 1, 2]),
        family,
        rival_side,
        low_variance_quantile=0.1,
        policy_config=_policy_config(),
        rows_per_batch=97,
    )
    assert pools.natural.numel() == 3 * 256
    assert pools.current_policy_applicable.numel() == 2 * 256
    assert pools.historical_opponent.numel() == 2 * 128
    assert 0 < pools.low_teacher_variance.numel() < pools.natural.numel()
    assert pools.manifest["membership_source"] == "teacher_only_no_student_outputs"
    verify_retention_pools(pools, pools.manifest)
    modified = copy.deepcopy(pools.manifest)
    modified["pools"]["historical_opponent"]["rows"] += 1
    with pytest.raises(RuntimeError, match="retention strata changed"):
        verify_retention_pools(pools, modified)

    counts = {
        "natural": 7,
        "current_policy_applicable": 5,
        "historical_opponent": 3,
        "low_teacher_variance": 2,
    }
    first, realized = sample_retention_rows(
        pools, counts, generator=torch.Generator().manual_seed(19)
    )
    second, _ = sample_retention_rows(pools, counts, generator=torch.Generator().manual_seed(19))
    assert torch.equal(first, second)
    assert realized == counts
    assert first.numel() == sum(counts.values())


def test_tail_barrier_is_smooth_and_penalizes_rare_large_kl() -> None:
    config = _policy_config()
    teacher = torch.zeros(16, config.actor_outputs)
    safe = teacher.clone().requires_grad_(True)
    tail = teacher.clone()
    tail[-1, 0] = 2.0
    tail = tail.requires_grad_(True)
    kwargs = {
        "policy_config": config,
        "mean_kl_coefficient": 2.0,
        "barrier_threshold": 0.5,
        "barrier_temperature": 0.05,
        "barrier_coefficient": 4.0,
    }
    safe_loss = tail_aware_actor_retention_loss(teacher, safe, **kwargs)
    tail_loss = tail_aware_actor_retention_loss(teacher, tail, **kwargs)
    assert tail_loss.maximum_sample_kl > 0.5
    assert tail_loss.barrier > safe_loss.barrier
    assert tail_loss.loss > safe_loss.loss
    tail_loss.loss.backward()
    assert torch.isfinite(tail.grad).all()
    assert float(tail.grad[-1, 0].abs()) > 0.0


def test_complete_retention_metrics_cover_all_required_perspectives() -> None:
    config = _policy_config()
    teacher = _model().eval().requires_grad_(False)
    student = copy.deepcopy(teacher).eval().requires_grad_(False)
    with torch.no_grad():
        student.actor.bias[0] += 0.02
    observations = _observations(3)
    family = torch.tensor([0, 1, 1])
    rival_side = torch.tensor([0, 1, 0])
    metrics = evaluate_detailed_retention(
        teacher,
        student,
        observations,
        np.asarray([0, 1, 2]),
        family,
        rival_side,
        low_variance_threshold_log_std=100.0,
        policy_config=config,
        worlds_per_batch=2,
    )
    assert metrics["all_perspectives"]["sample_count"] == 3 * 256
    assert metrics["current_policy_applicable"]["sample_count"] == 2 * 256
    assert metrics["counterfactual_opponent"]["sample_count"] == 2 * 128
    assert metrics["historical_opponent"]["sample_count"] == 2 * 128
    assert metrics["low_teacher_variance"]["sample_count"] == 3 * 256
    assert metrics["critic"]["max_absolute_drift"] == 0.0
    guard = detailed_retention_guard(
        metrics,
        {
            "actor_mean_kl": 0.02,
            "actor_max_sample_kl": 2.0,
            "actor_max_channel_kl": 0.01,
            "critic_rmse": 0.075,
            "critic_max_absolute_drift": 0.5,
        },
    )
    assert guard["accepted"]


def test_committed_strata_bind_teacher_only_membership() -> None:
    manifest = json.loads(STRATA_PATH.read_text(encoding="utf-8"))
    assert manifest["membership_source"] == "teacher_only_no_student_outputs"
    assert manifest["low_variance_definition"]["teacher"].startswith("frozen Human BC V1")
    assert manifest["pools"]["historical_opponent"]["rows"] > 0
    assert manifest["pools"]["low_teacher_variance"]["rows"] > 0
