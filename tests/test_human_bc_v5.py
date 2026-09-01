from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import benchmarks.run_rival2_human_bc_v5 as runner
from benchmarks.run_rival2_human_bc_continuation_v1 import (
    _configure_trainable_parameters,
)
from rivalsim.rival2_policy import Rival2ActorCritic, Rival2PolicyConfig

ROOT = Path(__file__).resolve().parents[1]
GROUPS = (
    "all_perspectives",
    "current_policy_applicable",
    "counterfactual_opponent",
    "historical_opponent",
    "low_teacher_variance",
    "orientation_sensitive",
)


def _distribution_metrics(boost: dict[str, float]) -> dict[str, object]:
    return {
        name: {
            "actor_distribution": {
                "finite": True,
                "button_probability": {
                    "boost": {"saturation_fraction": boost[name]}
                },
            }
        }
        for name in GROUPS
    }


def _distribution_config(parent: dict[str, float]) -> dict[str, object]:
    return {
        "validation": {"groups": list(GROUPS)},
        "parent_relative_boost_saturation": {
            "absolute_baseline_limit": 0.95,
            "worsening_allowance": 0.005,
            "complete_validation": {
                name: {
                    "bc_v1_parent_saturation_fraction": value,
                    "candidate_limit": max(0.95, value + 0.005),
                }
                for name, value in parent.items()
            },
            "stress_validation": {
                name: {
                    "bc_v1_parent_saturation_fraction": value,
                    "candidate_limit": max(0.95, value + 0.005),
                }
                for name, value in parent.items()
            },
        },
    }


def test_parent_relative_boost_saturation_exact_formula(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = {name: 0.97 for name in GROUPS}
    parent["historical_opponent"] = 0.90
    candidate = {name: 0.975 for name in GROUPS}
    candidate["historical_opponent"] = 0.95
    raw_checks = {"finite": True}
    raw_checks.update(
        {f"{name}.button.boost.not_saturated": False for name in GROUPS}
    )
    monkeypatch.setattr(
        runner.continuation,
        "_distribution_guard",
        lambda _wrapped, _config: {"checks": dict(raw_checks), "accepted": False},
    )

    result = runner._simulator_distribution_guard(
        _distribution_metrics(candidate),
        _distribution_metrics(parent),
        _distribution_config(parent),
        authority_phase="complete_validation",
    )

    assert result["accepted"]
    assert result["parent_relative_boost_saturation"]["all_perspectives"][
        "candidate_limit"
    ] == pytest.approx(0.975)
    assert result["parent_relative_boost_saturation"]["historical_opponent"][
        "candidate_limit"
    ] == pytest.approx(0.95)


def test_parent_relative_boost_rejects_worsening_and_preserves_other_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = {name: 0.97 for name in GROUPS}
    candidate = {name: 0.975 for name in GROUPS}
    candidate["orientation_sensitive"] = 0.975001
    raw_checks = {
        "finite": True,
        "all_perspectives.analog.roll.nonconstant": False,
    }
    raw_checks.update(
        {f"{name}.button.boost.not_saturated": False for name in GROUPS}
    )
    monkeypatch.setattr(
        runner.continuation,
        "_distribution_guard",
        lambda _wrapped, _config: {"checks": dict(raw_checks), "accepted": False},
    )

    result = runner._simulator_distribution_guard(
        _distribution_metrics(candidate),
        _distribution_metrics(parent),
        _distribution_config(parent),
        authority_phase="stress_validation",
    )

    assert not result["accepted"]
    assert not result["checks"][
        "orientation_sensitive.button.boost.not_saturated"
    ]
    assert not result["checks"]["all_perspectives.analog.roll.nonconstant"]


def test_v5_validation_has_no_extra_one_or_orientation_margin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner,
        "detailed_retention_guard",
        lambda _metrics, _guard: {"accepted": True, "checks": {"hard": True}},
    )

    result = runner._v5_validation_guard({}, {"actor_max_sample_kl": 2.0})

    assert result["accepted"]
    assert result["extra_max_sample_selection_margin_removed"]
    assert result["extra_orientation_channel_selection_margin_removed"]


def test_actor_only_optimizer_membership() -> None:
    model = Rival2ActorCritic(Rival2PolicyConfig(hidden_dim=16, hidden_layers=1))
    config = {"trainable_parameters": {"mode": "actor_head_only"}}

    parameters, names = _configure_trainable_parameters(model, config)
    optimizer = torch.optim.AdamW(parameters, lr=5e-5)

    assert names == ("actor.weight", "actor.bias")
    assert sum(len(group["params"]) for group in optimizer.param_groups) == 2
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name not in names
    )


def test_frozen_v5_authority_exact_and_reuses_sealed_v4_test() -> None:
    config = json.loads(
        (ROOT / "results/rival2/human_bc_v5/frozen_config.json").read_text(
            encoding="utf-8"
        )
    )
    reuse = json.loads(
        (
            ROOT
            / "results/rival2/human_bc_v5/v4_authority_reuse_manifest.json"
        ).read_text(encoding="utf-8")
    )

    assert config["simulator_authority"]["v4_assets_reused_exactly"]
    assert not config["simulator_authority"]["new_v5_corpora_generated"]
    assert reuse["new_simulator_corpora_generated"] == 0
    assert reuse["untouched_test_sealed_checks"][
        "v4_simulator_test_student_evaluations_zero"
    ]
    assert config["validation"]["hard_guard"]["actor_max_sample_kl"] == 2.0
    assert "selection_margin" not in config["validation"]
    assert config["tail_aware_retention"]["ordinary_mean_kl_coefficient"] == 0.5
    assert config["tail_aware_retention"]["total_mean_barrier_coefficient"] == 1.5
    assert config["tail_aware_retention"]["total_cvar_barrier_coefficient"] == 2.0
    assert (
        config["tail_aware_retention"]["orientation_cvar_barrier_coefficient"]
        == 2.0
    )
    assert config["optimizer"]["exact_retry_learning_rates"] == [
        5e-5,
        2.5e-5,
        1.25e-5,
    ]
    assert config["training"]["validation_interval_optimizer_steps"] == 128
    assert config["training"]["maximum_accepted_supervised_steps"] == 10_000
    assert config["selection"]["minimum_accepted_steps_before_plateau"] == 3072
    assert config["selection"]["early_stopping_patience_validations"] == 16
    assert sum(config["selection"]["score_weights"].values()) == pytest.approx(1.0)


def test_v4_test_runtime_artifacts_remain_absent() -> None:
    assert not (ROOT / "results/rival2/human_bc_v4/test_access_ledger.json").exists()
    assert not (ROOT / "results/rival2/human_bc_v4/untouched_test_results.json").exists()
    assert not (ROOT / "results/rival2/human_bc_v4/human_test_metrics.json").exists()
