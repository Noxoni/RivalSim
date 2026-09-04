from __future__ import annotations

import copy
import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from benchmarks import run_rival2_ssl_foundation_v5_restart_v1 as campaign
from rivalsim.rival2_unified_policy import Rival2UnifiedActorCritic, Rival2UnifiedPolicyConfig


@pytest.fixture(scope="module")
def source():
    prior = torch.get_num_threads()
    torch.set_num_threads(2)
    payload = campaign.engine.source_payload()
    yield payload
    torch.set_num_threads(prior)


def test_frozen_v5_actor_value_hidden_parity_and_fresh_optimizer(source):
    original = Rival2UnifiedActorCritic(Rival2UnifiedPolicyConfig(**source["policy_config"]))
    original.load_state_dict(source["model"], strict=True)
    upgraded = campaign.build_root_model(source)
    generator = torch.Generator().manual_seed(9027)
    obs = torch.randn(4, 7, 182, generator=generator)
    hidden = torch.randn(1, 4, 256, generator=generator)
    reset = torch.zeros(4, 7, dtype=torch.bool)
    reset[1, 3] = True
    with torch.no_grad():
        for a, b in zip(
            original(obs, hidden, reset_before=reset),
            upgraded(obs, hidden, reset_before=reset),
            strict=True,
        ):
            assert torch.equal(a, b)
    for key, value in source["model"].items():
        if not key.startswith("critic."):
            assert torch.equal(value, upgraded.state_dict()[key])
    trainer = SimpleNamespace(model=upgraded)
    policy_lr, critic_lr = (
        campaign.engine.POLICY_LEARNING_RATE,
        campaign.engine.CRITIC_LEARNING_RATE,
    )
    try:
        campaign.engine.POLICY_LEARNING_RATE = 1e-4
        campaign.engine.CRITIC_LEARNING_RATE = 3e-4
        campaign.engine.configure_optimizer(trainer)
    finally:
        campaign.engine.POLICY_LEARNING_RATE, campaign.engine.CRITIC_LEARNING_RATE = (
            policy_lr,
            critic_lr,
        )
    assert not trainer.optimizer.state
    assert {g["name"]: g["lr"] for g in trainer.optimizer.param_groups} == {
        "policy": 1e-4,
        "critic": 3e-4,
    }
    assert len({id(p) for p in upgraded.parameters()}) == sum(
        len(g["params"]) for g in trainer.optimizer.param_groups
    )


def test_ppo_descendant_and_wrong_contract_rejected(source):
    wrong = copy.copy(source)
    wrong["format"] = "RIVAL2_SSL_FOUNDATION_PPO_V2_CHECKPOINT"
    with pytest.raises(ValueError, match="never a PPO descendant"):
        campaign.build_root_model(wrong)
    wrong = copy.copy(source)
    wrong["policy_config_sha256"] = "wrong"
    with pytest.raises(ValueError, match="config hash"):
        campaign.build_root_model(wrong)
    with pytest.raises(ValueError, match="bounded V5 restart"):
        campaign.validate_resume_payload(source)


def test_authority_preserves_settings_and_freezes_comparison():
    prior = json.loads(campaign.amended.AUTHORITY.read_text())
    payload = campaign.authority_payload("A" * 40, "2026-09-04T00:00:00Z")
    for key in ("ppo", "reward", "reset_curriculum", "exploration", "opponents", "source"):
        assert payload[key] == prior[key]
    assert "amendment" not in payload
    assert payload["campaign"]["maximum_accepted_updates"] == 100
    assert payload["campaign"]["continuation_review_marker"] == 100
    assert payload["campaign"]["evaluation_ticks"] == 3600
    assert payload["restart"]["initial_resume_checkpoint"] is None
    assert payload["restart"]["initial_accepted_updates"] == 0
    assert payload["restart"]["exploration_schedule_offset"] == 600


def test_exploration_matches_mature_amended_values():
    for update in (0, 1, 30, 50, 100):
        actual = campaign.restart_exploration(update)
        assert actual.analog_sigma == 0.04
        assert actual.button_temperature == 0.25
        assert actual.accepted_update == update + 600
        assert actual.normalized_progress == 1.0
    with pytest.raises(ValueError):
        campaign.restart_exploration(-1)


def test_bounded_launch_defaults():
    args = campaign.parser().parse_args([])
    assert args.resume is None
    assert not args.continue_after_600
    assert args.run_dir == str(campaign.RUN_DIR)
    assert campaign.REVIEW_UPDATES == 100


def test_resume_rejects_wrong_authority_parent_settings_and_bound(monkeypatch, source):
    monkeypatch.setattr(campaign.engine, "sha256_file", lambda _path: "A" * 64)
    payload = {
        "format": campaign.FORMAT + "_CHECKPOINT",
        "lineage": campaign.LINEAGE,
        "source": {"sha256": campaign.engine.SOURCE_SHA256, "authority_sha256": "A" * 64},
        "phase_transition": {
            "restart": {"version": campaign.FORMAT, "optimizer_state_loaded": False}
        },
        "ppo_config": asdict(campaign.amended.new_ppo_config()),
        "ppo_config_sha256": campaign.amended.new_ppo_config().content_hash,
        "policy_config": asdict(campaign.build_root_model(source).config),
        "opponents": {"config": asdict(campaign.amended.OPPONENT_CONFIG)},
        "accepted_updates_total": 50,
    }
    campaign.validate_resume_payload(payload)
    for key, value in (
        ("accepted_updates_total", 101),
        ("lineage", "old SSL lineage"),
        ("source", {"sha256": "wrong", "authority_sha256": "A" * 64}),
        ("source", {"sha256": campaign.engine.SOURCE_SHA256, "authority_sha256": "B" * 64}),
        ("ppo_config_sha256", "wrong"),
    ):
        wrong = {**payload, key: value}
        with pytest.raises(ValueError, match="bounded V5 restart"):
            campaign.validate_resume_payload(wrong)


def test_runtime_is_bounded_and_unlimited_flag_is_rejected(monkeypatch):
    # All shared engine mutations are restored so adjacent historical tests
    # remain isolated even though production entry points reuse the engine.
    for name, value in list(vars(campaign.engine).items()):
        if not name.startswith("__"):
            monkeypatch.setattr(campaign.engine, name, value)
    campaign.configure_engine()
    assert campaign.engine.CONTINUATION_REVIEW_MARKER == 100
    assert campaign.engine.MAXIMUM_ACCEPTED_UPDATES == 100
    assert campaign.engine.ACTIVE_SNAPSHOT_INTERVAL == 50
    assert campaign.engine.CHECKPOINT_FORMAT == campaign.FORMAT + "_CHECKPOINT"
    assert campaign.engine.PPO_EPOCHS == 2
    args = campaign.parser().parse_args(["--continue-after-600"])
    with pytest.raises(ValueError, match="cannot continue beyond 100"):
        campaign.run(args)
