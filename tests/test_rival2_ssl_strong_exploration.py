"""Focused tests for the user-authorized stronger exploration override."""

import copy
import json
from dataclasses import asdict

import pytest
import torch

from benchmarks import run_rival2_ssl_foundation_strong_exploration as strong
from rivalsim.rival2_policy import (
    deterministic_hybrid_action,
    hybrid_distribution_parameters,
    hybrid_entropy,
    hybrid_log_probability,
    sample_hybrid_action,
)


def test_publication_audit_allows_episode_resets_but_not_curriculum_changes():
    from benchmarks.report_rival2_ssl_strong_exploration import curriculum_contract_preserved

    parent = {
        "reset_curriculum": {
            "scenario_family": torch.tensor([0, 1, 2]),
            "summary": {"counts": {"a": 1, "b": 1, "c": 1}},
            "scenario_id_in_observation": False,
        }
    }
    current = copy.deepcopy(parent)
    current["reset_curriculum"]["scenario_family"] = torch.tensor([2, 0, 1])
    assert curriculum_contract_preserved(parent, current)
    current["reset_curriculum"]["scenario_family"][0] = 3
    assert not curriculum_contract_preserved(parent, current)
    current = copy.deepcopy(parent)
    current["reset_curriculum"]["scenario_id_in_observation"] = True
    assert not curriculum_contract_preserved(parent, current)


@pytest.mark.parametrize("update", [0, 84, 85, 99, 100])
def test_fixed_stronger_distribution_without_lr_or_cadence_changes(update):
    exploration = strong.exploration_for_update(update)
    assert exploration.analog_sigma == 0.30
    assert exploration.button_temperature == 1.0
    assert exploration.contract_sha256 == strong.CONTRACT_HASH
    assert exploration.accepted_update == update
    assert strong.c.new_ppo_config().rollout_horizon == 360
    assert strong.c.new_ppo_config().epochs == 2
    assert strong.c.new_ppo_config().learning_rate == 1e-4


def test_negative_counter_rejected():
    with pytest.raises(ValueError):
        strong.exploration_for_update(-1)


def test_sampling_log_probability_ratio_and_finite_gradients():
    torch.manual_seed(39251)
    actor = torch.randn(512, 13, dtype=torch.float64, requires_grad=True)
    override = strong.exploration_for_update(84).distribution_override
    sample = sample_hybrid_action(
        actor, generator=torch.Generator().manual_seed(6), distribution_override=override
    )
    recomputed = hybrid_log_probability(
        actor, sample.action, pre_tanh=sample.pre_tanh, distribution_override=override
    )
    torch.testing.assert_close(recomputed, sample.log_probability, rtol=0, atol=0)
    torch.testing.assert_close(
        (recomputed - sample.log_probability).exp(), torch.ones_like(recomputed), rtol=0, atol=0
    )
    assert sample.action[:, :5].abs().max() <= 1
    assert torch.all((sample.action[:, 5:] == 0) | (sample.action[:, 5:] == 1))
    # Stored rollout actions/logp must be treated as constants during PPO.
    logp = hybrid_log_probability(
        actor,
        sample.action.detach(),
        pre_tanh=sample.pre_tanh.detach(),
        distribution_override=override,
    )
    loss = -(logp - sample.log_probability.detach()).exp().mean()
    loss.backward()
    assert torch.isfinite(actor.grad).all()
    assert torch.count_nonzero(actor.grad[:, 5:10]) == 0  # fixed sigma bypasses raw log-std
    assert torch.isfinite(hybrid_entropy(actor, distribution_override=override)).all()


def test_noise_is_7_5_times_wider_and_buttons_less_committed_not_forced():
    actor = torch.zeros(20000, 13, dtype=torch.float64)
    actor[:, :5] = torch.tensor([0.4, -0.3, 0.2, -0.1, 0.0])
    actor[:, 10:] = torch.tensor([-1.0, 0.5, 2.0])
    original = actor.clone()
    old = strong.c.prior.restart_exploration(84).distribution_override
    new = strong.exploration_for_update(84).distribution_override
    a, b = [
        sample_hybrid_action(
            actor, generator=torch.Generator().manual_seed(8), distribution_override=value
        )
        for value in (old, new)
    ]
    torch.testing.assert_close(
        b.pre_tanh - actor[:, :5], (a.pre_tanh - actor[:, :5]) * 7.5, rtol=1e-12, atol=1e-12
    )
    assert (b.action[:, :5] - torch.tanh(actor[:, :5])).square().mean() > 30 * (
        a.action[:, :5] - torch.tanh(actor[:, :5])
    ).square().mean()
    _, _, old_logits = hybrid_distribution_parameters(actor, distribution_override=old)
    _, _, new_logits = hybrid_distribution_parameters(actor, distribution_override=new)
    assert torch.all((new_logits.sigmoid() - 0.5).abs() < (old_logits.sigmoid() - 0.5).abs())
    assert torch.equal(old_logits > 0, new_logits > 0)
    assert torch.equal(actor, original)
    assert torch.equal(deterministic_hybrid_action(actor), deterministic_hybrid_action(original))


def test_authority_changes_only_exploration_and_provenance(tmp_path, monkeypatch):
    # Build from the original execution authority, before or after its archival.
    old_path = strong.EVIDENCE / "original/authority.json"
    if not old_path.exists():
        old_path = strong.c.AUTHORITY
    old = json.loads(old_path.read_text())
    evidence = tmp_path / "evidence"
    (evidence / "original").mkdir(parents=True)
    (evidence / "original/authority.json").write_bytes(old_path.read_bytes())
    anchor = evidence / "anchor.json"
    anchor.write_text(json.dumps({"accepted_update": 84, "contract_sha256": strong.CONTRACT_HASH}))
    monkeypatch.setattr(strong, "EVIDENCE", evidence)
    monkeypatch.setattr(strong, "ANCHOR", anchor)
    new = strong.authority_payload("a" * 40, "2026-09-05T01:45:00Z")
    for key in (
        "ppo",
        "reward",
        "opponents",
        "reset_curriculum",
        "campaign",
        "source",
        "credit_assignment_amendment",
        "operational_inference_optimization",
    ):
        assert old[key] == new[key]
    assert new["exploration"]["contract"] == strong.CONTRACT
    assert new["campaign"]["maximum_accepted_updates"] == 100


def test_post_transition_checkpoint_cannot_silently_restore_old_noise(tmp_path, monkeypatch):
    anchor = {"accepted_update": 84, "version": strong.VERSION}
    path = tmp_path / "anchor.json"
    path.write_text(json.dumps(anchor))
    monkeypatch.setattr(strong, "ANCHOR", path)
    monkeypatch.setattr(strong.c, "validate_resume_payload", lambda payload: None)
    checkpoint = {
        "accepted_updates_total": 85,
        "phase_transition": {"strong_exploration": anchor},
        "exploration": strong.exploration_for_update(84).as_dict(),
    }
    strong.validate_resume(checkpoint)
    wrong = copy.deepcopy(checkpoint)
    wrong["exploration"] = strong.c.prior.restart_exploration(84).as_dict()
    with pytest.raises(ValueError, match="unexpected exploration"):
        strong.validate_resume(wrong)
    wrong = copy.deepcopy(checkpoint)
    wrong["accepted_updates_total"] = 83
    with pytest.raises(ValueError, match="roll back"):
        strong.validate_resume(wrong)


def test_configure_keeps_original_learning_settings(monkeypatch):
    for name, value in list(vars(strong.c.engine).items()):
        if not name.startswith("__"):
            monkeypatch.setattr(strong.c.engine, name, value)
    strong.configure()
    assert strong.c.engine.exploration_for_update is strong.exploration_for_update
    assert strong.c.engine.CONTINUATION_REVIEW_MARKER == 100
    assert strong.c.engine.ACTIVE_SNAPSHOT_INTERVAL == 50
    assert strong.c.engine.POLICY_LEARNING_RATE == 1e-4
    assert strong.c.engine.CRITIC_LEARNING_RATE == 3e-4
    assert asdict(strong.c.new_ppo_config())["entropy_coefficient"] == 0
