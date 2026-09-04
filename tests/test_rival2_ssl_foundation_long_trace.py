from __future__ import annotations

import ast
import copy
import json
from dataclasses import asdict

import pytest
import torch

from benchmarks import run_rival2_ssl_foundation_v5_long_trace_v1 as campaign
from rivalsim.rival2_ppo import compute_gae_gpu
from rivalsim.rival2_recurrent_ppo import Rival2RecurrentRolloutBuffer, _sequence_major


def test_only_horizon_and_lambda_change_and_gamma_is_preserved():
    old = asdict(campaign.prior.amended.new_ppo_config())
    new = asdict(campaign.new_ppo_config())
    assert {key for key in new if old[key] != new[key]} == {"rollout_horizon", "gae_lambda"}
    assert new["rollout_horizon"] == 360
    assert 0 < new["gae_lambda"] < 1
    assert (new["gamma"] * new["gae_lambda"]) ** 360 == pytest.approx(0.5)
    assert (
        campaign.new_ppo_config().content_hash
        != campaign.prior.amended.new_ppo_config().content_hash
    )


@pytest.mark.parametrize("sign", (-1.0, 1.0))
def test_delayed_outcome_impulse_preserves_sign_and_half_life(sign):
    config = campaign.new_ppo_config()
    rewards = torch.zeros(361, 2, dtype=torch.float64)
    rewards[-1] = sign * 10
    zero = torch.zeros_like(rewards)
    terminated = torch.zeros_like(rewards, dtype=torch.bool)
    terminated[-1] = True
    advantage, returns = compute_gae_gpu(
        rewards,
        zero,
        zero,
        terminated,
        torch.zeros_like(terminated),
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
    )
    assert torch.allclose(advantage[0], torch.full((2,), sign * 5, dtype=torch.float64))
    assert torch.equal(advantage, returns)
    old = campaign.prior.amended.new_ppo_config()
    previous, _ = compute_gae_gpu(
        rewards,
        zero,
        zero,
        terminated,
        torch.zeros_like(terminated),
        gamma=old.gamma,
        gae_lambda=old.gae_lambda,
    )
    assert (advantage[0] / previous[0]).min() > 79


@pytest.mark.parametrize("kind", ("terminal", "truncation"))
def test_trace_does_not_cross_episode_boundary_and_truncation_bootstraps(kind):
    config = campaign.new_ppo_config()
    reward = torch.zeros(360, 1, dtype=torch.float64)
    reward[-1] = 10
    zero = torch.zeros_like(reward)
    next_value = zero.clone()
    next_value[100] = 3
    terminal = torch.zeros_like(reward, dtype=torch.bool)
    truncated = terminal.clone()
    (terminal if kind == "terminal" else truncated)[100] = True
    adv, _ = compute_gae_gpu(
        reward,
        zero,
        next_value,
        terminal,
        truncated,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
    )
    expected = 0 if kind == "terminal" else 3 * config.gamma
    assert adv[100].item() == pytest.approx(expected)
    assert adv[99].item() == pytest.approx(expected * config.gamma * config.gae_lambda)
    assert adv[101].item() > 0


def test_complete_360_tick_sequences_no_frame_shuffle_or_bulk_observation_copy():
    buffer = Rival2RecurrentRolloutBuffer(360, 2, torch.zeros(1, 2, 2, 256), "cpu")
    obs = buffer.observations
    obs.copy_(torch.arange(obs.numel()).reshape_as(obs))
    sequence = _sequence_major(obs)
    assert sequence.shape == (4, 360, 182)
    assert sequence.untyped_storage().data_ptr() == obs.untyped_storage().data_ptr()
    assert torch.equal(sequence[3], obs[:, 1, 1])
    assert buffer.sequence_layout(65536).sequences_per_minibatch == 182
    assert 182 * 360 == 65520


def test_authority_preserves_reward_opponents_and_existing_bound():
    old = json.loads(campaign.prior.AUTHORITY.read_text())
    new = campaign.authority_payload("a" * 40, "2026-09-04T00:00:00Z")
    for key in ("reward", "source", "reset_curriculum", "opponents", "exploration", "campaign"):
        assert old[key] == new[key]
    assert new["campaign"]["maximum_accepted_updates"] == 100
    assert new["credit_assignment_amendment"]["parent_update"] == 10
    assert not new["ppo"]["fresh_optimizer"]


def test_checkpoint_migration_preserves_every_learning_tensor_counter_and_rng(monkeypatch):
    parent = torch.load(campaign.PARENT, map_location="cpu", weights_only=False)
    assert campaign.engine.sha256_file(campaign.PARENT) == campaign.PARENT_SHA256
    real_sha = campaign.engine.sha256_file
    monkeypatch.setattr(
        campaign.engine,
        "sha256_file",
        lambda p: (
            "A" * 64
            if p == campaign.AUTHORITY
            else "B" * 64
            if p == campaign.LAUNCH
            else real_sha(p)
        ),
    )
    migrated = campaign.migrate_payload(parent, "A" * 64, "B" * 64)
    changed = {
        key
        for key in parent
        if campaign.tree_sha256(parent[key]) != campaign.tree_sha256(migrated[key])
    }
    assert changed == {
        "format",
        "lineage",
        "source",
        "phase_transition",
        "ppo_config",
        "ppo_config_sha256",
    }
    campaign.validate_resume_payload(migrated)
    for key, value in (
        ("ppo_config", parent["ppo_config"]),
        ("optimizer", {"state": {}}),
        ("accepted_updates_total", 101),
        ("lineage", "wrong"),
    ):
        bad = {**migrated, key: value}
        with pytest.raises(ValueError):
            campaign.validate_resume_payload(bad)
    wrong = copy.deepcopy(parent)
    wrong["accepted_updates_total"] = 11
    with pytest.raises(ValueError, match="exact saved update-10"):
        campaign.migrate_payload(wrong, "A" * 64, "B" * 64)


def test_finished_rollout_released_before_next_collection():
    tree = ast.parse((campaign.ROOT / "benchmarks/run_rival2_ssl_foundation_ppo_v1.py").read_text())
    assert any(
        isinstance(node, ast.Delete)
        and any(isinstance(target, ast.Name) and target.id == "rollout" for target in node.targets)
        for node in ast.walk(tree)
    )


def test_runtime_uses_three_second_config_without_unlimited_extension(monkeypatch):
    for name, value in list(vars(campaign.engine).items()):
        if not name.startswith("__"):
            monkeypatch.setattr(campaign.engine, name, value)
    campaign.configure_engine()
    assert campaign.engine.CONTINUATION_REVIEW_MARKER == 100
    assert campaign.engine.EVALUATION_TICKS == 3600
    assert campaign.engine.ACTIVE_SNAPSHOT_INTERVAL == 50
    args = campaign.parser().parse_args(["--continue-after-600"])
    with pytest.raises(ValueError, match="bound"):
        campaign.run(args)
