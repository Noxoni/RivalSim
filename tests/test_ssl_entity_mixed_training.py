import copy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from rivalsim.fresh_ground_30hz import FreshGroundEnv, scenarios
from rivalsim.ssl_entity_mixed_training import (
    MixedEntityRolloutCollector,
    family_normalize,
    learner_mask,
    mixed_joint_ppo_update,
    mixed_sequence_data,
)
from rivalsim.ssl_entity_policy import EntityJointControlActorCritic
from rivalsim.ssl_entity_training import (
    EntityRolloutCollector,
    fresh_entity_optimizer,
    joint_ppo_update,
    joint_sequence_loss,
)
from tests.test_ssl_entity_training import fixture


def test_masks_keep_both_current_players_and_only_current_against_nexto():
    result = learner_mask(torch.tensor([False, True, True]), torch.tensor([0, 0, 1]))
    assert result.tolist() == [[True, True], [True, False], [False, True]]


def test_family_normalization_excludes_opponent_outliers():
    advantage = torch.tensor([[1.0, 3.0, 1e6], [100.0, 200.0, -1e6]])
    mask = torch.tensor([[True, True, False], [True, True, False]])
    family = torch.tensor([[0, 0, 99], [1, 1, 99]])
    expected = torch.tensor([[-1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]])
    torch.testing.assert_close(family_normalize(advantage, mask, family), expected, rtol=0, atol=0)


def test_existing_two_evaluation_nexto_schedule_no_early_activation():
    # No model, simulator or Nexto is instantiated to test this state transition.
    collector = object.__new__(MixedEntityRolloutCollector)
    collector.nexto_probability, collector.competence_streak = 0.0, 0
    good = {
        "acquisition_selfplay": {
            "focal_touch_fraction": 0.8,
            "median_first_touch_seconds_if_touched": 1.0,
        },
        "finishing_selfplay": {"goals_for": 35},
    }
    bad = copy.deepcopy(good)
    bad["acquisition_selfplay"]["focal_touch_fraction"] = 0.5
    assert collector.accept_evaluation(good) == 0
    assert collector.accept_evaluation(bad) == 0
    assert collector.accept_evaluation(good) == 0
    assert collector.accept_evaluation(good) == 0.2
    assert collector.accept_evaluation(bad) == 0.2


def test_pure_selfplay_update_is_identical_to_frozen_pilot():
    model, config, rollout = fixture()
    candidate = copy.deepcopy(model)
    rollout.opponent_family = torch.zeros_like(rollout.train_mask, dtype=torch.long)
    old_optimizer, new_optimizer = fresh_entity_optimizer(model), fresh_entity_optimizer(candidate)
    old = joint_ppo_update(model, old_optimizer, rollout, config, torch.Generator().manual_seed(9))
    new = mixed_joint_ppo_update(
        candidate, new_optimizer, rollout, config, torch.Generator().manual_seed(9)
    )
    assert old == new
    assert all(
        torch.equal(value, candidate.state_dict()[key]) for key, value in model.state_dict().items()
    )
    for a, b in zip(old_optimizer.state.values(), new_optimizer.state.values(), strict=True):
        assert all(torch.equal(value, b[key]) for key, value in a.items())


def test_unused_opponent_rewards_and_actions_cannot_affect_update():
    model, config, rollout = fixture()
    rollout.opponent_family = torch.zeros_like(rollout.train_mask, dtype=torch.long)
    rollout.opponent_family[:, 1] = 1
    rollout.train_mask[:, 1, 1] = False
    changed = copy.deepcopy(rollout)
    changed.rewards[:, 1, 1] = 1e6
    changed.action_indices[:, 1, 1] = 88
    for source in (rollout, changed):
        data = mixed_sequence_data(source, config)
        loss, _ = joint_sequence_loss(model, data, torch.arange(4), config)
        assert torch.isfinite(loss)
    copy_model = copy.deepcopy(model)
    original = mixed_joint_ppo_update(
        model, fresh_entity_optimizer(model), rollout, config, torch.Generator().manual_seed(9)
    )
    modified = mixed_joint_ppo_update(
        copy_model,
        fresh_entity_optimizer(copy_model),
        changed,
        config,
        torch.Generator().manual_seed(9),
    )
    assert original == modified
    assert all(
        torch.equal(value, copy_model.state_dict()[key])
        for key, value in model.state_dict().items()
    )


def test_mixed_update_corruption_restores_model_adam_and_shuffle_rng(monkeypatch):
    model, config, rollout = fixture()
    rollout.opponent_family = torch.zeros_like(rollout.train_mask, dtype=torch.long)
    optimizer = fresh_entity_optimizer(model)
    generator = torch.Generator().manual_seed(9)
    mixed_joint_ppo_update(model, optimizer, rollout, config, generator)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    adam = copy.deepcopy(optimizer.state_dict())
    rng = generator.get_state().clone()
    step = optimizer.step

    def corrupt():
        step()
        with torch.no_grad():
            model.entity_actor.weight.fill_(float("nan"))

    monkeypatch.setattr(optimizer, "step", corrupt)
    with pytest.raises(RuntimeError, match="nonfinite_entity_parameter_or_adam"):
        mixed_joint_ppo_update(model, optimizer, rollout, config, generator)
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
    for index, state in adam["state"].items():
        assert all(
            torch.equal(value, optimizer.state_dict()["state"][index][key])
            for key, value in state.items()
        )
    assert torch.equal(rng, generator.get_state())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native CUDA required")
def test_native_rollout_pure_parity_and_nexto_masking(monkeypatch):
    torch.set_num_threads(4)
    collision = Path("G:/dev/RLBot-Rival/bot/collision_meshes")
    if not collision.exists():
        pytest.skip("local arena required")
    model = EntityJointControlActorCritic().cuda()
    bank = scenarios(32, seed=2026090502)
    env_a = FreshGroundEnv(32, collision, device="cuda:0", seed=42, ssl_foundation_scenarios=bank)
    env_b = FreshGroundEnv(32, collision, device="cuda:0", seed=42, ssl_foundation_scenarios=bank)
    original = EntityRolloutCollector(env_a, model, seed=11)
    mixed = MixedEntityRolloutCollector(env_b, copy.deepcopy(model), seed=11)
    original.config = mixed.config = replace(original.config, rollout_horizon=4)
    a, b = original.collect(), mixed.collect()
    for key in (
        "observations",
        "action_indices",
        "old_log_probability",
        "values",
        "rewards",
        "next_values",
        "train_mask",
        "reset_before",
        "terminated",
        "truncated",
    ):
        torch.testing.assert_close(getattr(a, key), getattr(b, key), rtol=0, atol=0)
    assert mixed.nexto is None
    mixed.nexto_probability = 0.2
    mixed.assign(torch.ones(32, device="cuda:0", dtype=torch.bool))
    active, side = mixed.is_nexto.clone(), mixed.side.clone()
    assert 0 < active.sum() < 32
    mixed.assign(torch.zeros_like(active))
    assert torch.equal(active, mixed.is_nexto) and torch.equal(side, mixed.side)
    teacher_before = {key: value.clone() for key, value in mixed.nexto.actor.state_dict().items()}
    # Observe the actual bridge inputs, not just the PPO buffer's ignored slots.
    applied, nexto_controls = [], []
    set_actions, tick_action = env_b.bridge.set_actions, mixed.nexto.tick_action

    def capture_actions(action):
        applied.append(action.clone())
        return set_actions(action)

    def capture_nexto(*args, **kwargs):
        output = tick_action(*args, **kwargs)
        nexto_controls.append(output[0].clone())
        return output

    monkeypatch.setattr(env_b.bridge, "set_actions", capture_actions)
    monkeypatch.setattr(mixed.nexto, "tick_action", capture_nexto)
    buffer = mixed.collect()
    assert len(applied) == 5 * buffer.horizon
    assert len(nexto_controls) == 4 * buffer.horizon
    for decision in range(buffer.horizon):
        mask = buffer.train_mask[decision]
        for tick in range(4):
            action = applied[decision * 5 + 1 + tick]
            assert torch.equal(action[mask], buffer.actions[decision][mask])
            assert torch.equal(
                action[active, 1 - side[active]],
                nexto_controls[decision * 4 + tick][active],
            )
    assert torch.equal(buffer.train_mask[0], learner_mask(active, side))
    assert not buffer.train_mask[0][active, 1 - side[active]].any()
    assert mixed.nexto.inference_calls > 0
    assert mixed.last_metrics["nexto_training_sample_count"] > 0
    assert mixed.last_metrics["trainable_agent_samples"] == int(buffer.train_mask.sum())
    assert sum(mixed.last_metrics["action_index_counts"]) == int(buffer.train_mask.sum())
    assert torch.equal(
        model.action_table[buffer.action_indices][buffer.train_mask],
        buffer.actions[buffer.train_mask],
    )
    assert all(
        torch.equal(value, mixed.nexto.actor.state_dict()[key])
        for key, value in teacher_before.items()
    )
    assert all(
        parameter.grad is None and not parameter.requires_grad
        for parameter in mixed.nexto.actor.parameters()
    )
